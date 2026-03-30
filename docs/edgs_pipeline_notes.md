# EDGS Pipeline Notes

## Scope

This document captures what the active EDGS pipeline is doing today, with emphasis on the parts that matter for porting its initialization logic into `frankenstein_base`.

Primary entrypoints inspected:

- `EDGS/run_exp.py`
- `EDGS/train.py`
- `EDGS/source/trainer.py`
- `EDGS/source/corr_init.py`
- `EDGS/source/networks.py`
- `EDGS/configs/train.yaml`

## High-level behavior

EDGS is not a separate renderer or scene format. It wraps a forked `gaussian-splatting` stack and changes the initial Gaussian population before normal optimization proceeds.

The important distinction is:

- vanilla 3DGS initializes Gaussians from the COLMAP point cloud via `create_from_pcd(...)`
- EDGS still builds the same scene first, but then injects many more Gaussians from RoMa correspondences triangulated across training-image pairs

EDGS therefore behaves more like an initialization module plus a training recipe than a fundamentally different training loop.

## What `EDGS/run_exp.py` does

`EDGS/run_exp.py` is a batch launcher. It does not contain the RoMa logic itself.

Its current role is:

- define a grid of scene names and initialization/training settings
- construct an experiment name and output path per configuration
- shell out to `python train.py ...` with Hydra overrides

The launch parameters that directly control RoMa-based initialization are:

- `init_wC.matches_per_ref`
- `init_wC.nns_per_ref`
- `init_wC.num_refs`
- `init_wC.exp_name`

It also toggles EDGS / VGS / vanilla modes through `gs.vgs.*`, but the RoMa path is driven through `init_wC`.

## What `EDGS/train.py` does

`EDGS/train.py` is the real training entrypoint.

Sequence:

1. Hydra loads `configs/train.yaml`.
2. The output directory and `cfg_args` file are prepared.
3. `cfg.gs` is instantiated into a `Warper3DGS`.
4. `EDGSTrainer` is created around that object.
5. `trainer.init_with_corr(cfg.init_wC, cfg.gs.opt)` runs before the main optimization loop.
6. `trainer.train(cfg.train)` performs the normal training iterations.

The important implication for a port:

- EDGS keeps initialization as a distinct pre-training phase
- it does not intermingle RoMa matching with every optimization step

## What `Warper3DGS` contributes

`EDGS/source/networks.py` shows that EDGS still constructs:

- a `GaussianModel`
- a `Scene`
- a viewpoint stack
- the standard renderer / pipeline state

This is useful because `frankenstein_base` already has the same basic objects.

## What `EDGSTrainer.init_with_corr(...)` does

`EDGSTrainer.init_with_corr(...)` is the main bridge between the normal 3DGS setup and EDGS initialization.

Behavior:

- exits early when `cfg.use` is false
- records the current Gaussian count, which corresponds to the SfM initialization
- chooses `init_gaussians_with_corr_fast` when `nns_per_ref == 1`
- otherwise uses `init_gaussians_with_corr`
- calls the chosen initialization routine with:
  - current `GaussianModel`
  - current `Scene`
  - `cfg.init_wC`
  - optimization config
  - device
  - optional pre-instantiated RoMa model
- refreshes the optimizer reference afterward
- optionally removes the SfM initialization logically when `add_SfM_init` is false
- halves the current Gaussian scales after initialization

Observations that matter for integration:

- EDGS mutates an already-created Gaussian set in place
- it expects the Gaussian model to support appending new Gaussians after initial setup
- it treats RoMa initialization as compatible with later pruning / densification behavior

## What `source/corr_init.py` actually does

### Camera selection

For the full path `init_gaussians_with_corr(...)`:

- obtains train cameras from `scene.getTrainCameras()`
- selects up to `num_refs` reference cameras using K-means over flattened camera transforms
- computes nearest-neighbor cameras using Euclidean distance over those flattened transforms

This means EDGS does not match all image pairs. It reduces pair count with:

- reference view subsampling
- nearest-neighbor pairing

### RoMa inference

RoMa model selection:

- `roma_indoor(...)` when `cfg.roma_model == "indoors"`
- otherwise `roma_outdoor(...)`

Runtime settings:

- `upsample_preds = False`
- `symmetric = False`

EDGS performs a warm-up `roma_model.match(...)` call once before the real loop.

### Match sampling and triangulation

For each selected reference view:

1. Compute RoMa warp and certainty against each nearest-neighbor view.
2. Aggregate per-pixel confidence across neighbors.
3. Sample up to `matches_per_ref` match candidates from the certainty map.
4. Convert sampled normalized coordinates into image pixel coordinates.
5. Triangulate 3D points with the paired camera projection matrices.
6. Score by reprojection error and keep the best triangulated points.

### Gaussian construction

For every accepted triangulated point, EDGS derives:

- `xyz`: triangulated 3D position
- `features_dc`: SH DC term from source image color
- `features_rest`: zeros
- `opacity`: copied from an existing template Gaussian, optionally pushed near invisible for bad reprojection fits
- `scaling`: based on distance from the reference camera times `cfg.scaling_factor`
- `rotation`: copied from an existing template Gaussian

After accumulating all reference-view contributions, EDGS appends them with `gaussians.densification_postfix(...)`.

That is the critical integration seam.

### Fast path

`init_gaussians_with_corr_fast(...)` is a cheaper variant used when `nns_per_ref == 1`.

Differences:

- only one neighbor per reference view
- uses `roma_model.match(...)` directly instead of multi-neighbor aggregation
- triangulates sampled correspondences from a single pair

The fast path is simpler to port first if a staged integration is preferred.

## EDGS configuration surface

From `configs/train.yaml`, the relevant `init_wC` parameters are:

- `use`
- `matches_per_ref`
- `num_refs`
- `nns_per_ref`
- `proj_err_tolerance`
- `add_SfM_init`
- `scaling_factor`
- `roma_model`
- `exp_name`

These parameters are a good starting point for a matching CLI surface in `frankenstein_base`.

## EDGS assumptions that are currently non-local

The EDGS implementation depends on repository-local path hacks:

- `sys.path.append('./submodules/gaussian-splatting/')`
- `sys.path.append('../submodules/RoMa')`

It also assumes:

- RoMa is installed or importable from the vendored submodule
- EDGS can call into its forked `gaussian-splatting` implementation directly

For `frankenstein_base`, these assumptions should not be carried over as-is.

## Porting implications

The EDGS initialization logic is conceptually portable because it relies on:

- train camera tensors and poses already available from `Scene`
- Gaussian append semantics already available from `GaussianModel`
- standard 3DGS training after initialization

The non-portable parts are mainly:

- import layout
- EDGS-specific wrapper classes
- direct dependence on the EDGS repo tree for RoMa code

The recommended port shape is therefore:

- keep the RoMa-based initialization as a pre-training phase
- implement it as local `frankenstein_base` modules
- expose it through `frankenstein_base/train_nomask.py` and `frankenstein_base/run_exp.py`
- avoid runtime dependence on `EDGS/source/*`
