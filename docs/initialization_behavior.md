# Frankenstein Base Initialization Behavior

## Scope

This document describes how Gaussian initialization currently works in `frankenstein_base`.

Relevant files:

- `train_nomask.py`
- `scene/__init__.py`
- `scene/gaussian_model.py`
- `run_exp.py`

It describes the code as it exists now.

## High-level summary

`frankenstein_base` supports two initialization modes:

- plain SfM / COLMAP point-cloud initialization
- EDGS-style RoMa correspondence initialization layered on top of the point-cloud seed

The EDGS path is enabled with explicit `edgs_*` arguments.

Primary entrypoints:

- `run_exp.py`
- `train_nomask.py`
- `edgs_init.py`

The effective initialization behavior today is:

- if `--edgs_init` is not set:
  - base scene: initialize from the input scene's COLMAP point cloud
  - split scene: initialize the first block from `model0`, and precompute extension blocks from sibling `model1`, `model2`, and so on
  - packet scene: initialize from the packet loader's generated seed point cloud
- if `--edgs_init` is set:
  - base scene: initialize from the point cloud, then run EDGS / RoMa correspondence initialization
  - split scene: initialize `model0` the same way, and optionally do the same for precomputed extension blocks
  - packet scene: initialize from the generated packet seed, then run EDGS / RoMa correspondence initialization on the retained packet camera set

## Base-scene initialization path

The active base initialization path in `train_nomask.py` is:

1. create `GaussianModel`
2. create `Scene`
3. let `Scene` initialize Gaussians from the scene point cloud
4. call `gaussians.training_setup(opt)`
5. enter the training loop

Code seam:

- `gaussians = GaussianModel(...)`
- `scene = Scene(...)`
- `gaussians.training_setup(opt)`

Inside `Scene.__init__(...)`:

- the input source scene is read with the normal COLMAP / Blender loader
- packet exports with `packets.jsonl` use the OpenVINS packet loader instead
- train and test cameras are built for each configured resolution scale
- when not resuming from checkpoint, `self.gaussians.create_from_pcd(...)` is called
- when EDGS init is enabled, the scene then runs a local EDGS bridge after an initial `training_setup(...)`

That `create_from_pcd(...)` call is the actual initializer.

When EDGS is enabled, it is the seed initializer, not the final initializer.

For packet-backed scenes, the current seed point cloud comes from:

- triangulated packet sparse tracks when enough multi-view geometry is available
- otherwise, a deterministic forward-ray fallback over the retained packet camera set

Current packet-specific preprocessing before initialization:

- optional dataset-level packet subsampling through `--packet_stride` / `--packet_offset`
- packet `radtan` image undistortion into a rectified pinhole model
- optional packet image mirroring through `--packet_flip_lr` / `--packet_flip_ud`
- rectified packet sparse-track rays for seed-cloud construction

## What `create_from_pcd(...)` sets

`scene/gaussian_model.py` builds the initial tensors from the COLMAP point cloud:

- `xyz`: point positions from the point cloud
- `features_dc`: SH DC term from point colors
- `features_rest`: zeros
- `scaling`: derived from nearest-neighbor point distances
- `rotation`: initialized to identity quaternion form
- `opacity`: initialized from a fixed inverse-sigmoid value

It also initializes exposure state for every camera in the loaded camera set.

This is standard 3DGS-style initialization.

If the CUDA `simple-knn` distance kernel fails during this initial scale estimation, the active code falls back to a chunked `torch.cdist` nearest-neighbor estimate. This has been observed on the packet port with a tiny seed cloud and is treated as a local extension/runtime compatibility issue rather than real model-size VRAM pressure.

## EDGS / RoMa initialization path

`frankenstein_base` now wires EDGS initialization through [edgs_init.py](/home/christoa/Workspace/splatting/frankenstein/frankenstein_base/edgs_init.py).

Current design:

- `edgs_init.py` adds the local EDGS and RoMa paths to `sys.path`
- it imports `EDGS/source/corr_init.py` directly
- it calls either:
  - `init_gaussians_with_corr_fast(...)` when `edgs_nns_per_ref == 1`
  - `init_gaussians_with_corr(...)` otherwise

The bridge applies EDGS initialization after `GaussianModel.training_setup(...)` because EDGS appends new Gaussians through `densification_postfix(...)`, which depends on optimizer-backed append utilities.

Current EDGS post-processing mirrors EDGS behavior:

- optional pruning of the original SfM seed when `--edgs_add_sfm_init` is not set
- scale shrink by multiplying current activated scaling by `0.5` and reapplying the inverse activation

So with `--edgs_init`, the actual order is:

1. initialize from COLMAP point cloud
2. set up optimizer state
3. run EDGS / RoMa correspondence initialization
4. optionally prune original SfM points
5. shrink Gaussian scales
6. later, `train_nomask.py` calls `training_setup(opt)` again before the main loop starts

That final reset is acceptable because initialization happens before training and no optimizer moments need to be preserved yet.

Fresh runs save the initialized Gaussian state before the first optimizer step:

- `point_cloud/iteration_0/point_cloud.ply`

For EDGS runs, this file is the main artifact for checking whether bad geometry comes from initialization or from later training.

## Split-scene initialization path

When `xtend > 0`, `Scene.__init__(...)` prepares additional Gaussian blocks in advance.

Current behavior:

- `model0` is loaded from the source path passed to training
- sibling directories `model1`, `model2`, ... are searched under the same parent directory
- each sibling block gets its own camera lists and its own GaussianModel initialized from that block's point cloud
- these precomputed blocks are stored in `self.x_gauss`
- the matching camera sets are stored in `self.extension_set`

This means split initialization is eager, not lazy.

When EDGS is disabled:

- extension blocks remain pure point-cloud initialization

When EDGS is enabled and `--edgs_init_extensions` is active:

- each extension block is initialized from its point cloud
- then the same EDGS / RoMa bridge is applied to that block before it is stored in `self.x_gauss`

## What happens during `scene.extend()`

Later in training, when the split extension trigger fires:

- cameras from the next extension block are appended into the active train and test camera sets
- the corresponding prebuilt Gaussian block is merged into the active GaussianModel using `concat_new_gaussian(...)`

`concat_new_gaussian(...)` does not just copy tensors directly.

It:

- samples offsets using the new block's scaling and rotation
- generates new positions from those samples
- carries over SH features, opacity, rotation, and scaling state
- appends them through `densification_postfix(...)`

So even split extension is still routed through the same append machinery used by densification-style tensor growth.

## Relationship to LoD

Initialization and LoD are separate concerns.

Initialization decides:

- which Gaussians exist at the start
- which additional Gaussian blocks are available for split extension

LoD decides:

- which image resolution scale is sampled at a given iteration

The current LoD path does not modify initialization.

It only changes viewpoint/image scale selection during training.

## Current EDGS arguments

`train_nomask.py` currently accepts:

- `--edgs_init`
- `--edgs_matches_per_ref`
- `--edgs_num_refs`
- `--edgs_nns_per_ref`
- `--edgs_scaling_factor`
- `--edgs_proj_err_tolerance`
- `--edgs_roma_model`
- `--edgs_add_sfm_init`
- `--edgs_init_extensions` / `--no-edgs_init_extensions`

`run_exp.py` exposes the same family so the experiment runner can launch EDGS-enabled runs directly.

## Validation status

Validation was run in the `frankenstein` conda environment against:

- `home_split2/model0`

Observed behavior:

- core compiled dependencies import successfully
- RoMa imports successfully
- the training entrypoint runs through scene loading and Gaussian initialization
- the split-scene initialization path works for `home_split2/model0`
- the evaluation helper bug where `training_report(...)` could return uninitialized metrics has been fixed
- direct EDGS bridge import was validated successfully

An EDGS-enabled one-iteration dry run was also started with small `edgs_*` settings, and the run created its output directory and initialization artifacts. In practice, EDGS-enabled startup is much heavier because it pulls in RoMa model initialization.

## Practical interpretation

If you run `frankenstein_base` today, the initialization is:

- SfM-only when `--edgs_init` is off
- SfM-seeded plus EDGS / RoMa correspondence augmentation when `--edgs_init` is on
- optionally multi-block when using split scenes
- appended into the main model via the Gaussian append utilities

## Main seams

The EDGS integration now touches exactly these seams:

- base block initialization in `Scene.__init__(...)`
- extension block initialization in `Scene.create_2nd_set(...)`
- optional pruning of original SfM points after RoMa append in `edgs_init.py`

Those are the places where the point-cloud seed is established and optionally augmented by EDGS.
