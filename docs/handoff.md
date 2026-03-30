# Handoff

## Scope

This is the operational handoff for `frankenstein_base` as of March 30, 2026.

It captures:

- what is implemented
- what was validated
- what is still coupled to sibling repositories
- what is most likely to break on another machine

## What is implemented

`frankenstein_base` currently supports:

- base-scene and split-scene training
- naive LoD training via `train_nomask.py`
- experiment launching via `run_exp.py`
- optional EDGS / RoMa initialization through explicit `edgs_*` flags
- explicit `--densify` / `--no-densify`

The EDGS integration is live, not just planned.

## Main integration points

Relevant files:

- `run_exp.py`
- `train_nomask.py`
- `scene/__init__.py`
- `scene/gaussian_model.py`
- `edgs_init.py`

Functional flow:

1. `run_exp.py` builds the experiment command
2. `train_nomask.py` parses LoD, EDGS, densification, W&B, and split settings
3. `Scene(...)` initializes the base Gaussian set from COLMAP
4. if EDGS is enabled, `edgs_init.py` appends RoMa-triangulated Gaussians
5. split extension blocks are preloaded the same way when requested

## Important behavior decisions

### LoD scale semantics

The experiment dict is now the authority for scale.

Current runner rule:

- always pass `-r 1`
- use `resolution_scales` as the real LoD resolution levels

This fixed the previous double-downsampling bug.

### Densification semantics

`--no-densify` is implemented in the real optimization path.

When disabled:

- densification stats are not accumulated
- `densify_and_prune(...)` is not called
- densification-phase opacity resets are skipped

### W&B and viewer defaults

Current runner defaults:

- W&B enabled
- viewer disabled

This was chosen because:

- W&B is useful for experiment tracking
- the viewer caused port conflicts in runner-launched jobs

## What was validated

Validation was performed in conda env `frankenstein`.

Confirmed imports:

- `torch`
- `numpy`
- `scipy`
- `PIL`
- `plyfile`
- `wandb`
- `simple_knn._C`
- `diff_gaussian_rasterization`
- `romatch`
- `roma_outdoor`
- `roma_indoor`

Smoke-tested scenes:

- `home_base/model0`
- `home_split2/model0`

Observed results:

- one-iteration no-densify runs completed successfully
- split-scene loading worked for `home_split2/model0`
- EDGS-enabled initialization was brought to a working state after fixing two runtime issues

## Runtime issues already fixed

The following issues were encountered and patched:

### EDGS correspondence shape mismatch

File:

- `EDGS/source/corr_init.py`

Fix:

- triangulation batching now uses the actual sampled keypoint count instead of fixed `matches_per_ref`

### `tmp_radii` initialization / device mismatch

File:

- `scene/gaussian_model.py`

Fix:

- initialize `tmp_radii` in `GaussianModel.__init__`
- ensure appended radii and fallback tensors are moved to the active Gaussian device

### Eval metric return bug

File:

- `train_nomask.py`

Fix:

- initialize eval metric locals before conditional assignment

### Double-downsampling in runner

File:

- `run_exp.py`

Fix:

- `resolution_scales` is now the single source of truth for LoD scale

## Current non-self-contained dependencies

`frankenstein_base` is not fully self-contained for EDGS mode.

The main coupling is in `edgs_init.py`:

- it expects a sibling repo `../EDGS`
- it expects `../EDGS/submodules/RoMa`
- it adds both to `sys.path`
- it imports `EDGS/source/corr_init.py`

Implication:

- cloning `frankenstein_base` alone is enough for vanilla mode only if its own submodules and environment are installed
- cloning `frankenstein_base` alone is not enough for `--edgs_init`
- EDGS mode currently requires the larger repo layout with sibling `EDGS`

## Other machine requirements

Another machine still needs:

- CUDA-compatible PyTorch
- compiled `frankenstein_base/submodules/diff-gaussian-rasterization`
- compiled `frankenstein_base/submodules/simple-knn`
- the environment dependencies from `environment.yml` or equivalent
- EDGS repo and RoMa submodule if EDGS mode is needed

## Known weak spots

### Silent runner failure handling

`run_exp.py` currently catches subprocess exceptions and continues.

That is weak for debugging because failed training children do not stop the full batch loudly.

### EDGS portability

The current EDGS bridge is operational but not portable.

It should eventually be replaced by one of:

- a local port of the needed EDGS initialization code into `frankenstein_base`
- a documented vendored RoMa dependency plus local `roma_init/*`

### Split-path assumptions

The split scene path contract is rigid.

Paths must match the expected `_base` and `_splitN/model0` conventions for the runner to infer extension behavior correctly.

## Recommended next step

The next structural improvement should be:

1. remove runtime imports from sibling `EDGS/source/*`
2. vendor or port the minimal RoMa-based initialization path into `frankenstein_base`
3. keep the experiment dict as the only authority for LoD scale

## Minimal handoff commands

Vanilla run:

```bash
conda run -n frankenstein python train_nomask.py \
  -s ../home_base/model0 \
  -m /tmp/frankenstein_smoke_base \
  --iterations 1 \
  --disable_viewer \
  --disable_wandb \
  --no-densify
```

Split run:

```bash
conda run -n frankenstein python train_nomask.py \
  -s ../home_split2/model0 \
  -m /tmp/frankenstein_smoke_split \
  --iterations 1 \
  --disable_viewer \
  --disable_wandb \
  --no-densify \
  -x 1
```

Runner launch:

```bash
conda run -n frankenstein python run_exp.py \
  --base_source ../home_base/model0 \
  --split_source ../home_split2/model0 \
  --final_extension_iteration 7500 \
  --edgs_init \
  --no-densify
```
