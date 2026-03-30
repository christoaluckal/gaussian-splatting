# RoMa Integration Plan

## Goal

Integrate EDGS-style RoMa-based Gaussian initialization into the copied `frankenstein_base` pipeline without changing EDGS itself.

Primary future targets:

- `frankenstein_base/train_nomask.py`
- `frankenstein_base/run_exp.py`

Related likely touch points:

- `frankenstein_base/arguments/__init__.py`
- `frankenstein_base/scene/__init__.py`
- `frankenstein_base/scene/gaussian_model.py`
- new local helper modules under `frankenstein_base`

## Current baseline in `frankenstein_base`

Today the initialization sequence in `frankenstein_base/train_nomask.py` is:

1. build `GaussianModel`
2. build `Scene`
3. let `Scene` call `gaussians.create_from_pcd(...)`
4. call `gaussians.training_setup(opt)`
5. start optimization

For split scenes:

- `Scene` pre-creates extension Gaussian sets from `model1`, `model2`, ...
- `scene.extend()` later concatenates those prebuilt Gaussian blocks and appends their cameras

This means `frankenstein_base` currently has three initialization regimes:

- vanilla base scene initialization from `model0` COLMAP point cloud
- incremental split initialization through later `scene.extend()`
- LoD scheduling layered on top of either of the above

RoMa initialization must fit all three.

## Integration target behavior

### Vanilla

For base training on `model0`:

- optionally run RoMa-based initialization after `Scene(...)` is constructed and before the main optimization loop
- choose whether to keep or discard the original SfM seed Gaussians

### Incremental

For split training:

- optionally run RoMa-based initialization for the initial `model0` block
- optionally run the same initialization for each extension block before or during `scene.extend()`
- ensure camera indexing and viewpoint scheduling remain consistent after extension

### LoD

LoD should remain orthogonal:

- RoMa affects which Gaussians exist
- LoD affects which resolution of images is sampled during optimization

The initialization should therefore happen once per scene block, before that block participates in training, independent of the current LoD scale schedule.

## Recommended architecture

## 1. Add a local initialization module

Create a local module namespace in `frankenstein_base`, for example:

- `frankenstein_base/roma_init/__init__.py`
- `frankenstein_base/roma_init/corr_init.py`
- `frankenstein_base/roma_init/config.py` if needed

Reason:

- keeps the future implementation inside the target tree
- avoids runtime imports from `EDGS/source`
- allows controlled adaptation to the local `GaussianModel` and `Scene` APIs

## 2. Add explicit CLI/config controls to `train_nomask.py`

Add future arguments for:

- `--roma_init`
- `--roma_matches_per_ref`
- `--roma_num_refs`
- `--roma_nns_per_ref`
- `--roma_proj_err_tolerance`
- `--roma_scaling_factor`
- `--roma_model` with `indoors|outdoors`
- `--roma_keep_sfm_init` or inverse form
- possibly `--roma_init_extensions`

Reason:

- EDGS currently exposes these through Hydra
- `frankenstein_base` uses argparse and should keep a native CLI surface

## 3. Keep initialization as a pre-training phase

Preferred order in future `train_nomask.py`:

1. construct `gaussians`
2. construct `scene`
3. if RoMa init enabled, call a local `init_gaussians_with_corr(...)`
4. call `gaussians.training_setup(opt)` or rebuild optimizer state as needed
5. enter the existing training loop

This mirrors EDGS and keeps the feature isolated from the training loop.

## 4. Treat extensions as scene-block initialization events

There are two viable designs for split extension support.

Option A: precompute extension RoMa initialization during `Scene` construction

- when `Scene.create_2nd_set(...)` loads `modelN`, also build a Gaussian block initialized with RoMa for that block
- `scene.extend()` keeps concatenating ready-made Gaussian blocks

Pros:

- aligns with current `Scene` design
- keeps the training loop unchanged except for the existing `scene.extend()` call

Cons:

- front-loads RoMa cost before training starts
- requires `Scene` to own more initialization policy

Option B: initialize extension blocks lazily at extension time

- `Scene` stores raw extension scene/camera metadata
- when `scene.extend()` is triggered, build RoMa-based Gaussians for that new block then concatenate them

Pros:

- cost is aligned with the extension event
- easier to reason about block-local initialization

Cons:

- more moving parts during training
- extension timing must remain robust

Recommended first implementation:

- start with Option A because it matches the current `Scene` ownership model

## 5. Keep LoD logic unchanged initially

Do not entangle RoMa with:

- `_resolve_naive_lod_scale(...)`
- `_maybe_update_naive_lod_scale(...)`
- `match_resolution`

The only LoD-related compatibility requirement is:

- any newly added cameras from a split extension must still be available at all configured resolution scales

That already exists in the current `Scene` loading path.

## Minimum viable implementation plan

### Phase 1: base-scene RoMa initialization only

Files to change later:

- `frankenstein_base/train_nomask.py`
- `frankenstein_base/arguments/__init__.py`
- new `frankenstein_base/roma_init/*`

Behavior:

- enable RoMa initialization for `model0`
- keep split extension behavior unchanged
- keep LoD behavior unchanged

Why first:

- lowest-risk path
- validates imports, RoMa dependency management, and Gaussian append semantics

### Phase 2: split extension initialization

Files to change later:

- `frankenstein_base/scene/__init__.py`
- possibly `frankenstein_base/scene/gaussian_model.py`
- `frankenstein_base/train_nomask.py`
- local `roma_init/*`

Behavior:

- initialize extension Gaussian blocks with RoMa rather than pure PCD-only `create_from_pcd(...)`
- preserve current viewpoint-block reset behavior after `scene.extend()`

### Phase 3: runner support

Files to change later:

- `frankenstein_base/run_exp.py`

Behavior:

- propagate RoMa init flags into experiment commands
- define experiment naming conventions that distinguish:
  - plain baseline
  - RoMa-initialized baseline
  - RoMa-initialized split
  - RoMa + LoD variants

## Concrete seams to target

### Seam A: initial Gaussian creation in `train_nomask.py`

Current seam:

- `gaussians = GaussianModel(...)`
- `scene = Scene(...)`
- `gaussians.training_setup(opt)`

Future behavior:

- insert local RoMa initializer between scene creation and optimizer setup, or rerun setup afterward if append logic needs optimizer rebuild

### Seam B: split block creation in `scene/__init__.py`

Current seam:

- base block created from `scene_info.point_cloud`
- extension blocks created in `create_2nd_set(...)`
- `scene.extend()` concatenates extension Gaussian blocks

Future behavior:

- allow each block to choose between:
  - PCD-only initialization
  - PCD + RoMa augmentation
  - RoMa-only initialization if SfM seed removal is requested

### Seam C: experiment command generation in `run_exp.py`

Current seam:

- runner only knows about base/split and LoD settings

Future behavior:

- runner should surface a second axis for initialization mode

Suggested future experiment labels:

- `baseline`
- `roma-baseline`
- `naive-lod`
- `roma-naive-lod`
- `matched-naive-lod`
- `roma-matched-naive-lod`

This is only a naming suggestion for now.

## Open design questions

These should be settled before code changes start:

1. Should RoMa initialization augment the COLMAP seed or replace it by default?
2. Should split extension blocks use RoMa initialization eagerly during scene construction or lazily during `scene.extend()`?
3. Should the first integration port only the full EDGS path, only the fast path, or both?
4. Should RoMa dependencies live as a vendored module inside `frankenstein_base`, a shared repo-level dependency, or a git submodule?
5. How much of EDGS reprojection-error filtering should be preserved verbatim versus simplified for the first pass?

## Recommended answers for the first implementation

- default to augmenting SfM rather than replacing it
- initialize split extension blocks eagerly during scene construction
- port the fast path first, then the multi-neighbor path
- keep RoMa access local to `frankenstein_base` runtime imports, not via EDGS paths
- preserve reprojection-error filtering because it directly controls initialization quality

## Acceptance criteria for the future implementation

The future code change should be considered minimally complete when:

- `frankenstein_base/train_nomask.py` can run with and without RoMa initialization from CLI flags
- base-scene training works without importing `EDGS/source/*`
- split training can extend with RoMa-initialized blocks or explicitly documented fallback behavior
- `frankenstein_base/run_exp.py` can launch RoMa-enabled variants
- docs in this directory are updated to describe the implemented behavior rather than the plan
