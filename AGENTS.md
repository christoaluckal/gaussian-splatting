# gaussian-splatting Notes

## Purpose

Use this file as the quick handoff for the local training behavior in this folder.

The main customization here is in [`train_nomask.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/train_nomask.py), which extends the vanilla trainer with incremental Gaussian-set growth and deterministic fixed-iteration LoD scheduling.

## Start here

If the task is about training behavior, begin with:

- [`train_nomask.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/train_nomask.py)
- [`train.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/train.py)
- [`scene/__init__.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/scene/__init__.py)
- [`utils/camera_utils.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/utils/camera_utils.py)
- [`run_exp.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/run_exp.py)
- [`docs/current_runner_behavior.md`](/mnt/share/nas/christo/splatting/gaussian-splatting/docs/current_runner_behavior.md)

## Current `train_nomask.py` behavior

- The script keeps the incremental-Gaussians extension path via `scene.extend()`.
- Extension is controlled by `--splitter_itr` and skipped when `--default` is enabled.
- The script supports deterministic LoD training through:
  - `--resolution_scales`
  - `--naive_lod_stage_iterations`
- Training order is coarse-to-fine by reversing `--resolution_scales` internally.
- The active training scale is chosen only from iteration count, not from a probability or uncertainty controller.
- Evaluation still runs at the finest configured training scale, which is the first element of `--resolution_scales`.
- Allowed LoD scales are restricted to `[2, 4, 8]`. Scale `1` is not accepted.

## Resolution semantics

This codebase already has a base image-resolution control through `-r/--resolution` from [`arguments/__init__.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/arguments/__init__.py).

The LoD schedule is applied on top of that.

Practical meaning:

- `--resolution` controls the base dataset resize behavior.
- `--resolution_scales` controls the staged LoD multipliers inside the training loop.
- `--resolution_scales` must be drawn only from `[2, 4, 8]`.
- `--match_resolution` keeps lower-LoD tensors at the same final size as the finest level by loading lower-resolution images and resizing them back up.

Example:

- `-r 2 --resolution_scales 2 4 8 --naive_lod_stage_iterations 5000`

means the effective training path is approximately:

- coarse stage: base `r=2` plus LoD multiplier `8`
- middle stage: base `r=2` plus LoD multiplier `4`
- fine stage: base `r=2` plus LoD multiplier `2`

## Logging outputs

`train_nomask.py` writes per-run CSV logs in the output directory:

- `train_metrics.csv`
- `eval_metrics.csv`
- `runtime_metrics.csv`

`train_metrics.csv` includes:

- `iteration`
- `photometric_loss`
- `total_loss`
- `depth_loss`
- `lod_scale`
- `lod_stage_idx`
- `num_gaussians`
- `iter_time_ms`

`eval_metrics.csv` includes:

- `iteration`
- `split`
- `eval_scale`
- `num_cameras`
- `l1`
- `psnr`

`runtime_metrics.csv` includes:

- `scene_load` with `gpu_memory_mb` and `scene_load_time_sec`
- `iteration` rows with per-iteration `gpu_memory_mb`
- `training_loop` with the first in-loop `gpu_memory_mb` measurement
- `training_complete` with `total_training_time_sec`

TensorBoard logging is enabled when available and logs `train/lod_scale` plus the runtime memory/time summary points.

Optional Weights & Biases logging is available through:

- `--wandb_project`
- `--wandb_name`
- `--disable_wandb`

Current W&B train logs include:

- `train/photometric_loss`
- `train/total_loss`
- `train/depth_loss`
- `train/lod_scale`
- `train/lod_stage_idx`
- `train/num_gaussians`

Current W&B runtime logs include:

- `runtime/gpu_memory_scene_load_mb`
- `runtime/scene_load_time_sec`
- `runtime/gpu_memory_training_loop_mb`
- `runtime/total_training_time_sec` as a final-step metric and in W&B summary at the end of training

Current W&B eval logs include:

- `eval/L1`
- `eval/PSNR`
- `eval/lod_scale`
- one fixed render / ground-truth preview pair

Default evaluation cadence:

- test evaluation runs every `1000` iterations
- the final training iteration is always included

## Important implementation detail

- Camera sampling is index-based across all configured scales.
- One sampled viewpoint index is reused across scales, so the same scene/view is seen at the active LoD level.
- When `scene.extend()` adds new Gaussians and cameras, the multi-scale viewpoint stacks are rebuilt.
- After a real extension adds new viewpoints, split training resets the LoD phase to the coarsest configured scale for the next iteration.
- After that reset, the sampler is restricted to the newly appended viewpoint index range rather than the full accumulated training set.

## Files and functions to check before editing

- `_build_viewpoint_stacks(...)`
- `_refill_viewpoint_indices(...)`
- `_prepare_resolution_scales(...)`
- `_resolve_naive_lod_scale(...)`
- `_maybe_update_naive_lod_scale(...)`
- `_select_fixed_wandb_eval_view(...)`
- `training(...)`
- `training_report(...)`
- `Scene.extend(...)`

## Runner behavior

[`run_exp.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/run_exp.py) assumes both source scenes already exist. It does not create split scenes.

Current runner inputs:

- `--base_source`
- `--split_source`
- `--final_extension_iteration`

Expected source layout:

- `--base_source /path/to/<scene>_base/model0`
- `--split_source /path/to/<scene>_splitN/model0`

Conventions and derived values:

- The runner warns when either input does not point to a `model0` directory.
- The runner warns when the base parent directory does not follow `<scene>_base`.
- The split parent directory must follow `<scene>_splitN`, because `N` is used to infer the extension count.
- For the split scene:
  - `xtend = N - 1`
  - `splitter_itr = final_extension_iteration // (N - 1)`
- For the base scene:
  - `xtend = 0`
  - `--default` is enabled
  - no `splitter_itr` is passed

Current experiment variants:

- `baseline`
- `naive-lod`
- `matched-naive-lod`

Variant meanings:

- `baseline`: constant resolution, no LoD schedule
- `naive-lod`: fixed-iteration coarse-to-fine LoD schedule
- `matched-naive-lod`: same fixed-iteration LoD schedule, with `--match_resolution`

Current scale sets:

- start scale `2`
  - `baseline`: `[2]`
  - `naive-lod`: `[2, 4, 8]`
  - `matched-naive-lod`: `[2, 4, 8]`
- start scale `4`
  - `baseline`: `[4]`
  - `naive-lod`: `[4, 8]`
  - `matched-naive-lod`: `[4, 8]`
- start scale `8`
  - `baseline`: `[8]`

Split LoD timing:

- For split LoD variants, stage length is derived from the split extension interval.
- `naive_lod_stage_iterations = splitter_itr // total_levels`
- `total_levels` means `len(resolution_scales)` for that experiment.
- For base variants and all baselines, the runner keeps the default stage length of `5000`.
- When an extension actually appends a new viewpoint block, the trainer restarts the LoD phase from the coarsest scale on the next iteration.
- The sampler is then restricted to that new viewpoint block instead of the previously accumulated viewpoints.
- Example: with `split2`, `final_extension_iteration = 7500`, and `resolution_scales = [2, 4, 8]`, the split run uses `splitter_itr = 7500`, `naive_lod_stage_iterations = 7500 // 3 = 2500`, so training promotes `8 -> 4 -> 2` by iteration `7500`, then the extension fires at iteration `7500`, and the next iteration restarts the new-viewpoint phase at scale `8`.

Current runner matrix:

- `base` × `baseline`
- `base` × `naive-lod`
- `base` × `matched-naive-lod`
- `split` × `baseline`
- `split` × `naive-lod`
- `split` × `matched-naive-lod`

Equivalent shorthand:

- `(base | split) * (baseline | naive-lod | matched-naive-lod)`

Important limitation:

- This is not the full `(base | split) * (upscaled | constant) * (non LoD | LoD)` matrix.
- The missing quadrant is `upscaled + non-LoD`, because the runner does not define a `matched-baseline` variant.

W&B integration:

- `WANDB_PROJECT = 'gaussian-splatting-lod'`
- each run gets `--wandb_name` from the derived experiment name
