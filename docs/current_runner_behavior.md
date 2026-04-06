# Current Runner Behavior

## Scope

This document describes the current behavior of:

- [`run_exp.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/run_exp.py)
- [`train_nomask.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/train_nomask.py)

It is intentionally descriptive of the current implementation, not a proposal for a future runner layout.

## Source-scene contract

The runner assumes the source scenes already exist on disk.

It does not create split scenes.

For the current splitter-side scene format and split-script behavior, see [`docs/colmap_splitter_behavior.md`](/mnt/share/nas/christo/splatting/gaussian-splatting/docs/colmap_splitter_behavior.md).

Expected inputs:

- `--base_source /path/to/<scene>_base/model0`
- `--split_source /path/to/<scene>_splitN/model0`

The runner warns when:

- either input does not point to a `model0` directory
- the base parent directory does not match `<scene>_base`

The split parent directory is stricter than the base one. It must match `<scene>_splitN`, because the runner uses `N` to infer how many extensions the split experiment should perform.

## Runner inputs

The current runner takes:

- `--base_source`
- `--split_source`
- `--final_extension_iteration`

Interpretation:

- `base_source` is the non-split scene
- `split_source` is the split scene
- `final_extension_iteration` is the iteration where the final extension should occur for the split variant

Derived split behavior:

- if `split_source` is `<scene>_splitN/model0`
- then the split experiment uses:
  - `xtend = N - 1`
  - `splitter_itr = final_extension_iteration // (N - 1)`

Derived base behavior:

- `xtend = 0`
- `--default` is enabled
- `splitter_itr` is omitted

## Experiment variants

The current experiment variants are:

- `baseline`
- `naive-lod`
- `matched-naive-lod`

Their meanings are:

- `baseline`: single-scale training with one fixed LoD scale
- `naive-lod`: fixed-iteration coarse-to-fine LoD training
- `matched-naive-lod`: same fixed-iteration LoD schedule, but with `--match_resolution`

Allowed LoD scales:

- only `2`, `4`, and `8`
- scale `1` is not supported by the current LoD path

Current start-scale grid:

- start scale `2`
- start scale `4`
- start scale `8`

Current scale schedules:

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

## LoD timing rule

Inside [`train_nomask.py`](/mnt/share/nas/christo/splatting/gaussian-splatting/train_nomask.py), the active scale is still chosen only from iteration count.

For a given stage length `S`:

- iterations `1..S` use the coarsest configured scale
- iterations `S+1..2S` use the next finer scale
- later iterations continue promoting until the finest configured scale is reached

Runner-specific stage length behavior:

- for split LoD variants, `naive_lod_stage_iterations = splitter_itr // total_levels`
- `total_levels` is `len(resolution_scales)` for that experiment
- for base variants and baseline variants, the runner uses the default `5000`
- when an extension actually appends new viewpoints, the trainer resets the active LoD phase to the coarsest configured scale on the next iteration
- after that reset, viewpoint sampling remains uniform over all accumulated viewpoints; previously promoted viewpoints render at the finest scale, while the new active block follows the restarted LoD phase
- example: if `split_source` is `..._split2/model0`, `final_extension_iteration = 7500`, and the LoD variant uses `resolution_scales = [2, 4, 8]`, then `splitter_itr = 7500`, `total_levels = 3`, and `naive_lod_stage_iterations = 2500`; the run trains at scales `8`, then `4`, then `2`, the extension triggers at iteration `7500`, and the next iteration restarts the new-viewpoint phase at scale `8`

## Actual experiment matrix

The current runner executes exactly two scene variants:

- base
- split

and crosses them with all configured experiment variants.

So the current matrix is:

- base baseline
- base naive-lod
- base matched-naive-lod
- split baseline
- split naive-lod
- split matched-naive-lod

Equivalent shorthand:

- `(base | split) * (baseline | naive-lod | matched-naive-lod)`

This is close to:

- `(base | split) * (upscaled | constant) * (non LoD | LoD)`

but not identical.

The missing quadrant is:

- upscaled + non-LoD

because the runner does not currently define a `matched-baseline` variant.

## Training and logging behavior

`train_nomask.py` currently provides:

- deterministic fixed-iteration LoD scheduling through `--resolution_scales` and `--naive_lod_stage_iterations`
- split-specific LoD phase resets after extensions that add a new viewpoint block
- uniform viewpoint sampling across all accumulated viewpoints, with finest-scale rendering for previously promoted blocks and LoD rendering for the active block
- finest-scale evaluation through the shared reporting path
- default test evaluation every `1000` iterations, plus the final iteration
- runtime logging for scene-load allocated/reserved/peak GPU memory, Gaussian model size, scene-load time, per-iteration memory/model snapshots in `runtime_metrics.csv`, and total training time
- CSV logging:
  - `train_metrics.csv`
  - `eval_metrics.csv`
  - `runtime_metrics.csv`
- TensorBoard logging with LoD plus the runtime memory/time summary points
- optional W&B logging

Current W&B train logs:

- `train/photometric_loss`
- `train/total_loss`
- `train/depth_loss`
- `train/lod_scale`
- `train/lod_stage_idx`
- `train/num_gaussians`

Current W&B runtime logs:

- `runtime/gpu_allocated_scene_load_mb`
- `runtime/gpu_reserved_scene_load_mb`
- `runtime/gpu_peak_allocated_scene_load_mb`
- `runtime/gpu_peak_reserved_scene_load_mb`
- `runtime/gaussian_model_scene_load_mb`
- `runtime/scene_load_time_sec`
- `runtime/gpu_peak_allocated_training_loop_mb`
- `runtime/gpu_peak_reserved_training_loop_mb`
- `runtime/gaussian_model_training_loop_mb`
- `runtime/total_training_time_sec` as a final-step metric and in W&B summary at the end of training
- `runtime/gpu_peak_allocated_final_mb` in the final-step log and W&B summary
- `runtime/gpu_peak_reserved_final_mb` in the final-step log and W&B summary
- `runtime/gaussian_model_final_mb` in the final-step log and W&B summary

Current W&B eval logs:

- `eval/L1`
- `eval/PSNR`
- `eval/lod_scale`
- one fixed test-view render / ground-truth image pair

## Match-resolution behavior

When `--match_resolution` is enabled:

- lower LoD levels are still loaded from lower-resolution source images
- they are then resized back to the finest active training tensor size
- the LoD difference becomes blur/detail rather than tensor shape
