# Logging Behavior

## Scope

This document describes what the active `frankenstein_base` training pipeline logs today.

Primary logging surfaces:

- Weights & Biases through `wandb.log(...)`
- local CSV files written under each run output directory
- TensorBoard scalars for the same core runtime and training metrics

Relevant files:

- `run_exp.py`
- `train_nomask.py`
- `scene/__init__.py`

## W&B run identity

When runs are launched through `run_exp.py`, the runner assigns:

- one shared W&B project from `--wandb_project`
- one shared W&B group from `--wandb_group`, or a derived default group when not provided
- one distinct W&B run name per experiment from the experiment name builder

For the comparison runner this means:

- all runs in one comparison sweep can land in the same project
- the shared group keeps the 12-run matrix together
- the run name still encodes the per-run factors such as:
  - base vs split
  - EDGS init vs vanilla
  - densify vs no-densify
  - baseline vs naive LoD vs matched naive LoD

`train_nomask.py` also logs `config=vars(args)` to W&B init, so each run records its exact CLI-derived configuration.

Direct `train_nomask.py` launches now behave differently from older fixed-name runs:

- `--wandb_project` still controls the target W&B project
- `--wandb_group` still controls optional grouping
- if `--wandb_name` is omitted, `train_nomask.py` generates a random run name on each launch
- if `--wandb_name` is provided explicitly, that value is used unchanged

## W&B training metrics

The active training loop logs these scalar keys each iteration:

- `train/photometric_loss`: image reconstruction L1 before any depth term is added
- `train/total_loss`: full optimized loss after optional depth regularization
- `train/depth_loss`: current depth regularization contribution, or `0.0` when inactive
- `train/lod_scale`: resolution scale actually used for the sampled viewpoint that iteration
- `train/lod_stage_idx`: current naive LoD stage index
- `train/num_gaussians`: current active Gaussian count

Interpretation notes:

- `train/lod_scale` is the actual sampled training scale, not just the configured experiment scale list
- for split runs, the active Gaussian count can jump when `scene.extend()` merges the next block

## W&B evaluation metrics

At test intervals, the active evaluation path logs:

- `eval/L1`: mean test-view L1 over the evaluated test camera set
- `eval/PSNR`: mean test-view PSNR over the evaluated test camera set
- `eval/lod_scale`: the training-time LoD scale active when the eval was triggered

For test logging only, W&B also records a fixed render/ground-truth pair when available:

- `eval/images/render`
- `eval/images/ground_truth`

The fixed evaluation view is chosen once from the middle of the current test camera list at the finest configured scale.

## W&B runtime metrics

At `step=0`, the runtime logger writes scene setup and initialization metrics:

- `runtime/init_time_sec`: wall-clock time spent in the initialization section around `Scene(...)`
- `runtime/gpu_memory_scene_load_mb`: reserved GPU memory after scene load metrics are captured
- `runtime/scene_load_time_sec`: wall-clock time from scene-load start until the full scene setup phase completes

When EDGS initialization is active, the same runtime log now also includes:

- `runtime/edgs_base_init_time_sec`: wall-clock time spent on base-block EDGS initialization
- `runtime/edgs_base_init_gpu_memory_mb`: reserved GPU memory measured after base-block EDGS initialization
- `runtime/edgs_extensions_init_time_sec`: total wall-clock time spent initializing split extension blocks with EDGS
- `runtime/edgs_extensions_init_gpu_memory_mb`: maximum reserved GPU memory observed across extension-block EDGS initialization
- `runtime/edgs_extensions_init_count`: number of extension blocks initialized through EDGS
- `runtime/edgs_total_init_time_sec`: sum of base-block and extension-block EDGS initialization time

During training startup, the runtime logger also records:

- `runtime/gpu_memory_training_loop_mb`: reserved GPU memory measured at the first training-loop checkpoint

At the end of training, the runtime logger records:

- `runtime/total_training_time_sec`

## CSV files

Every run output directory currently writes three CSV files.

## Iteration-0 Point Cloud

Fresh runs also save the initialized Gaussian state before the first training step:

- `point_cloud/iteration_0/point_cloud.ply`

This file is not a metric log, but it is part of the debugging surface for packet and EDGS runs.

Interpretation:

- without EDGS, it shows the point-cloud seed converted into initial Gaussians
- with EDGS, it shows the state after EDGS/RoMa append and before optimization
- checkpoint resumes skip this save because the loaded checkpoint is not a fresh initialization

### `train_metrics.csv`

Columns:

- `iteration`
- `photometric_loss`
- `total_loss`
- `depth_loss`
- `lod_scale`
- `lod_stage_idx`
- `num_gaussians`
- `iter_time_ms`

Each row corresponds to one training iteration.

### `eval_metrics.csv`

Columns:

- `iteration`
- `split`
- `eval_scale`
- `num_cameras`
- `l1`
- `psnr`

Each row corresponds to one evaluation event for one split.

Current active usage is primarily:

- `split=test`

The `eval_scale` column records the finest configured eval scale used by the evaluation helper.

### `runtime_metrics.csv`

Columns:

- `event`
- `iteration`
- `gpu_memory_mb`
- `init_time_sec`
- `scene_load_time_sec`
- `total_training_time_sec`

This file is event-oriented rather than iteration-oriented.

Currently observed event names include:

- `initialization`
- `scene_load`
- `edgs_base_init`
- `edgs_extensions_init`
- `training_loop`
- `iteration`
- `training_complete`

Interpretation:

- `initialization` stores `init_time_sec`
- `scene_load` stores `scene_load_time_sec` and scene-load GPU memory
- `edgs_base_init` stores base-block EDGS init time and GPU memory when EDGS is active
- `edgs_extensions_init` stores total split-extension EDGS init time and GPU memory when extension EDGS init is active
- `training_loop` stores the first recorded training-loop GPU memory checkpoint
- `iteration` stores per-iteration GPU memory snapshots
- `training_complete` stores final total training time

## Split-scene EDGS timing semantics

The EDGS extension timing metrics measure eager extension initialization during `Scene(...)` construction.

They do not measure:

- later `scene.extend()` merge cost
- render-time cost after extension

So for split runs:

- `runtime/edgs_extensions_init_time_sec` means "time spent prebuilding extension Gaussian blocks with EDGS"
- it does not mean "time spent extending the active model during training"

## Comparison-matrix logging behavior

The active comparison runner in `run_exp.py` can launch:

- 6 vanilla runs
- 6 EDGS runs
- all 12 runs together

For a full comparison launch:

- all 12 runs can share one W&B project
- all 12 runs can share one W&B group
- each run still keeps its own run name and its own local CSV files

This is the intended setup for comparing improvements attributable to:

- EDGS initialization
- naive LoD scheduling
- incremental split-scene training
- matched vs unmatched LoD image handling
