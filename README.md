# Frankenstein Base

Original implementation:
> https://github.com/graphdeco-inria/gaussian-splatting

This directory is the working fork used for experiment-pipeline changes around:

- split-scene training
- naive LoD training via `train_nomask.py`
- runner-based experiment launches via `run_exp.py`
- optional EDGS / RoMa-based Gaussian initialization

## What

`frankenstein_base` keeps the original Gaussian Splatting training structure, but adds a direct bridge to EDGS-style initialization.

Current additions on top of the upstream baseline:

- `run_exp.py` can launch runs with explicit `edgs_*` settings
- `run_exp.py` can launch the current 12-run comparison matrix with shared W&B project/group metadata
- `train_nomask.py` accepts `edgs_*` arguments and forwards them into scene construction
- `edgs_init.py` imports the EDGS correspondence initializer and applies it locally
- split extension blocks can also use EDGS initialization
- runtime logging now records EDGS base-block and split-extension initialization timing

Important current limitation:

- `--edgs_init` is not fully self-contained inside `frankenstein_base`
- the active bridge still imports from sibling `../EDGS/source/*` and `../EDGS/submodules/RoMa`
- cloning `frankenstein_base` by itself is therefore not enough for EDGS mode

The main EDGS-facing functionality is:

- initialize from COLMAP / SfM point clouds as usual
- optionally augment or replace that seed using RoMa correspondences triangulated by EDGS logic

## Why

The purpose of this fork is to include EDGS-style initialization directly inside the local experiment pipeline instead of keeping it as a separate repo workflow.

That matters because:

- the runner should be able to toggle EDGS initialization as part of the experiment matrix
- base, incremental, and LoD experiments should all share the same initialization controls
- split-scene extensions should be initialized through the same pipeline as the base scene

In short:

- upstream Gaussian Splatting provides the base optimizer and scene handling
- EDGS provides the stronger correspondence-based initialization
- `frankenstein_base` combines them inside one experiment pipeline

## How

### Plain initialization

Without `--edgs_init`, initialization is standard 3DGS:

- `Scene(...)` loads cameras and point clouds
- `GaussianModel.create_from_pcd(...)` seeds the splats from COLMAP points

### EDGS initialization

With `--edgs_init`, the pipeline does:

1. initialize from the COLMAP point cloud
2. set up the Gaussian optimizer state
3. call the EDGS correspondence initializer through `edgs_init.py`
4. optionally prune the original SfM seed
5. continue with the normal training loop

For split scenes:

- `model0` can use EDGS initialization
- sibling extension blocks can also use EDGS initialization when `--edgs_init_extensions` is enabled

### Relevant arguments

`train_nomask.py` currently supports:

- `--edgs_init`
- `--edgs_matches_per_ref`
- `--edgs_num_refs`
- `--edgs_nns_per_ref`
- `--edgs_scaling_factor`
- `--edgs_proj_err_tolerance`
- `--edgs_roma_model`
- `--edgs_add_sfm_init`
- `--edgs_init_extensions` / `--no-edgs_init_extensions`

`run_exp.py` exposes the same controls for experiment launches.

Current comparison-runner additions also include:

- `--run_group` for `vanilla`, `edgs`, or full 12-run comparison launches
- `--iterations` pass-through into `train_nomask.py`
- shared `--wandb_project` and `--wandb_group` wiring across all runner-launched jobs
- EDGS-only compatibility training mode applied per job inside the comparison matrix

### Example

```bash
python run_exp.py \
  --base_source /path/to/home_base/model0 \
  --split_source /path/to/home_split2/model0 \
  --final_extension_iteration 7500 \
  --edgs_init \
  --edgs_matches_per_ref 15000 \
  --edgs_num_refs 180 \
  --edgs_nns_per_ref 3
```

## Docs

More detailed notes live in:

- `docs/initialization_behavior.md`
- `docs/current_runner_behavior.md`
- `docs/colmap_splitter_behavior.md`
- `docs/logging_behavior.md`
- `docs/handoff.md`
- `docs/edgs_pipeline_notes.md`
