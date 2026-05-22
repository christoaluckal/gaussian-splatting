# Current Runner Behavior

## Scope

This document describes the active behavior of `frankenstein_base/run_exp.py`.

It is the runner handoff reference for:

- experiment naming
- comparison-matrix launch behavior
- LoD scale semantics
- split-scene handling
- dynamic viewpoint-split handling
- EDGS initialization toggles
- densification toggles
- W&B grouping behavior
- default runtime flags

## Experiment authority

The experiment dict in `run_exp.py` is the authority for LoD scale.

Current rule:

- `resolution_scales` defines the actual training and eval image scale levels
- the runner always passes `-r 1`
- `start_scale`, if present, must match `resolution_scales[0]`

This was changed to avoid accidental double downsampling.

Before the fix, the runner passed both:

- `-r <start_scale>`
- `--resolution_scales <start_scale> ...`

That caused the effective resolution to become:

- `orig / (args.resolution * resolution_scale)`

So a `1920x1080` image with `-r 8` and `resolution_scales=[8]` became roughly `30x17`.

Current effective behavior is:

- `resolution_scales=[1]` means full input image size
- `resolution_scales=[2,4,8]` means `1/2`, `1/4`, and `1/8` of original size
- W&B eval images reflect those resized camera tensors directly

For visual-quality debugging, prefer a full-resolution direct run with
`--resolution_scales 1`. A run at `resolution_scales=[8]` is useful for fast
plumbing checks, but its eval render is only one eighth of the input width and
height and will look blurry when inspected at larger display size.

## Active comparison matrix

The active runner now supports explicit launch groups through `--run_group`.

Current intended comparison matrix:

- `vanilla`: 6 runs
  - init mode: no EDGS
  - scene mode: base and split
  - schedule mode: baseline, naive LoD, matched naive LoD
- `edgs` or `non-vanilla`: 6 runs
  - init mode: EDGS
  - scene mode: base and split
  - schedule mode: baseline, naive LoD, matched naive LoD
- `full-comparison` or `all`: all 12 runs together

This reflects the meaningful comparison space:

- EDGS init vs no EDGS init
- split incremental training vs non-split training
- no LoD vs LoD
- unmatched LoD vs matched LoD

Important nuance:

- `matched-naive-lod` is only meaningful for the LoD path
- baseline runs still exist for both base and split scenes, but "matched" does not create a separate non-LoD baseline mode

## Scene variants launched

`run_exp.py` can launch both scene variants from one command:

- base scene from `--base_source`
- split scene from `--split_source`

Expected path conventions:

- base scene should look like `<scene>_base/model0`
- split scene should look like `<scene>_splitN/model0`

For the split scene:

- the runner infers `xtend = N - 1`
- it derives `splitter_itr = final_extension_iteration // (N - 1)`

## Dynamic viewpoint splitting

The training entrypoint now also supports a dynamic split path that does not require pre-split sibling `modelN` folders.

Current controls are passed directly to `train_nomask.py`:

- `--viewpoint_splitter`
- `--viewpoint_splitter_config`

Semantics:

- if `--viewpoint_splitter` is omitted, split behavior stays on the legacy sibling-folder path
- if `--viewpoint_splitter` is set, `Scene(...)` partitions the base train-camera set into `xtend + 1` viewpoint groups
- the first partition is trained as the base block
- later partitions are appended through the existing `scene.extend()` path

When split append is active, LoD is also block-aware:

- effective per-block LoD stage length is derived inside `train_nomask.py` as `splitter_itr // len(resolution_scales)`
- older viewpoint blocks keep the highest resolution scale they have already reached
- newly appended viewpoint blocks start at the coarsest configured LoD scale and promote independently
- split runs treat `densify_until_iter` as a minimum cutoff; if needed, densification is extended through the final append iteration and then kept alive for a short post-append buffer

So the runner-level `splitter_itr` now affects two things:

- when the next viewpoint block is appended
- how long each per-block LoD promotion stage lasts in split mode

Current default implementation available in-tree:

- `pose_kmeans`

This keeps the partition criterion replaceable by module import rather than embedding one fixed policy in the trainer or runner.

## Splitter Matrix Runner

There is now a dedicated matrix runner for splitter-iteration and cluster-count sweeps:

- `run_tartanair_splitter_matrix.py`

Its default purpose is different from the older sweep runners:

- hold `densify_grad_threshold` fixed at `4e-4`
- hold total iterations at `40000`
- hold `densify_until_iter` at `30000`
- sweep `cluster_count` from `2` through `10`
- sweep `splitter_itr` from `1000` through `15000`
- launch both:
  - split baseline
  - split LoD
- keep one EDGS non-split non-LoD control

Default output layout:

- run outputs:
  - `output/tartan_splitter_matrix_runs`
- EDGS cache:
  - `output/tartan_splitter_matrix_cache`

Default launcher behavior:

- resume mode is enabled by default
- rerunning the same command skips run directories that already look completed
- interrupting with `Ctrl+C` stops the current launch loop cleanly
- rerunning the same command resumes unfinished jobs from the same output root

The matrix runner also builds local summary artifacts over completed runs:

- `splitter_matrix_summary.csv`
- `splitter_matrix_report.md`
- `splitter_matrix_heatmaps.png`

The heatmap figure uses:

- X axis: `splitter_itr`
- Y axis: `cluster_count`
- separate baseline and LoD rows
- separate metric panels for:
  - final Gaussian count
  - final PSNR
  - end-to-end time

This runner is intended to answer:

- which `splitter_itr` is best
- which `cluster_count` is best
- whether LoD helps or hurts at each split geometry

The matrix runner also applies a dynamic feasibility filter before launching split jobs.

Current rule:

- `cluster_count * splitter_itr + post_append_densify_buffer(densification_interval) <= iterations`

Where:

- `post_append_densify_buffer(densification_interval)` is:
  - at least `2000` iterations
  - and rounded up to a multiple of `densification_interval`

This means the runner skips combinations that cannot both:

- append all viewpoint blocks within the configured total iteration budget
- leave a short post-append tail so the final block still gets some densification time before training ends

So the invalidity check is runtime-driven from the current CLI values, not hardcoded to one particular pair such as `cluster_count=10` and `splitter_itr=10000`.

## EDGS Cache Reuse

The matrix runner now passes:

- `--edgs_cache_root`

This enables scene-level reuse of EDGS-initialized Gaussian blocks across repeated runs that share the same split geometry.

Current cache semantics:

- cache reuse is keyed by:
  - `source_path`
  - `xtend`
  - `viewpoint_splitter`
  - `viewpoint_splitter_config`
  - reference resolution scale
  - EDGS configuration
- for a fixed `cluster_count`, different `splitter_itr` values reuse the same cached EDGS base and extension Gaussian blocks
- baseline and LoD runs also reuse the same cached EDGS initialization when their split geometry matches

This means the cache is primarily a practical throughput optimization for matrix-style repeated runs, not a change to the training objective.

## Naming

Experiment names currently include:

- scene name
- init mode
- densification mode
- EDGS train-recipe marker when active
- experiment label
- first resolution scale

Mode labels:

- `vanilla` means `--no-edgs_init` and `--densify`
- `edgs-init-densify`
- `edgs-init-no-densify`
- `sfm-init-no-densify`

When EDGS compatibility training is active, the runner appends:

- `-edgs-train-recipe`

## Per-job comparison policy

The comparison runner now chooses some settings per job instead of globally per invocation.

Current policy:

- vanilla comparison jobs:
  - no EDGS init
  - densification enabled
  - no EDGS compatibility training recipe
- EDGS comparison jobs:
  - EDGS init enabled
  - densification disabled
  - EDGS compatibility training recipe enabled

This is the current fairness policy for comparing:

- EDGS initialization
- naive LoD scheduling
- incremental split-scene training

## Densification control

The runner exposes:

- `--densify`
- `--no-densify`

This is forwarded into `train_nomask.py`.

When disabled:

- the training loop uses `effective_densify_until_iter = 0`
- densification stats are not accumulated
- `densify_and_prune(...)` is never called
- opacity reset behavior tied to densification phase is skipped

## EDGS control

The runner exposes:

- `--edgs_matches_per_ref`
- `--edgs_num_refs`
- `--edgs_nns_per_ref`
- `--edgs_scaling_factor`
- `--edgs_proj_err_tolerance`
- `--edgs_roma_model`
- `--edgs_add_sfm_init`
- `--edgs_init_extensions` / `--no-edgs_init_extensions`

These are forwarded into EDGS jobs inside the comparison matrix.

The runner no longer uses a single top-level `--edgs_init` toggle to decide the whole launch set.

## Default runtime flags

Current defaults in the runner:

- viewer disabled by default
- W&B enabled by default

So runner-launched jobs:

- do not bind the network viewer unless explicitly re-enabled
- do log to W&B unless `--disable_wandb` is passed

The runner also exposes:

- `--wandb_project`
- `--wandb_group`

Current behavior:

- all jobs in one comparison launch can share one W&B project
- all jobs in one comparison launch can share one W&B group
- each job still gets its own distinct W&B run name

More detailed logging behavior lives in `logging_behavior.md`.

## Failure handling

The runner currently catches exceptions around `subprocess.run(...)` and then continues silently.

That behavior is convenient for bulk launching but weak for debugging because a failed child process does not stop the batch with a clear error summary.

This is important context for handoff: child training failures may be easy to miss unless the console output is watched directly.
