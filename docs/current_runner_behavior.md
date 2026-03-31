# Current Runner Behavior

## Scope

This document describes the active behavior of `frankenstein_base/run_exp.py`.

It is the runner handoff reference for:

- experiment naming
- comparison-matrix launch behavior
- LoD scale semantics
- split-scene handling
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

- `resolution_scales=[2,4,8]` means `1/2`, `1/4`, and `1/8` of original size
- W&B eval images reflect those resized camera tensors directly

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
