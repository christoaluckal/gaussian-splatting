# TartanAir Vanilla, EDGS, and LoD Sweep

## Scope

`run_tartanair_vanilla_edgs_lod_sweep.py` launches a focused three-run
comparison on one vanilla COLMAP-formatted TartanAir scene.

The three jobs are:

| Job | EDGS init | Viewpoint clusters | Resolution scales |
| --- | --- | --- | --- |
| `vanilla_no_lod` | no | none | `2` |
| `edgs_no_lod` | yes | `3` | `2` |
| `edgs_lod` | yes | `3` | `2 4 8` |

The vanilla job is intentionally non-clustered. It passes:

- `-x 0`
- `--default`

It does not pass a viewpoint splitter.

Both EDGS jobs use dynamic `pose_kmeans` partitioning. With the default
`cluster_count=3`, they pass `-x 2`, train the first cluster immediately, and
append the other two clusters at the configured splitter interval.

## Interpretation

This sweep answers two related questions:

1. How does the conventional non-clustered vanilla baseline compare with the
   clustered EDGS pipeline?
2. Within the same clustered EDGS pipeline, what changes when naive LoD is
   enabled?

Only the second comparison is matched for clustering:

- `edgs_no_lod` versus `edgs_lod`

The vanilla-versus-EDGS result includes both the initialization difference and
the clustered append policy. It must not be described as an EDGS-only
ablation.

## Default Scene

The default source is:

```text
../bags/tartanair_colmap_vanilla_full/CyberPunkDowntown_P0000
```

This scene came from the TartanAir packet-to-COLMAP check and has a complete
registered image set. The runner expects the standard Gaussian Splatting
layout, including:

```text
images/
sparse/0/
```

Use `--source` to select another successful converted environment.

## Default Training Settings

Shared settings:

- total iterations: `40000`
- densification cutoff: `30000`
- densification interval: `100`
- densification gradient threshold: `1e-3`
- eval mode enabled
- viewer disabled
- vanilla and EDGS no-LoD resolution scales: `2`
- EDGS LoD resolution scales: `2 4 8`
- naive LoD stage iterations: `5000`

Clustered EDGS settings:

- cluster count: `3`
- extension count: `2`
- splitter interval: `10000`
- viewpoint splitter: `pose_kmeans`
- splitter config:

```json
{"position_scale": 1.0, "forward_scale": 0.5, "max_iterations": 32}
```

EDGS settings:

- SfM seed retained with `--edgs_add_sfm_init`
- matches per reference: `200`
- number of references: `500`
- projection error tolerance: `0.01`
- RoMa model: `outdoors`
- skipped frames: `0`
- extension-block EDGS initialization remains enabled through the trainer
  default

## Outputs and W&B

Default local output root:

```text
output/tartanair_vanilla_edgs_lod_sweep
```

Default EDGS cache root:

```text
output/tartanair_vanilla_edgs_lod_cache
```

Default W&B project and group:

```text
tartanair-colmap-vanilla-edgs-lod
cyberpunkdowntown-cluster3-g1e-3
```

Run names include the scene, clustering mode, variant, resolution scales,
gradient threshold, iteration count, and densification cutoff. The vanilla
name contains `no_cluster`; EDGS names contain `cluster3_split10000` with the
defaults.

## Launch

From `gaussian-splatting`:

```bash
conda run -n frankenstein python \
  run_tartanair_vanilla_edgs_lod_sweep.py
```

Inspect all generated training commands without launching:

```bash
conda run -n frankenstein python \
  run_tartanair_vanilla_edgs_lod_sweep.py \
  --dry-run
```

Use a different converted environment and a distinct W&B group:

```bash
conda run -n frankenstein python \
  run_tartanair_vanilla_edgs_lod_sweep.py \
  --source ../bags/tartanair_colmap_vanilla_full/SeasideTown_P0000 \
  --wandb-group seasidetown-cluster3-g1e-3
```

## Resume and Failure Behavior

Resume is enabled by default.

A job is considered complete only when:

- `runtime_metrics.csv` contains `training_complete`
- `train_metrics.csv` contains the configured final iteration

Rerunning the same command skips jobs that satisfy both checks. Use
`--no-resume` to launch all jobs again.

The runner stops on the first failed child process by default. Use
`--continue-on-error` to continue to later jobs. `Ctrl+C` stops the current
training process and exits the sweep; rerun the same command to resume.

## Important Overrides

The following options affect both EDGS jobs but not the vanilla clustering
policy:

- `--cluster-count`
- `--splitter-itr`
- `--viewpoint-splitter`
- `--viewpoint-splitter-config`

The vanilla run remains non-clustered regardless of `--cluster-count`.

The runner validates that the final EDGS cluster append occurs before the end
of training. It also requires:

- positive cluster count
- positive splitter interval
- `densify_until_iter < iterations`

Changing the source, training schedule, or EDGS configuration should normally
be paired with a new W&B group. Changing settings while reusing an existing
output root can otherwise make old completed directories eligible for resume
skipping when their generated run names still match.
