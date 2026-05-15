# Experiment Reporting

Use `scripts/collate_run_metrics.py` to create a compact Markdown report and flat CSV from run directories under `output`.

```bash
python3 scripts/collate_run_metrics.py \
  --output_root output \
  --summary_csv output/collated_summary.csv \
  --report_md output/cost_analysis.md
```

The current OpenVINS packet report is:

- `output/cost_analysis.md`
- `output/collated_summary.csv`

## Report interpretation

`Post Init Max GPU (MB)` is the initialization-phase peak reserved CUDA memory. It comes from the `post_initialization_peak` event in `runtime_metrics.csv` and covers scene construction, Gaussian creation, EDGS initialization when enabled, and iteration-0 saving when enabled.

`Peak GPU (MB)` is the maximum reserved CUDA memory seen across all rows in `runtime_metrics.csv`, including startup and training-loop snapshots.

`Training GPU GB-hours` is a time-weighted reserved-memory proxy computed from per-iteration memory samples and iteration times. It is not direct GPU utilization or FLOP usage.

## EDGS and LoD inference

The collator supports structured experiment names from `run_exp.py`, but it also handles short ad hoc names. When a name is ambiguous, the report infers:

- scene name from `cfg_args.source_path`
- EDGS mode from nonzero EDGS initialization timing in `runtime_metrics.csv`
- LoD mode from multiple observed `lod_scale` values in `train_metrics.csv`

For example, a run with observed LoD scales `8,4,2` is treated as `naive-lod` even if the directory name does not contain `lod`.

## Current two-run comparison

The current report compares:

- `rpng_2`: EDGS-only baseline at observed scale `2`
- `rpng_248`: EDGS plus naive LoD with observed scales `8,4,2`

In the generated report, the EDGS+LoD run is faster end to end, uses fewer final Gaussians, and has slightly lower final PSNR than the EDGS-only run. Its post-init max GPU usage is higher because loading multiple resolution-scale camera tensors increases the startup footprint.
