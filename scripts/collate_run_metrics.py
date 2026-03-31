#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


EXPERIMENT_LABELS = ["baseline", "naive-lod", "matched-naive-lod"]
MODE_PREFIXES = ["-vanilla", "-edgs-init", "-sfm-init"]


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value):
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_expected_iterations(cfg_args_path):
    if not cfg_args_path.exists():
        return None
    content = cfg_args_path.read_text()
    match = re.search(r"iterations=(\d+)", content)
    if not match:
        return None
    return int(match.group(1))


def _parse_run_name(run_name):
    resolution_match = re.search(r"-r(?P<resolution>\d+)$", run_name)
    if resolution_match is None:
        raise ValueError(f"Could not parse resolution from run name: {run_name}")
    resolution = int(resolution_match.group("resolution"))
    prefix = run_name[: resolution_match.start()]

    experiment_label = None
    for label in sorted(EXPERIMENT_LABELS, key=len, reverse=True):
        suffix = f"-{label}"
        if prefix.endswith(suffix):
            experiment_label = label
            prefix = prefix[: -len(suffix)]
            break
    if experiment_label is None:
        raise ValueError(f"Could not parse experiment label from run name: {run_name}")

    scene_name = None
    mode_label = None
    for mode_prefix in MODE_PREFIXES:
        if prefix.endswith(mode_prefix):
            scene_name = prefix[: -len(mode_prefix)]
            mode_label = mode_prefix.lstrip("-")
            break
        marker = mode_prefix + "-"
        if marker in prefix:
            scene_name, mode_rest = prefix.split(marker, 1)
            mode_label = mode_prefix.lstrip("-") + "-" + mode_rest
            break
    if scene_name is None or mode_label is None:
        raise ValueError(f"Could not parse scene/mode from run name: {run_name}")

    is_edgs = "edgs-init" in mode_label
    uses_edgs_recipe = "edgs-train-recipe" in mode_label
    densify = "no-densify" not in mode_label
    scene_mode = "split" if "_split" in scene_name else "base"
    lod_mode = "baseline" if experiment_label == "baseline" else "lod"
    matched = experiment_label == "matched-naive-lod"

    return {
        "run_name": run_name,
        "scene_name": scene_name,
        "scene_mode": scene_mode,
        "mode_label": mode_label,
        "init_mode": "edgs" if is_edgs else "vanilla",
        "densify": densify,
        "uses_edgs_train_recipe": uses_edgs_recipe,
        "experiment_label": experiment_label,
        "lod_mode": lod_mode,
        "matched_resolution": matched,
        "resolution": resolution,
    }


def _summarize_runtime_rows(rows):
    gpu_values_all = []
    gpu_values_iteration = []
    training_complete_row = None
    scene_load_row = None
    initialization_row = None
    event_init_times = defaultdict(float)

    for row in rows:
        event = row.get("event", "")
        gpu_memory_mb = _safe_float(row.get("gpu_memory_mb"))
        if gpu_memory_mb is not None:
            gpu_values_all.append(gpu_memory_mb)
            if event in {"iteration", "training_loop"}:
                gpu_values_iteration.append(gpu_memory_mb)

        init_time_sec = _safe_float(row.get("init_time_sec"))
        if init_time_sec is not None:
            event_init_times[event] += init_time_sec

        if event == "training_complete":
            training_complete_row = row
        elif event == "scene_load":
            scene_load_row = row
        elif event == "initialization":
            initialization_row = row

    total_training_time_sec = _safe_float(
        training_complete_row.get("total_training_time_sec") if training_complete_row else None
    )
    scene_load_time_sec = _safe_float(scene_load_row.get("scene_load_time_sec") if scene_load_row else None)
    init_time_sec = _safe_float(initialization_row.get("init_time_sec") if initialization_row else None)

    return {
        "total_training_time_sec": total_training_time_sec,
        "scene_load_time_sec": scene_load_time_sec,
        "init_time_sec": init_time_sec,
        "edgs_base_init_time_sec": event_init_times.get("edgs_base_init", 0.0),
        "edgs_extensions_init_time_sec": event_init_times.get("edgs_extensions_init", 0.0),
        "peak_gpu_memory_mb_overall": max(gpu_values_all) if gpu_values_all else None,
        "avg_gpu_memory_mb_training": mean(gpu_values_iteration) if gpu_values_iteration else None,
        "peak_gpu_memory_mb_training": max(gpu_values_iteration) if gpu_values_iteration else None,
    }


def _summarize_eval_rows(rows):
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        return {
            "final_eval_iteration": None,
            "final_eval_l1": None,
            "final_eval_psnr": None,
        }
    final_row = max(test_rows, key=lambda row: _safe_int(row.get("iteration")) or -1)
    return {
        "final_eval_iteration": _safe_int(final_row.get("iteration")),
        "final_eval_l1": _safe_float(final_row.get("l1")),
        "final_eval_psnr": _safe_float(final_row.get("psnr")),
    }


def _summarize_train_rows(rows):
    if not rows:
        return {
            "final_train_iteration": None,
            "final_num_gaussians": None,
            "sum_iter_time_sec": None,
            "training_gpu_memory_gb_hours": None,
        }

    final_row = max(rows, key=lambda row: _safe_int(row.get("iteration")) or -1)
    final_iteration = _safe_int(final_row.get("iteration"))
    final_num_gaussians = _safe_int(final_row.get("num_gaussians"))

    sum_iter_time_sec = 0.0
    for row in rows:
        iter_time_ms = _safe_float(row.get("iter_time_ms"))
        if iter_time_ms is not None:
            sum_iter_time_sec += iter_time_ms / 1000.0

    return {
        "final_train_iteration": final_iteration,
        "final_num_gaussians": final_num_gaussians,
        "sum_iter_time_sec": sum_iter_time_sec,
    }


def _compute_training_memory_gb_hours(train_rows, runtime_rows):
    iter_gpu_by_iteration = {}
    for row in runtime_rows:
        if row.get("event") != "iteration":
            continue
        iteration = _safe_int(row.get("iteration"))
        gpu_memory_mb = _safe_float(row.get("gpu_memory_mb"))
        if iteration is None or gpu_memory_mb is None:
            continue
        iter_gpu_by_iteration[iteration] = gpu_memory_mb

    gb_hours = 0.0
    have_samples = False
    for row in train_rows:
        iteration = _safe_int(row.get("iteration"))
        iter_time_ms = _safe_float(row.get("iter_time_ms"))
        gpu_memory_mb = iter_gpu_by_iteration.get(iteration)
        if iteration is None or iter_time_ms is None or gpu_memory_mb is None:
            continue
        have_samples = True
        gb_hours += (gpu_memory_mb / 1024.0) * (iter_time_ms / 1000.0 / 3600.0)

    if not have_samples:
        return None
    return gb_hours


def _load_run_summary(run_dir):
    parsed = _parse_run_name(run_dir.name)

    cfg_args_path = run_dir / "cfg_args"
    train_metrics_path = run_dir / "train_metrics.csv"
    eval_metrics_path = run_dir / "eval_metrics.csv"
    runtime_metrics_path = run_dir / "runtime_metrics.csv"

    train_rows = _read_csv_rows(train_metrics_path)
    eval_rows = _read_csv_rows(eval_metrics_path)
    runtime_rows = _read_csv_rows(runtime_metrics_path)

    runtime_summary = _summarize_runtime_rows(runtime_rows)
    eval_summary = _summarize_eval_rows(eval_rows)
    train_summary = _summarize_train_rows(train_rows)
    training_gpu_memory_gb_hours = _compute_training_memory_gb_hours(train_rows, runtime_rows)

    expected_iterations = _parse_expected_iterations(cfg_args_path)
    final_train_iteration = train_summary["final_train_iteration"]
    completed = runtime_summary["total_training_time_sec"] is not None
    if expected_iterations is not None and final_train_iteration is not None:
        completed = completed and final_train_iteration >= expected_iterations

    scene_load_time_sec = runtime_summary["scene_load_time_sec"]
    total_training_time_sec = runtime_summary["total_training_time_sec"]
    end_to_end_time_sec = None
    if scene_load_time_sec is not None and total_training_time_sec is not None:
        end_to_end_time_sec = scene_load_time_sec + total_training_time_sec

    edgs_total_init_time_sec = (
        runtime_summary["edgs_base_init_time_sec"] + runtime_summary["edgs_extensions_init_time_sec"]
    )

    summary = {
        **parsed,
        "run_dir": str(run_dir),
        "expected_iterations": expected_iterations,
        "completed": completed,
        "status": "completed" if completed else "incomplete",
        "final_train_iteration": final_train_iteration,
        "final_num_gaussians": train_summary["final_num_gaussians"],
        "sum_iter_time_sec": train_summary["sum_iter_time_sec"],
        "final_eval_iteration": eval_summary["final_eval_iteration"],
        "final_eval_l1": eval_summary["final_eval_l1"],
        "final_eval_psnr": eval_summary["final_eval_psnr"],
        "total_training_time_sec": total_training_time_sec,
        "scene_load_time_sec": scene_load_time_sec,
        "init_time_sec": runtime_summary["init_time_sec"],
        "edgs_total_init_time_sec": edgs_total_init_time_sec,
        "peak_gpu_memory_mb_overall": runtime_summary["peak_gpu_memory_mb_overall"],
        "avg_gpu_memory_mb_training": runtime_summary["avg_gpu_memory_mb_training"],
        "peak_gpu_memory_mb_training": runtime_summary["peak_gpu_memory_mb_training"],
        "training_gpu_memory_gb_hours": training_gpu_memory_gb_hours,
        "end_to_end_time_sec": end_to_end_time_sec,
    }
    return summary


def _build_error_summary(run_dir, status):
    return {
        "run_name": run_dir.name,
        "status": status,
        "run_dir": str(run_dir),
        "scene_name": None,
        "scene_mode": "NA",
        "mode_label": None,
        "init_mode": "NA",
        "densify": None,
        "uses_edgs_train_recipe": None,
        "experiment_label": "NA",
        "lod_mode": "NA",
        "matched_resolution": False,
        "resolution": None,
        "expected_iterations": None,
        "completed": False,
        "final_train_iteration": None,
        "final_num_gaussians": None,
        "sum_iter_time_sec": None,
        "final_eval_iteration": None,
        "final_eval_l1": None,
        "final_eval_psnr": None,
        "total_training_time_sec": None,
        "scene_load_time_sec": None,
        "init_time_sec": None,
        "edgs_total_init_time_sec": None,
        "peak_gpu_memory_mb_overall": None,
        "avg_gpu_memory_mb_training": None,
        "peak_gpu_memory_mb_training": None,
        "training_gpu_memory_gb_hours": None,
        "end_to_end_time_sec": None,
    }


def _comparison_key(summary):
    return (
        summary["scene_mode"],
        summary["experiment_label"],
        summary["resolution"],
    )


def _format_float(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{value:.{digits}f}"


def _format_int(value):
    if value is None:
        return "NA"
    return str(int(value))


def _write_summary_csv(path, summaries):
    fieldnames = [
        "run_name",
        "status",
        "scene_name",
        "scene_mode",
        "init_mode",
        "densify",
        "uses_edgs_train_recipe",
        "experiment_label",
        "lod_mode",
        "matched_resolution",
        "resolution",
        "expected_iterations",
        "final_train_iteration",
        "final_eval_iteration",
        "final_eval_l1",
        "final_eval_psnr",
        "final_num_gaussians",
        "scene_load_time_sec",
        "total_training_time_sec",
        "end_to_end_time_sec",
        "peak_gpu_memory_mb_overall",
        "avg_gpu_memory_mb_training",
        "peak_gpu_memory_mb_training",
        "training_gpu_memory_gb_hours",
        "edgs_total_init_time_sec",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def _build_markdown_report(summaries, output_root, csv_path):
    completed = [summary for summary in summaries if summary["completed"]]
    incomplete = [summary for summary in summaries if not summary["completed"]]

    vanilla_refs = {
        _comparison_key(summary): summary
        for summary in completed
        if summary["init_mode"] == "vanilla"
    }

    lines = []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("# Run Cost Analysis")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append(f"Output root: `{output_root}`")
    lines.append("")
    lines.append(f"Summary CSV: `{csv_path}`")
    lines.append("")
    lines.append("## Cost Definition")
    lines.append("")
    lines.append("This report prioritizes rolled-up cost metrics rather than individual timing subcomponents.")
    lines.append("")
    lines.append("Primary cost metrics:")
    lines.append("")
    lines.append("- `end_to_end_time_sec`: `scene_load_time_sec + total_training_time_sec`")
    lines.append("- `peak_gpu_memory_mb_overall`: maximum reserved GPU memory captured in `runtime_metrics.csv`")
    lines.append("- `training_gpu_memory_gb_hours`: time-weighted reserved GPU-memory footprint during the training loop")
    lines.append("")
    lines.append("Important limitation:")
    lines.append("")
    lines.append("- exact GPU utilization or FLOP usage is not logged by the current pipeline")
    lines.append("- `training_gpu_memory_gb_hours` is therefore a cost proxy, not a direct compute-utilization measurement")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Run | Status | Scene | Init | Schedule | Match | Final PSNR | Final L1 | End-to-End Time (s) | Peak GPU (MB) | Training GPU GB-hours | Final Gaussians |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in sorted(summaries, key=lambda item: item["run_name"]):
        match_label = "matched" if summary["matched_resolution"] else "unmatched"
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["run_name"],
                    summary["status"],
                    summary["scene_mode"],
                    summary["init_mode"],
                    summary["experiment_label"],
                    match_label,
                    _format_float(summary["final_eval_psnr"]),
                    _format_float(summary["final_eval_l1"]),
                    _format_float(summary["end_to_end_time_sec"]),
                    _format_float(summary["peak_gpu_memory_mb_overall"]),
                    _format_float(summary["training_gpu_memory_gb_hours"]),
                    _format_int(summary["final_num_gaussians"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Improvement Over Vanilla")
    lines.append("")
    lines.append("| Run | Vanilla Reference | Delta PSNR | Delta L1 | Delta End-to-End Time (s) | Delta Peak GPU (MB) | Delta Training GPU GB-hours |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for summary in sorted(completed, key=lambda item: item["run_name"]):
        if summary["init_mode"] == "vanilla":
            continue
        reference = vanilla_refs.get(_comparison_key(summary))
        if reference is None:
            continue
        delta_psnr = None
        if summary["final_eval_psnr"] is not None and reference["final_eval_psnr"] is not None:
            delta_psnr = summary["final_eval_psnr"] - reference["final_eval_psnr"]
        delta_l1 = None
        if summary["final_eval_l1"] is not None and reference["final_eval_l1"] is not None:
            delta_l1 = summary["final_eval_l1"] - reference["final_eval_l1"]
        delta_time = None
        if summary["end_to_end_time_sec"] is not None and reference["end_to_end_time_sec"] is not None:
            delta_time = summary["end_to_end_time_sec"] - reference["end_to_end_time_sec"]
        delta_peak_gpu = None
        if summary["peak_gpu_memory_mb_overall"] is not None and reference["peak_gpu_memory_mb_overall"] is not None:
            delta_peak_gpu = summary["peak_gpu_memory_mb_overall"] - reference["peak_gpu_memory_mb_overall"]
        delta_gpu_gb_hours = None
        if summary["training_gpu_memory_gb_hours"] is not None and reference["training_gpu_memory_gb_hours"] is not None:
            delta_gpu_gb_hours = summary["training_gpu_memory_gb_hours"] - reference["training_gpu_memory_gb_hours"]

        lines.append(
            "| "
            + " | ".join(
                [
                    summary["run_name"],
                    reference["run_name"],
                    _format_float(delta_psnr),
                    _format_float(delta_l1),
                    _format_float(delta_time),
                    _format_float(delta_peak_gpu),
                    _format_float(delta_gpu_gb_hours),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Cost Winners")
    lines.append("")
    if completed:
        fastest = min(
            [summary for summary in completed if summary["end_to_end_time_sec"] is not None],
            key=lambda item: item["end_to_end_time_sec"],
            default=None,
        )
        best_psnr = max(
            [summary for summary in completed if summary["final_eval_psnr"] is not None],
            key=lambda item: item["final_eval_psnr"],
            default=None,
        )
        lowest_l1 = min(
            [summary for summary in completed if summary["final_eval_l1"] is not None],
            key=lambda item: item["final_eval_l1"],
            default=None,
        )
        lowest_cost_proxy = min(
            [summary for summary in completed if summary["training_gpu_memory_gb_hours"] is not None],
            key=lambda item: item["training_gpu_memory_gb_hours"],
            default=None,
        )
        if fastest is not None:
            lines.append(f"- Fastest end-to-end run: `{fastest['run_name']}` at `{_format_float(fastest['end_to_end_time_sec'])}` s")
        if best_psnr is not None:
            lines.append(f"- Best final PSNR: `{best_psnr['run_name']}` at `{_format_float(best_psnr['final_eval_psnr'])}`")
        if lowest_l1 is not None:
            lines.append(f"- Lowest final L1: `{lowest_l1['run_name']}` at `{_format_float(lowest_l1['final_eval_l1'])}`")
        if lowest_cost_proxy is not None:
            lines.append(
                f"- Lowest training GPU-memory cost proxy: `{lowest_cost_proxy['run_name']}` at "
                f"`{_format_float(lowest_cost_proxy['training_gpu_memory_gb_hours'])}` GB-hours"
            )
    lines.append("")

    if incomplete:
        lines.append("## Incomplete Runs")
        lines.append("")
        lines.append("| Run | Final Train Iteration | Expected Iterations | Last Eval Iteration |")
        lines.append("| --- | ---: | ---: | ---: |")
        for summary in sorted(incomplete, key=lambda item: item["run_name"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        summary["run_name"],
                        _format_int(summary["final_train_iteration"]),
                        _format_int(summary["expected_iterations"]),
                        _format_int(summary["final_eval_iteration"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `end_to_end_time_sec` includes setup through `scene_load_time_sec` plus loop-only `total_training_time_sec`.")
    lines.append("- `edgs_total_init_time_sec` is available in the summary CSV for EDGS-specific overhead attribution, but it is not treated as the primary cost metric here.")
    lines.append("- Split runs may have different test camera counts than base runs, so quality comparisons should remain within matched experiment families.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("gaussian-splatting/output"),
        help="Directory containing one subdirectory per run.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Path for the flat collated CSV. Defaults to <output_root>/collated_summary.csv",
    )
    parser.add_argument(
        "--report_md",
        type=Path,
        default=None,
        help="Path for the generated Markdown report. Defaults to <output_root>/cost_analysis.md",
    )
    args = parser.parse_args()

    output_root = args.output_root
    summary_csv = args.summary_csv or (output_root / "collated_summary.csv")
    report_md = args.report_md or (output_root / "cost_analysis.md")

    run_dirs = sorted([path for path in output_root.iterdir() if path.is_dir()])
    summaries = []
    for run_dir in run_dirs:
        try:
            summaries.append(_load_run_summary(run_dir))
        except Exception as exc:
            summaries.append(_build_error_summary(run_dir, f"parse_error: {exc}"))

    _write_summary_csv(summary_csv, summaries)
    report_md.write_text(_build_markdown_report(summaries, output_root, summary_csv))

    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote Markdown report to {report_md}")


if __name__ == "__main__":
    main()
