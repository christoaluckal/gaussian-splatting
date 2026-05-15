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
MILESTONES = [1000, 5000, 10000, 20000, 30000]


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


def _parse_source_scene_name(cfg_args_path):
    if not cfg_args_path.exists():
        return None
    content = cfg_args_path.read_text(errors="replace")
    match = re.search(r"source_path='([^']+)'", content)
    if match is None:
        match = re.search(r'source_path="([^"]+)"', content)
    if match is None:
        return None
    return Path(match.group(1)).name


def _parse_run_name(run_name):
    try:
        return _parse_structured_run_name(run_name)
    except ValueError:
        return _parse_loose_run_name(run_name)


def _parse_structured_run_name(run_name):
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


def _parse_loose_run_name(run_name):
    is_edgs = "edgs" in run_name
    uses_edgs_recipe = "edgs-train-recipe" in run_name
    densify = "no_densify" not in run_name and "no-densify" not in run_name
    scene_mode = "split" if "_split" in run_name else "base"
    matched = "matched" in run_name

    scale_match = re.search(r"(?:lod|scale|r)_((?:\d+_)*\d+)", run_name)
    scale_tokens = []
    if scale_match:
        scale_tokens = [int(token) for token in scale_match.group(1).split("_") if token.isdigit()]
    resolution = min(scale_tokens) if scale_tokens else None

    if len(scale_tokens) > 1:
        experiment_label = "matched-naive-lod" if matched else "naive-lod"
        lod_mode = "lod"
    else:
        experiment_label = "baseline"
        lod_mode = "baseline"

    scene_name = run_name
    marker_match = re.search(r"_(?:edgs|vanilla|sfm)", run_name)
    if marker_match:
        scene_name = run_name[: marker_match.start()]

    if is_edgs:
        mode_label = "edgs-init-densify" if densify else "edgs-init-no-densify"
    else:
        mode_label = "vanilla" if densify else "sfm-init-no-densify"

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
    event_gpu_memory = {}

    for row in rows:
        event = row.get("event", "")
        gpu_memory_mb = _safe_float(row.get("gpu_memory_mb"))
        if gpu_memory_mb is not None:
            event_gpu_memory[event] = gpu_memory_mb
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
        "post_scene_init_gpu_memory_mb": event_gpu_memory.get("post_scene_init"),
        "post_scene_init_peak_gpu_memory_mb": event_gpu_memory.get("post_scene_init_peak"),
        "edgs_base_init_peak_gpu_memory_mb": event_gpu_memory.get("edgs_base_init_peak"),
        "edgs_extensions_init_peak_gpu_memory_mb": event_gpu_memory.get("edgs_extensions_init_peak"),
        "post_initialization_gpu_memory_mb": event_gpu_memory.get("post_initialization"),
        "post_initialization_peak_gpu_memory_mb": event_gpu_memory.get("post_initialization_peak"),
        "peak_gpu_memory_mb_overall": max(gpu_values_all) if gpu_values_all else None,
        "avg_gpu_memory_mb_training": mean(gpu_values_iteration) if gpu_values_iteration else None,
        "peak_gpu_memory_mb_training": max(gpu_values_iteration) if gpu_values_iteration else None,
    }


def _summarize_eval_rows(rows):
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        summary = {
            "final_eval_iteration": None,
            "final_eval_l1": None,
            "final_eval_psnr": None,
            "best_eval_iteration": None,
            "best_eval_psnr": None,
        }
        for milestone in MILESTONES:
            summary[f"psnr_at_{milestone}"] = None
        return summary
    final_row = max(test_rows, key=lambda row: _safe_int(row.get("iteration")) or -1)
    rows_with_psnr = [row for row in test_rows if _safe_float(row.get("psnr")) is not None]
    best_row = max(rows_with_psnr, key=lambda row: _safe_float(row.get("psnr"))) if rows_with_psnr else None
    summary = {
        "final_eval_iteration": _safe_int(final_row.get("iteration")),
        "final_eval_l1": _safe_float(final_row.get("l1")),
        "final_eval_psnr": _safe_float(final_row.get("psnr")),
        "best_eval_iteration": _safe_int(best_row.get("iteration")) if best_row else None,
        "best_eval_psnr": _safe_float(best_row.get("psnr")) if best_row else None,
    }
    rows_by_iteration = {
        _safe_int(row.get("iteration")): row
        for row in test_rows
        if _safe_int(row.get("iteration")) is not None
    }
    sorted_iterations = sorted(rows_by_iteration)
    for milestone in MILESTONES:
        selected_iteration = None
        for iteration in sorted_iterations:
            if iteration <= milestone:
                selected_iteration = iteration
            else:
                break
        selected_row = rows_by_iteration.get(selected_iteration) if selected_iteration is not None else None
        summary[f"psnr_at_{milestone}"] = _safe_float(selected_row.get("psnr")) if selected_row else None
    return summary


def _summarize_train_rows(rows):
    if not rows:
        summary = {
            "final_train_iteration": None,
            "final_num_gaussians": None,
            "sum_iter_time_sec": None,
            "observed_lod_scales": None,
        }
        for milestone in MILESTONES:
            summary[f"gaussians_at_{milestone}"] = None
        return summary

    final_row = max(rows, key=lambda row: _safe_int(row.get("iteration")) or -1)
    final_iteration = _safe_int(final_row.get("iteration"))
    final_num_gaussians = _safe_int(final_row.get("num_gaussians"))
    observed_lod_scales = []
    seen_lod_scales = set()
    rows_by_iteration = {}

    sum_iter_time_sec = 0.0
    for row in rows:
        iteration = _safe_int(row.get("iteration"))
        if iteration is not None:
            rows_by_iteration[iteration] = row
        lod_scale = _safe_int(row.get("lod_scale"))
        if lod_scale is not None and lod_scale not in seen_lod_scales:
            observed_lod_scales.append(lod_scale)
            seen_lod_scales.add(lod_scale)
        iter_time_ms = _safe_float(row.get("iter_time_ms"))
        if iter_time_ms is not None:
            sum_iter_time_sec += iter_time_ms / 1000.0

    summary = {
        "final_train_iteration": final_iteration,
        "final_num_gaussians": final_num_gaussians,
        "sum_iter_time_sec": sum_iter_time_sec,
        "observed_lod_scales": ",".join(str(scale) for scale in observed_lod_scales),
    }
    sorted_iterations = sorted(rows_by_iteration)
    for milestone in MILESTONES:
        selected_iteration = None
        for iteration in sorted_iterations:
            if iteration <= milestone:
                selected_iteration = iteration
            else:
                break
        selected_row = rows_by_iteration.get(selected_iteration) if selected_iteration is not None else None
        summary[f"gaussians_at_{milestone}"] = _safe_int(selected_row.get("num_gaussians")) if selected_row else None
    return summary


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
    source_scene_name = _parse_source_scene_name(cfg_args_path)
    observed_lod_scales = train_summary["observed_lod_scales"]
    observed_lod_scale_tokens = [
        token
        for token in str(observed_lod_scales or "").split(",")
        if _safe_int(token) is not None
    ]
    if source_scene_name is not None:
        parsed["scene_name"] = source_scene_name
    if edgs_total_init_time_sec > 0.0:
        parsed["init_mode"] = "edgs"
        parsed["mode_label"] = "edgs-init-densify" if parsed["densify"] else "edgs-init-no-densify"
    if observed_lod_scale_tokens:
        parsed["resolution"] = min(_safe_int(token) for token in observed_lod_scale_tokens)
    if len(observed_lod_scale_tokens) > 1:
        parsed["lod_mode"] = "lod"
        parsed["experiment_label"] = "matched-naive-lod" if parsed["matched_resolution"] else "naive-lod"
    elif observed_lod_scale_tokens:
        parsed["lod_mode"] = "baseline"
        parsed["experiment_label"] = "baseline"

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
        "best_eval_iteration": eval_summary["best_eval_iteration"],
        "best_eval_psnr": eval_summary["best_eval_psnr"],
        "total_training_time_sec": total_training_time_sec,
        "scene_load_time_sec": scene_load_time_sec,
        "init_time_sec": runtime_summary["init_time_sec"],
        "edgs_total_init_time_sec": edgs_total_init_time_sec,
        "post_scene_init_gpu_memory_mb": runtime_summary["post_scene_init_gpu_memory_mb"],
        "post_scene_init_peak_gpu_memory_mb": runtime_summary["post_scene_init_peak_gpu_memory_mb"],
        "edgs_base_init_peak_gpu_memory_mb": runtime_summary["edgs_base_init_peak_gpu_memory_mb"],
        "edgs_extensions_init_peak_gpu_memory_mb": runtime_summary["edgs_extensions_init_peak_gpu_memory_mb"],
        "post_initialization_gpu_memory_mb": runtime_summary["post_initialization_gpu_memory_mb"],
        "post_initialization_peak_gpu_memory_mb": runtime_summary["post_initialization_peak_gpu_memory_mb"],
        "peak_gpu_memory_mb_overall": runtime_summary["peak_gpu_memory_mb_overall"],
        "avg_gpu_memory_mb_training": runtime_summary["avg_gpu_memory_mb_training"],
        "peak_gpu_memory_mb_training": runtime_summary["peak_gpu_memory_mb_training"],
        "training_gpu_memory_gb_hours": training_gpu_memory_gb_hours,
        "end_to_end_time_sec": end_to_end_time_sec,
        "observed_lod_scales": observed_lod_scales,
    }
    for milestone in MILESTONES:
        summary[f"psnr_at_{milestone}"] = eval_summary[f"psnr_at_{milestone}"]
        summary[f"gaussians_at_{milestone}"] = train_summary[f"gaussians_at_{milestone}"]
    return summary


def _build_error_summary(run_dir, status):
    summary = {
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
        "best_eval_iteration": None,
        "best_eval_psnr": None,
        "total_training_time_sec": None,
        "scene_load_time_sec": None,
        "init_time_sec": None,
        "edgs_total_init_time_sec": None,
        "post_scene_init_gpu_memory_mb": None,
        "post_scene_init_peak_gpu_memory_mb": None,
        "edgs_base_init_peak_gpu_memory_mb": None,
        "edgs_extensions_init_peak_gpu_memory_mb": None,
        "post_initialization_gpu_memory_mb": None,
        "post_initialization_peak_gpu_memory_mb": None,
        "peak_gpu_memory_mb_overall": None,
        "avg_gpu_memory_mb_training": None,
        "peak_gpu_memory_mb_training": None,
        "training_gpu_memory_gb_hours": None,
        "end_to_end_time_sec": None,
        "observed_lod_scales": None,
    }
    for milestone in MILESTONES:
        summary[f"psnr_at_{milestone}"] = None
        summary[f"gaussians_at_{milestone}"] = None
    return summary


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
        "best_eval_iteration",
        "best_eval_psnr",
        "final_num_gaussians",
        "observed_lod_scales",
        "scene_load_time_sec",
        "total_training_time_sec",
        "end_to_end_time_sec",
        "post_scene_init_gpu_memory_mb",
        "post_scene_init_peak_gpu_memory_mb",
        "edgs_base_init_peak_gpu_memory_mb",
        "edgs_extensions_init_peak_gpu_memory_mb",
        "post_initialization_gpu_memory_mb",
        "post_initialization_peak_gpu_memory_mb",
        "peak_gpu_memory_mb_overall",
        "avg_gpu_memory_mb_training",
        "peak_gpu_memory_mb_training",
        "training_gpu_memory_gb_hours",
        "edgs_total_init_time_sec",
        "run_dir",
    ]
    insert_at = fieldnames.index("scene_load_time_sec")
    milestone_fields = []
    for milestone in MILESTONES:
        milestone_fields.extend([f"psnr_at_{milestone}", f"gaussians_at_{milestone}"])
    fieldnames[insert_at:insert_at] = milestone_fields
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
    lines.append("- `post_initialization_peak_gpu_memory_mb`: maximum reserved GPU memory from scene construction through EDGS initialization and iteration-0 save")
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
    lines.append("| Run | Status | Scene | Init | Schedule | Observed LoD | Match | Final PSNR | Best PSNR | Final L1 | End-to-End Time (s) | Post Init Max GPU (MB) | Peak GPU (MB) | Training GPU GB-hours | Final Gaussians |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
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
                    summary.get("observed_lod_scales") or "NA",
                    match_label,
                    _format_float(summary["final_eval_psnr"]),
                    _format_float(summary["best_eval_psnr"]),
                    _format_float(summary["final_eval_l1"]),
                    _format_float(summary["end_to_end_time_sec"]),
                    _format_float(summary["post_initialization_peak_gpu_memory_mb"]),
                    _format_float(summary["peak_gpu_memory_mb_overall"]),
                    _format_float(summary["training_gpu_memory_gb_hours"]),
                    _format_int(summary["final_num_gaussians"]),
                ]
            )
            + " |"
        )
    lines.append("")

    schedule_warnings = []
    for summary in summaries:
        observed_lod_scales = [
            scale
            for scale in str(summary.get("observed_lod_scales") or "").split(",")
            if scale
        ]
        if summary["lod_mode"] == "lod" and len(observed_lod_scales) <= 1:
            schedule_warnings.append(
                f"- `{summary['run_name']}` is labeled as LoD but only observed "
                f"`{summary.get('observed_lod_scales') or 'NA'}` in `train_metrics.csv`."
            )
    if schedule_warnings:
        lines.append("## Schedule Warnings")
        lines.append("")
        lines.extend(schedule_warnings)
        lines.append("")

    edgs_completed = [
        summary
        for summary in completed
        if summary["init_mode"] == "edgs" and summary["scene_mode"] == "base"
    ]
    edgs_baselines = [summary for summary in edgs_completed if summary["lod_mode"] == "baseline"]
    edgs_lod_runs = [summary for summary in edgs_completed if summary["lod_mode"] == "lod"]
    if edgs_baselines and edgs_lod_runs:
        lines.append("## EDGS-only vs EDGS+LoD")
        lines.append("")
        lines.append("This section compares non-incremental EDGS runs. It does not make claims about incremental append behavior.")
        lines.append("")
        lines.append("| EDGS-only Run | EDGS+LoD Run | Delta Final PSNR | Delta Best PSNR | Delta Final L1 | Delta Time (s) | Delta Post Init Max GPU (MB) | Delta Peak GPU (MB) | Delta Final Gaussians |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for baseline in sorted(edgs_baselines, key=lambda item: item["run_name"]):
            for lod_run in sorted(edgs_lod_runs, key=lambda item: item["run_name"]):
                same_scene = baseline["scene_name"] == lod_run["scene_name"]
                if not same_scene:
                    continue
                delta_final_psnr = None
                if baseline["final_eval_psnr"] is not None and lod_run["final_eval_psnr"] is not None:
                    delta_final_psnr = lod_run["final_eval_psnr"] - baseline["final_eval_psnr"]
                delta_best_psnr = None
                if baseline["best_eval_psnr"] is not None and lod_run["best_eval_psnr"] is not None:
                    delta_best_psnr = lod_run["best_eval_psnr"] - baseline["best_eval_psnr"]
                delta_l1 = None
                if baseline["final_eval_l1"] is not None and lod_run["final_eval_l1"] is not None:
                    delta_l1 = lod_run["final_eval_l1"] - baseline["final_eval_l1"]
                delta_time = None
                if baseline["end_to_end_time_sec"] is not None and lod_run["end_to_end_time_sec"] is not None:
                    delta_time = lod_run["end_to_end_time_sec"] - baseline["end_to_end_time_sec"]
                delta_peak_gpu = None
                if baseline["peak_gpu_memory_mb_overall"] is not None and lod_run["peak_gpu_memory_mb_overall"] is not None:
                    delta_peak_gpu = lod_run["peak_gpu_memory_mb_overall"] - baseline["peak_gpu_memory_mb_overall"]
                delta_post_init_peak_gpu = None
                if (
                    baseline["post_initialization_peak_gpu_memory_mb"] is not None
                    and lod_run["post_initialization_peak_gpu_memory_mb"] is not None
                ):
                    delta_post_init_peak_gpu = (
                        lod_run["post_initialization_peak_gpu_memory_mb"]
                        - baseline["post_initialization_peak_gpu_memory_mb"]
                    )
                delta_gaussians = None
                if baseline["final_num_gaussians"] is not None and lod_run["final_num_gaussians"] is not None:
                    delta_gaussians = lod_run["final_num_gaussians"] - baseline["final_num_gaussians"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            baseline["run_name"],
                            lod_run["run_name"],
                            _format_float(delta_final_psnr),
                            _format_float(delta_best_psnr),
                            _format_float(delta_l1),
                            _format_float(delta_time),
                            _format_float(delta_post_init_peak_gpu),
                            _format_float(delta_peak_gpu),
                            _format_int(delta_gaussians),
                        ]
                    )
                    + " |"
                )
        lines.append("")
        lines.append("### EDGS Convergence Milestones")
        lines.append("")
        lines.append("| Run | PSNR@1k | PSNR@5k | PSNR@10k | PSNR@20k | PSNR@30k | Gaussians@1k | Gaussians@10k | Gaussians@30k |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for summary in sorted(edgs_completed, key=lambda item: item["run_name"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        summary["run_name"],
                        _format_float(summary["psnr_at_1000"]),
                        _format_float(summary["psnr_at_5000"]),
                        _format_float(summary["psnr_at_10000"]),
                        _format_float(summary["psnr_at_20000"]),
                        _format_float(summary["psnr_at_30000"]),
                        _format_int(summary["gaussians_at_1000"]),
                        _format_int(summary["gaussians_at_10000"]),
                        _format_int(summary["gaussians_at_30000"]),
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
