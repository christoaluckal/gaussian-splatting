#!/usr/bin/env python3
"""Launch and summarize a TartanAir split-iteration/cluster-count matrix."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SOURCE = "../bags/tartanair_packets"
DEFAULT_OUTPUT_ROOT = "output/tartan_splitter_matrix_runs"
DEFAULT_EDGS_CACHE_ROOT = "output/tartan_splitter_matrix_cache"
DEFAULT_WANDB_PROJECT = "openvins-home-matrix"
DEFAULT_WANDB_GROUP = "tartan-splitter-matrix"
DEFAULT_ITERATIONS = 40000
DEFAULT_DENSIFY_UNTIL_ITER = 30000
DEFAULT_GRAD_THRESHOLD = 1e-3
DEFAULT_CLUSTER_COUNTS = list(range(2, 11))
DEFAULT_SPLITTER_ITRS = list(range(1000, 15001, 1000))
DEFAULT_VIEWPOINT_SPLITTER = "pose_kmeans"
DEFAULT_VIEWPOINT_SPLITTER_CONFIG = '{"position_scale": 1.0, "forward_scale": 0.5, "max_iterations": 32}'
DEFAULT_BASELINE_RESOLUTION_SCALES = [2]
DEFAULT_LOD_RESOLUTION_SCALES = [2, 4, 8]
DEFAULT_NAIVE_LOD_STAGE_ITERATIONS = 5000
DEFAULT_DENSIFICATION_INTERVAL = 100

SUMMARY_CSV_NAME = "splitter_matrix_summary.csv"
REPORT_MD_NAME = "splitter_matrix_report.md"
HEATMAP_PNG_NAME = "splitter_matrix_heatmaps.png"


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected only positive integers")
    return values


def format_float_for_name(value: float) -> str:
    return f"{value:.1e}".replace(".0", "").replace(".", "p").replace("-", "m")


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_expected_iterations(cfg_args_path: Path):
    if not cfg_args_path.exists():
        return None
    content = cfg_args_path.read_text(errors="replace")
    match = re.search(r"iterations=(\d+)", content)
    if match is None:
        return None
    return int(match.group(1))


def _build_control_run_name(args: argparse.Namespace) -> str:
    grad_label = format_float_for_name(args.grad_threshold)
    scales_label = "x".join(str(scale) for scale in args.baseline_resolution_scales)
    return (
        "tartan_matrix_control_baseline"
        f"_r{scales_label}"
        f"_g{grad_label}"
        f"_iter{args.iterations}"
        f"_dens{args.densify_until_iter}"
    )


def _build_split_run_name(
    args: argparse.Namespace,
    *,
    cluster_count: int,
    splitter_itr: int,
    variant: str,
    resolution_scales: list[int],
) -> str:
    grad_label = format_float_for_name(args.grad_threshold)
    scales_label = "x".join(str(scale) for scale in resolution_scales)
    return (
        f"tartan_matrix_cluster{cluster_count}"
        f"_split{splitter_itr}"
        f"_{variant}"
        f"_r{scales_label}"
        f"_g{grad_label}"
        f"_iter{args.iterations}"
        f"_dens{args.densify_until_iter}"
    )


def _build_command(
    args: argparse.Namespace,
    *,
    run_name: str,
    cluster_count: int,
    splitter_itr: int | None,
    resolution_scales: list[int],
    control: bool,
) -> list[str]:
    model_path = str(Path(args.output_root) / run_name)
    xtend = 0 if control else max(cluster_count - 1, 0)
    command = [
        args.python,
        args.train_script,
        "-s",
        args.source,
        "-x",
        str(xtend),
        "--eval",
        "--iterations",
        str(args.iterations),
        "--resolution_scales",
        *[str(scale) for scale in resolution_scales],
        "--edgs_init",
        "--edgs_add_sfm_init",
        "--edgs_proj_err_tolerance",
        str(args.edgs_proj_err_tolerance),
        "--edgs_roma_model",
        args.edgs_roma_model,
        "--edgs_skip_frames",
        str(args.edgs_skip_frames),
        "--densify_until_iter",
        str(args.densify_until_iter),
        "--densification_interval",
        str(args.densification_interval),
        "--densify_grad_threshold",
        f"{args.grad_threshold:.12g}",
        "--disable_viewer",
        "--edgs_cache_root",
        args.edgs_cache_root,
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        run_name,
        "-m",
        model_path,
        "--edgs_matches_per_ref",
        str(args.edgs_matches_per_ref),
        "--naive_lod_stage_iterations",
        str(args.naive_lod_stage_iterations),
        "--edgs_num_refs",
        str(args.edgs_num_refs),
    ]
    if control:
        command.append("--default")
    else:
        command.extend(
            [
                "--splitter_itr",
                str(splitter_itr),
                "--viewpoint_splitter",
                args.viewpoint_splitter,
                "--viewpoint_splitter_config",
                args.viewpoint_splitter_config,
            ]
        )
    return command


def _post_append_densify_buffer(densification_interval: int) -> int:
    post_append_buffer = max(2000, densification_interval)
    if densification_interval > 0:
        post_append_buffer = (
            ((post_append_buffer + densification_interval - 1) // densification_interval)
            * densification_interval
        )
    return post_append_buffer


def _is_valid_split_schedule(
    *,
    cluster_count: int,
    splitter_itr: int,
    iterations: int,
    densification_interval: int,
) -> bool:
    # Require one splitter-sized phase window per cluster, plus a small tail so
    # the final block still gets post-append densification time before training ends.
    minimum_required_iterations = (
        cluster_count * splitter_itr
        + _post_append_densify_buffer(densification_interval)
    )
    return minimum_required_iterations <= iterations


def build_jobs(args: argparse.Namespace) -> list[list[str]]:
    jobs: list[list[str]] = []
    control_run_name = _build_control_run_name(args)
    jobs.append(
        _build_command(
            args,
            run_name=control_run_name,
            cluster_count=1,
            splitter_itr=None,
            resolution_scales=args.baseline_resolution_scales,
            control=True,
        )
    )

    variants = [
        ("baseline", args.baseline_resolution_scales),
        ("lod", args.lod_resolution_scales),
    ]
    for cluster_count in args.cluster_counts:
        for splitter_itr in args.splitter_itrs:
            if not _is_valid_split_schedule(
                cluster_count=cluster_count,
                splitter_itr=splitter_itr,
                iterations=args.iterations,
                densification_interval=args.densification_interval,
            ):
                continue
            for variant, resolution_scales in variants:
                run_name = _build_split_run_name(
                    args,
                    cluster_count=cluster_count,
                    splitter_itr=splitter_itr,
                    variant=variant,
                    resolution_scales=resolution_scales,
                )
                jobs.append(
                    _build_command(
                        args,
                        run_name=run_name,
                        cluster_count=cluster_count,
                        splitter_itr=splitter_itr,
                        resolution_scales=resolution_scales,
                        control=False,
                    )
                )
    return jobs


def _parse_run_metadata(run_name: str) -> dict[str, object] | None:
    control_match = re.fullmatch(
        r"tartan_matrix_control_baseline_r(?P<scales>[0-9x]+)_g(?P<grad>[a-z0-9]+)_iter(?P<iterations>\d+)_dens(?P<dens>\d+)",
        run_name,
    )
    if control_match is not None:
        return {
            "run_name": run_name,
            "control": True,
            "lod": False,
            "variant": "baseline",
            "cluster_count": 1,
            "splitter_itr": 0,
        }

    split_match = re.fullmatch(
        r"tartan_matrix_cluster(?P<cluster>\d+)_split(?P<splitter>\d+)_(?P<variant>baseline|lod)_r(?P<scales>[0-9x]+)_g(?P<grad>[a-z0-9]+)_iter(?P<iterations>\d+)_dens(?P<dens>\d+)",
        run_name,
    )
    if split_match is None:
        return None

    variant = str(split_match.group("variant"))
    return {
        "run_name": run_name,
        "control": False,
        "lod": variant == "lod",
        "variant": variant,
        "cluster_count": int(split_match.group("cluster")),
        "splitter_itr": int(split_match.group("splitter")),
    }


def _load_run_summary(run_dir: Path) -> dict[str, object] | None:
    metadata = _parse_run_metadata(run_dir.name)
    if metadata is None:
        return None

    train_rows = _read_csv_rows(run_dir / "train_metrics.csv")
    eval_rows = _read_csv_rows(run_dir / "eval_metrics.csv")
    runtime_rows = _read_csv_rows(run_dir / "runtime_metrics.csv")
    expected_iterations = _parse_expected_iterations(run_dir / "cfg_args")

    final_train_row = None
    if train_rows:
        final_train_row = max(train_rows, key=lambda row: _safe_int(row.get("iteration")) or -1)
    final_train_iteration = _safe_int(final_train_row.get("iteration")) if final_train_row else None
    final_num_gaussians = _safe_int(final_train_row.get("num_gaussians")) if final_train_row else None

    test_rows = [row for row in eval_rows if row.get("split") == "test"]
    final_eval_row = None
    best_eval_row = None
    if test_rows:
        final_eval_row = max(test_rows, key=lambda row: _safe_int(row.get("iteration")) or -1)
        rows_with_psnr = [row for row in test_rows if _safe_float(row.get("psnr")) is not None]
        if rows_with_psnr:
            best_eval_row = max(rows_with_psnr, key=lambda row: _safe_float(row.get("psnr")))

    scene_load_time_sec = None
    total_training_time_sec = None
    for row in runtime_rows:
        event = row.get("event")
        if event == "scene_load":
            scene_load_time_sec = _safe_float(row.get("scene_load_time_sec"))
        elif event == "training_complete":
            total_training_time_sec = _safe_float(row.get("total_training_time_sec"))

    end_to_end_time_sec = None
    if scene_load_time_sec is not None and total_training_time_sec is not None:
        end_to_end_time_sec = scene_load_time_sec + total_training_time_sec

    completed = total_training_time_sec is not None
    if expected_iterations is not None and final_train_iteration is not None:
        completed = completed and final_train_iteration >= expected_iterations

    return {
        **metadata,
        "run_dir": str(run_dir),
        "expected_iterations": expected_iterations,
        "completed": completed,
        "status": "completed" if completed else "incomplete",
        "final_train_iteration": final_train_iteration,
        "final_num_gaussians": final_num_gaussians,
        "final_eval_iteration": _safe_int(final_eval_row.get("iteration")) if final_eval_row else None,
        "final_eval_l1": _safe_float(final_eval_row.get("l1")) if final_eval_row else None,
        "final_eval_psnr": _safe_float(final_eval_row.get("psnr")) if final_eval_row else None,
        "best_eval_iteration": _safe_int(best_eval_row.get("iteration")) if best_eval_row else None,
        "best_eval_psnr": _safe_float(best_eval_row.get("psnr")) if best_eval_row else None,
        "scene_load_time_sec": scene_load_time_sec,
        "total_training_time_sec": total_training_time_sec,
        "end_to_end_time_sec": end_to_end_time_sec,
    }


def _is_completed_run_dir(run_dir: Path) -> bool:
    summary = _load_run_summary(run_dir)
    if summary is None:
        return False
    return bool(summary["completed"])


def _model_path_from_command(command: list[str]) -> Path:
    model_flag_idx = command.index("-m")
    return Path(command[model_flag_idx + 1])


def _write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_name",
        "status",
        "control",
        "lod",
        "variant",
        "cluster_count",
        "splitter_itr",
        "expected_iterations",
        "final_train_iteration",
        "final_num_gaussians",
        "final_eval_iteration",
        "final_eval_l1",
        "final_eval_psnr",
        "best_eval_iteration",
        "best_eval_psnr",
        "scene_load_time_sec",
        "total_training_time_sec",
        "end_to_end_time_sec",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: str(item["run_name"])):
            writer.writerow({key: summary.get(key) for key in fieldnames})


def _build_metric_matrix(
    summaries: list[dict[str, object]],
    *,
    variant: str,
    cluster_counts: list[int],
    splitter_itrs: list[int],
    metric: str,
) -> np.ndarray:
    matrix = np.full((len(cluster_counts), len(splitter_itrs)), np.nan, dtype=np.float64)
    cluster_index = {value: idx for idx, value in enumerate(cluster_counts)}
    splitter_index = {value: idx for idx, value in enumerate(splitter_itrs)}
    for summary in summaries:
        if summary["control"] or summary["variant"] != variant or not summary["completed"]:
            continue
        cluster_count = int(summary["cluster_count"])
        splitter_itr = int(summary["splitter_itr"])
        value = _safe_float(summary.get(metric))
        if value is None:
            continue
        matrix[cluster_index[cluster_count], splitter_index[splitter_itr]] = value
    return matrix


def _draw_heatmap(ax, data: np.ndarray, x_labels: list[int], y_labels: list[int], title: str, cmap: str) -> None:
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(label) for label in x_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(label) for label in y_labels])
    ax.set_xlabel("splitter_itr")
    ax.set_ylabel("cluster_count")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _write_heatmap_figure(
    path: Path,
    summaries: list[dict[str, object]],
    *,
    cluster_counts: list[int],
    splitter_itrs: list[int],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    metrics = [
        ("final_num_gaussians", "Final Gaussians", "viridis"),
        ("final_eval_psnr", "Final PSNR", "magma"),
        ("end_to_end_time_sec", "End-to-End Time (s)", "cividis"),
    ]
    variants = [("baseline", "Baseline"), ("lod", "LoD")]
    for row_idx, (variant, variant_label) in enumerate(variants):
        for col_idx, (metric, metric_label, cmap) in enumerate(metrics):
            matrix = _build_metric_matrix(
                summaries,
                variant=variant,
                cluster_counts=cluster_counts,
                splitter_itrs=splitter_itrs,
                metric=metric,
            )
            _draw_heatmap(
                axes[row_idx, col_idx],
                matrix,
                splitter_itrs,
                cluster_counts,
                f"{variant_label}: {metric_label}",
                cmap,
            )
    fig.suptitle("TartanAir Splitter Matrix (g=4e-4)", fontsize=14)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _choose_best(summary_candidates: list[dict[str, object]], *, mode: str):
    candidates = [summary for summary in summary_candidates if summary["completed"]]
    if not candidates:
        return None
    if mode == "fastest":
        return min(
            candidates,
            key=lambda item: (
                math.inf if item["end_to_end_time_sec"] is None else item["end_to_end_time_sec"],
                math.inf if item["final_eval_psnr"] is None else -item["final_eval_psnr"],
            ),
        )
    if mode == "best_psnr":
        return max(
            candidates,
            key=lambda item: (
                -math.inf if item["final_eval_psnr"] is None else item["final_eval_psnr"],
                -math.inf if item["best_eval_psnr"] is None else item["best_eval_psnr"],
            ),
        )
    if mode == "smallest_model":
        return min(
            candidates,
            key=lambda item: (
                math.inf if item["final_num_gaussians"] is None else item["final_num_gaussians"],
                math.inf if item["end_to_end_time_sec"] is None else item["end_to_end_time_sec"],
            ),
        )
    if mode == "time_primary":
        return min(
            candidates,
            key=lambda item: (
                math.inf if item["end_to_end_time_sec"] is None else item["end_to_end_time_sec"],
                math.inf if item["final_eval_psnr"] is None else -item["final_eval_psnr"],
                math.inf if item["final_num_gaussians"] is None else item["final_num_gaussians"],
            ),
        )
    raise ValueError(f"unsupported mode: {mode}")


def _fmt_float(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_markdown_report(path: Path, summaries: list[dict[str, object]], heatmap_path: Path) -> None:
    completed = [summary for summary in summaries if summary["completed"]]
    control = next((summary for summary in completed if summary["control"]), None)

    lines = [
        "# Splitter Matrix Report",
        "",
        f"Heatmap figure: `{heatmap_path.name}`",
        "",
        f"Total runs discovered: `{len(summaries)}`",
        f"Completed runs: `{len(completed)}`",
        "",
        "## Control",
        "",
    ]

    if control is None:
        lines.append("No completed non-split control run found.")
    else:
        lines.extend(
            [
                f"- Run: `{control['run_name']}`",
                f"- Final PSNR: `{_fmt_float(control['final_eval_psnr'])}`",
                f"- Final L1: `{_fmt_float(control['final_eval_l1'])}`",
                f"- End-to-end time (s): `{_fmt_float(control['end_to_end_time_sec'])}`",
                f"- Final Gaussians: `{control['final_num_gaussians']}`",
            ]
        )

    for variant in ("baseline", "lod"):
        variant_runs = [summary for summary in completed if not summary["control"] and summary["variant"] == variant]
        fastest = _choose_best(variant_runs, mode="fastest")
        best_psnr = _choose_best(variant_runs, mode="best_psnr")
        smallest_model = _choose_best(variant_runs, mode="smallest_model")
        time_primary = _choose_best(variant_runs, mode="time_primary")

        lines.extend(
            [
                "",
                f"## {variant.capitalize()} Split Runs",
                "",
                f"Completed runs: `{len(variant_runs)}`",
                "",
            ]
        )

        def add_summary(label: str, summary):
            if summary is None:
                lines.append(f"- {label}: `NA`")
                return
            lines.extend(
                [
                    f"- {label}: `{summary['run_name']}`",
                    f"  cluster_count=`{summary['cluster_count']}`, splitter_itr=`{summary['splitter_itr']}`, final_psnr=`{_fmt_float(summary['final_eval_psnr'])}`, time_s=`{_fmt_float(summary['end_to_end_time_sec'])}`, gaussians=`{summary['final_num_gaussians']}`",
                ]
            )

        add_summary("Fastest", fastest)
        add_summary("Best PSNR", best_psnr)
        add_summary("Smallest Model", smallest_model)
        add_summary("Time-Primary Winner", time_primary)

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Heatmaps use `splitter_itr` on the X axis and `cluster_count` on the Y axis.",
            "- Rows/columns with missing values indicate runs that are incomplete or absent.",
            "- `Time-Primary Winner` ranks by end-to-end time first, then by higher final PSNR, then by smaller final Gaussian count.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(output_root: Path, *, cluster_counts: list[int], splitter_itrs: list[int]) -> None:
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()] if output_root.exists() else []
    summaries = []
    for run_dir in run_dirs:
        summary = _load_run_summary(run_dir)
        if summary is not None:
            summaries.append(summary)

    summary_csv = output_root / SUMMARY_CSV_NAME
    report_md = output_root / REPORT_MD_NAME
    heatmap_png = output_root / HEATMAP_PNG_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(summary_csv, summaries)
    _write_heatmap_figure(
        heatmap_png,
        summaries,
        cluster_counts=cluster_counts,
        splitter_itrs=splitter_itrs,
    )
    _write_markdown_report(report_md, summaries, heatmap_png)
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote Markdown report to {report_md}")
    print(f"Wrote heatmap figure to {heatmap_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a TartanAir split-iteration/cluster-count matrix with fixed g=4e-4, "
            "plus generate a report over the resulting runs."
        )
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", default="train_nomask.py")
    parser.add_argument("--edgs-cache-root", default=DEFAULT_EDGS_CACHE_ROOT)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--densify-until-iter", type=int, default=DEFAULT_DENSIFY_UNTIL_ITER)
    parser.add_argument("--densification-interval", type=int, default=DEFAULT_DENSIFICATION_INTERVAL)
    parser.add_argument("--grad-threshold", type=float, default=DEFAULT_GRAD_THRESHOLD)
    parser.add_argument("--cluster-counts", type=parse_int_list, default=DEFAULT_CLUSTER_COUNTS)
    parser.add_argument("--splitter-itrs", type=parse_int_list, default=DEFAULT_SPLITTER_ITRS)
    parser.add_argument("--viewpoint-splitter", default=DEFAULT_VIEWPOINT_SPLITTER)
    parser.add_argument("--viewpoint-splitter-config", default=DEFAULT_VIEWPOINT_SPLITTER_CONFIG)
    parser.add_argument("--baseline-resolution-scales", type=parse_int_list, default=DEFAULT_BASELINE_RESOLUTION_SCALES)
    parser.add_argument("--lod-resolution-scales", type=parse_int_list, default=DEFAULT_LOD_RESOLUTION_SCALES)
    parser.add_argument("--naive-lod-stage-iterations", type=int, default=DEFAULT_NAIVE_LOD_STAGE_ITERATIONS)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--edgs-matches-per-ref", type=int, default=200)
    parser.add_argument("--edgs-num-refs", type=int, default=500)
    parser.add_argument("--edgs-proj-err-tolerance", type=float, default=0.01)
    parser.add_argument("--edgs-roma-model", default="outdoors")
    parser.add_argument("--edgs-skip-frames", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    parser.add_argument("--report-only", action="store_true", help="Skip launching jobs and only rebuild the summary outputs.")
    parser.add_argument("--skip-report", action="store_true", help="Skip rebuilding summary outputs after launch.")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip runs whose output directories already look completed. Enabled by default.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable completion-based skipping and relaunch every job.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue launching later jobs after a failed training process.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)

    if args.report_only:
        generate_report(
            output_root,
            cluster_counts=args.cluster_counts,
            splitter_itrs=args.splitter_itrs,
        )
        return

    jobs = build_jobs(args)
    print(f"Prepared {len(jobs)} jobs.")
    print("Control family: one EDGS no-split no-LoD baseline")
    print(f"Cluster counts: {args.cluster_counts}")
    print(f"Splitter iterations: {args.splitter_itrs}")
    print(f"Grad threshold: {args.grad_threshold}")
    print(f"Iterations: {args.iterations}")
    print(f"Densify until iteration: {args.densify_until_iter}")
    print(f"Densification interval: {args.densification_interval}")
    print(f"Baseline resolution scales: {args.baseline_resolution_scales}")
    print(f"LoD resolution scales: {args.lod_resolution_scales}")
    print(f"EDGS cache root: {args.edgs_cache_root}")
    print(f"W&B project/group: {args.wandb_project}/{args.wandb_group}")
    print(f"Resume mode: {'enabled' if args.resume else 'disabled'}")
    print(
        "Split feasibility rule: "
        "cluster_count * splitter_itr + post_append_densify_buffer(densification_interval) <= iterations"
    )

    skipped_jobs = 0
    launched_jobs = 0
    interrupted = False
    for index, command in enumerate(jobs, start=1):
        model_path = _model_path_from_command(command)
        should_skip = args.resume and _is_completed_run_dir(model_path)
        status_prefix = "skip" if should_skip else "run"
        print(f"\n[{index}/{len(jobs)}:{status_prefix}] {' '.join(command)}")
        if args.dry_run:
            continue
        if should_skip:
            skipped_jobs += 1
            continue
        try:
            subprocess.run(command, check=True)
            launched_jobs += 1
        except KeyboardInterrupt:
            interrupted = True
            print(
                "\n[INTERRUPTED] Stopping launch loop. "
                "Re-run this command to resume unfinished jobs."
            )
            break
        except Exception as exc:
            launched_jobs += 1
            if not args.continue_on_error:
                raise
            print(
                f"[WARN] job {index} failed with {type(exc).__name__}: {exc}; "
                "continuing because --continue-on-error was set"
            )
        finally:
            gc.collect()

    if not args.dry_run:
        print(f"Skipped completed jobs: {skipped_jobs}")
        print(f"Launched jobs this session: {launched_jobs}")

    if not args.skip_report and not args.dry_run:
        generate_report(
            output_root,
            cluster_counts=args.cluster_counts,
            splitter_itrs=args.splitter_itrs,
        )
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
