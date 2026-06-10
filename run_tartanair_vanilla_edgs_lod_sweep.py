#!/usr/bin/env python3
"""Run a three-way vanilla, EDGS, and EDGS+LoD TartanAir comparison."""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE = "../bags/tartanair_colmap_vanilla_full/CyberPunkDowntown_P0000"
DEFAULT_OUTPUT_ROOT = "output/tartanair_vanilla_edgs_lod_sweep"
DEFAULT_EDGS_CACHE_ROOT = "output/tartanair_vanilla_edgs_lod_cache"
DEFAULT_WANDB_PROJECT = "tartanair-colmap-vanilla-edgs-lod"
DEFAULT_WANDB_GROUP = "cyberpunkdowntown-cluster3-g1e-3"
DEFAULT_ITERATIONS = 40000
DEFAULT_DENSIFY_UNTIL_ITER = 30000
DEFAULT_DENSIFICATION_INTERVAL = 100
DEFAULT_GRAD_THRESHOLD = 1e-3
DEFAULT_CLUSTER_COUNT = 3
DEFAULT_SPLITTER_ITR = 10000
DEFAULT_VIEWPOINT_SPLITTER = "pose_kmeans"
DEFAULT_VIEWPOINT_SPLITTER_CONFIG = (
    '{"position_scale": 1.0, "forward_scale": 0.5, "max_iterations": 32}'
)
DEFAULT_BASELINE_RESOLUTION_SCALES = [2]
DEFAULT_LOD_RESOLUTION_SCALES = [2, 4, 8]
DEFAULT_NAIVE_LOD_STAGE_ITERATIONS = 5000


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected only positive integers")
    return values


def format_float_for_name(value: float) -> str:
    return f"{value:.1e}".replace(".0", "").replace(".", "p").replace("-", "m")


def build_run_name(
    args: argparse.Namespace,
    *,
    variant: str,
    resolution_scales: list[int],
    clustered: bool,
) -> str:
    scene_name = Path(args.source).name
    scale_label = "x".join(str(scale) for scale in resolution_scales)
    grad_label = format_float_for_name(args.grad_threshold)
    split_label = (
        f"cluster{args.cluster_count}_split{args.splitter_itr}"
        if clustered
        else "no_cluster"
    )
    return (
        f"{scene_name}_{split_label}"
        f"_{variant}_r{scale_label}_g{grad_label}"
        f"_iter{args.iterations}_dens{args.densify_until_iter}"
    )


def build_command(
    args: argparse.Namespace,
    *,
    variant: str,
    resolution_scales: list[int],
    edgs: bool,
    clustered: bool,
) -> list[str]:
    run_name = build_run_name(
        args,
        variant=variant,
        resolution_scales=resolution_scales,
        clustered=clustered,
    )
    command = [
        args.python,
        args.train_script,
        "-s",
        args.source,
        "-x",
        str(args.cluster_count - 1 if clustered else 0),
        "--eval",
        "--iterations",
        str(args.iterations),
        "--resolution_scales",
        *[str(scale) for scale in resolution_scales],
        "--densify_until_iter",
        str(args.densify_until_iter),
        "--densification_interval",
        str(args.densification_interval),
        "--densify_grad_threshold",
        f"{args.grad_threshold:.12g}",
        "--naive_lod_stage_iterations",
        str(args.naive_lod_stage_iterations),
        "--disable_viewer",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        run_name,
        "-m",
        str(Path(args.output_root) / run_name),
    ]
    if clustered:
        command.extend(
            [
                "--splitter_itr",
                str(args.splitter_itr),
                "--viewpoint_splitter",
                args.viewpoint_splitter,
                "--viewpoint_splitter_config",
                args.viewpoint_splitter_config,
            ]
        )
    else:
        command.append("--default")
    if edgs:
        command.extend(
            [
                "--edgs_init",
                "--edgs_add_sfm_init",
                "--edgs_cache_root",
                args.edgs_cache_root,
                "--edgs_proj_err_tolerance",
                str(args.edgs_proj_err_tolerance),
                "--edgs_roma_model",
                args.edgs_roma_model,
                "--edgs_skip_frames",
                str(args.edgs_skip_frames),
                "--edgs_matches_per_ref",
                str(args.edgs_matches_per_ref),
                "--edgs_num_refs",
                str(args.edgs_num_refs),
            ]
        )
    return command


def build_jobs(args: argparse.Namespace) -> list[list[str]]:
    variants = [
        ("vanilla_no_lod", args.baseline_resolution_scales, False, False),
        ("edgs_no_lod", args.baseline_resolution_scales, True, True),
        ("edgs_lod", args.lod_resolution_scales, True, True),
    ]
    return [
        build_command(
            args,
            variant=variant,
            resolution_scales=resolution_scales,
            edgs=edgs,
            clustered=clustered,
        )
        for variant, resolution_scales, edgs, clustered in variants
    ]


def model_path_from_command(command: list[str]) -> Path:
    return Path(command[command.index("-m") + 1])


def completed_run(model_path: Path, iterations: int) -> bool:
    runtime_metrics = model_path / "runtime_metrics.csv"
    train_metrics = model_path / "train_metrics.csv"
    if not runtime_metrics.exists() or not train_metrics.exists():
        return False
    if "training_complete" not in runtime_metrics.read_text(errors="replace"):
        return False
    lines = train_metrics.read_text(errors="replace").splitlines()
    return any(line.startswith(f"{iterations},") for line in lines[1:])


def validate_args(args: argparse.Namespace) -> None:
    if args.cluster_count < 1:
        raise ValueError("cluster-count must be positive")
    if args.splitter_itr <= 0:
        raise ValueError("splitter-itr must be positive")
    if args.densify_until_iter >= args.iterations:
        raise ValueError("densify-until-iter must be less than iterations")
    final_append_iteration = (args.cluster_count - 1) * args.splitter_itr
    if final_append_iteration >= args.iterations:
        raise ValueError(
            "the final cluster append must occur before training finishes: "
            f"{final_append_iteration} >= {args.iterations}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-clustered vanilla TartanAir baseline followed by "
            "three-cluster EDGS comparisons without and with LoD."
        )
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--edgs-cache-root", default=DEFAULT_EDGS_CACHE_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", default="train_nomask.py")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--densify-until-iter", type=int, default=DEFAULT_DENSIFY_UNTIL_ITER)
    parser.add_argument("--densification-interval", type=int, default=DEFAULT_DENSIFICATION_INTERVAL)
    parser.add_argument("--grad-threshold", type=float, default=DEFAULT_GRAD_THRESHOLD)
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--splitter-itr", type=int, default=DEFAULT_SPLITTER_ITR)
    parser.add_argument("--viewpoint-splitter", default=DEFAULT_VIEWPOINT_SPLITTER)
    parser.add_argument("--viewpoint-splitter-config", default=DEFAULT_VIEWPOINT_SPLITTER_CONFIG)
    parser.add_argument(
        "--baseline-resolution-scales",
        type=parse_int_list,
        default=DEFAULT_BASELINE_RESOLUTION_SCALES,
    )
    parser.add_argument(
        "--lod-resolution-scales",
        type=parse_int_list,
        default=DEFAULT_LOD_RESOLUTION_SCALES,
    )
    parser.add_argument(
        "--naive-lod-stage-iterations",
        type=int,
        default=DEFAULT_NAIVE_LOD_STAGE_ITERATIONS,
    )
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--edgs-matches-per-ref", type=int, default=200)
    parser.add_argument("--edgs-num-refs", type=int, default=500)
    parser.add_argument("--edgs-proj-err-tolerance", type=float, default=0.01)
    parser.add_argument("--edgs-roma-model", default="outdoors")
    parser.add_argument("--edgs-skip-frames", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip completed output directories. Enabled by default.",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    validate_args(args)

    jobs = build_jobs(args)
    print(f"Prepared {len(jobs)} jobs for {args.source}")
    print(
        f"cluster_count={args.cluster_count}, splitter_itr={args.splitter_itr}, "
        f"grad_threshold={args.grad_threshold}"
    )
    print(f"W&B project/group: {args.wandb_project}/{args.wandb_group}")

    for index, command in enumerate(jobs, start=1):
        model_path = model_path_from_command(command)
        should_skip = args.resume and completed_run(model_path, args.iterations)
        status = "skip" if should_skip else "run"
        print(f"\n[{index}/{len(jobs)}:{status}] {' '.join(command)}")
        if args.dry_run or should_skip:
            continue
        try:
            subprocess.run(command, check=True)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Re-run the same command to resume.")
            raise SystemExit(130)
        except subprocess.CalledProcessError:
            if not args.continue_on_error:
                raise
            print(f"[WARN] Job {index} failed; continuing.")
        finally:
            gc.collect()


if __name__ == "__main__":
    main()
