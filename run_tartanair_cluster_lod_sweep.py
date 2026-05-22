#!/usr/bin/env python3
"""Launch paired clustered TartanAir baseline and LoD sweeps for train_nomask.py."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE = "../bags/tartanair_packets"
DEFAULT_OUTPUT_ROOT = "output/tartan_cluster_lod_sweep"
DEFAULT_WANDB_PROJECT = "openvins-home-cluster"
DEFAULT_WANDB_GROUP = "tartan-cluster-lod-densify-sweep"
DEFAULT_ITERATIONS = 30000
DEFAULT_CLUSTER_COUNT = 3
DEFAULT_SPLITTER_ITRS = [5000, 10000]
DEFAULT_POST_LOD_DENSIFY_MARGIN = 1000
DEFAULT_SCHEDULES = [(5000, 15000)]
DEFAULT_VIEWPOINT_SPLITTER = "pose_kmeans"
DEFAULT_VIEWPOINT_SPLITTER_CONFIG = '{"position_scale": 1.0, "forward_scale": 0.5, "max_iterations": 32}'


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_schedule(value: str) -> list[tuple[int, int]]:
    schedules: list[tuple[int, int]] = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "schedule entries must be NAIVE_LOD_STAGE_ITERATIONS:DENSIFY_UNTIL_ITER"
            )
        schedules.append((int(fields[0]), int(fields[1])))
    if not schedules:
        raise argparse.ArgumentTypeError("expected at least one schedule")
    return schedules


def parse_optional_int_list(value: str) -> list[int]:
    stripped = value.strip()
    if not stripped:
        return []
    values = [int(item) for item in stripped.replace(",", " ").split()]
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected only positive integers")
    return values


def format_float_for_name(value: float) -> str:
    return f"{value:.1e}".replace(".0", "").replace(".", "p").replace("-", "m")


def build_grad_thresholds(start: float, end: float, steps: int) -> list[float]:
    if steps < 2:
        return [start]
    stride = (end - start) / float(steps - 1)
    return [start + stride * idx for idx in range(steps)]


def validate_schedules(
    schedules: list[tuple[int, int]],
    iterations: int,
    post_lod_densify_margin: int,
) -> None:
    if post_lod_densify_margin < 0:
        raise ValueError("post_lod_densify_margin must be non-negative")
    for naive_lod_stage_iterations, densify_until_iter in schedules:
        if naive_lod_stage_iterations <= 0:
            raise ValueError("naive_lod_stage_iterations must be positive")
        if densify_until_iter <= 0:
            raise ValueError("densify_until_iter must be positive")
        min_densify_until_iter = 1 * naive_lod_stage_iterations + post_lod_densify_margin
        if min_densify_until_iter >= densify_until_iter:
            raise ValueError(
                "schedule violates "
                "naive_lod_stage_iterations + post_lod_densify_margin < densify_until_iter: "
                f"{naive_lod_stage_iterations}:{densify_until_iter} with margin "
                f"{post_lod_densify_margin}"
            )
        if densify_until_iter >= iterations:
            raise ValueError(
                "densify_until_iter should be less than total iterations: "
                f"{densify_until_iter} >= {iterations}"
            )


def build_command(
    *,
    args: argparse.Namespace,
    split_mode: str,
    variant: str,
    resolution_scales: list[int],
    grad_threshold: float,
    naive_lod_stage_iterations: int,
    densify_until_iter: int,
    splitter_itr: int | None,
) -> list[str]:
    grad_label = format_float_for_name(grad_threshold)
    scales_label = "x".join(str(scale) for scale in resolution_scales)
    split_label = "control" if split_mode == "control" else f"split{splitter_itr}"
    run_name = (
        f"tartan_cluster{args.cluster_count}_{split_label}_{variant}"
        f"_r{scales_label}"
        f"_g{grad_label}"
        f"_lod{naive_lod_stage_iterations}"
        f"_dens{densify_until_iter}"
    )
    model_path = str(Path(args.output_root) / run_name)
    xtend = 0 if split_mode == "control" else max(args.cluster_count - 1, 0)

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
        str(densify_until_iter),
        "--densify_grad_threshold",
        f"{grad_threshold:.12g}",
        "--disable_viewer",
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
        str(naive_lod_stage_iterations),
        "--edgs_num_refs",
        str(args.edgs_num_refs),
    ]
    if split_mode == "control":
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


def build_jobs(args: argparse.Namespace) -> list[list[str]]:
    grad_thresholds = build_grad_thresholds(args.grad_start, args.grad_end, args.grad_steps)
    schedules = parse_schedule(args.schedules)
    validate_schedules(schedules, args.iterations, args.post_lod_densify_margin)

    variants = [
        ("baseline", args.baseline_resolution_scales),
        ("lod", args.lod_resolution_scales),
    ]

    jobs: list[list[str]] = []
    split_modes: list[tuple[str, int | None]] = [("control", None)] + [
        ("split", splitter_itr) for splitter_itr in args.splitter_itrs
    ]
    for grad_threshold, schedule in itertools.product(grad_thresholds, schedules):
        naive_lod_stage_iterations, densify_until_iter = schedule
        for split_mode, splitter_itr in split_modes:
            for variant, resolution_scales in variants:
                jobs.append(
                    build_command(
                        args=args,
                        split_mode=split_mode,
                        variant=variant,
                        resolution_scales=resolution_scales,
                        grad_threshold=grad_threshold,
                        naive_lod_stage_iterations=naive_lod_stage_iterations,
                        densify_until_iter=densify_until_iter,
                        splitter_itr=splitter_itr,
                    )
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired clustered baseline and LoD TartanAir sweeps over densify grad "
            "threshold, naive LoD stage iterations, and densify-until iteration."
        )
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", default="train_nomask.py")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument(
        "--splitter-itrs",
        type=parse_optional_int_list,
        default=DEFAULT_SPLITTER_ITRS,
        help=(
            "Comma-separated split extension iterations for clustered runs. "
            "A non-split control is always included separately."
        ),
    )
    parser.add_argument("--viewpoint-splitter", default=DEFAULT_VIEWPOINT_SPLITTER)
    parser.add_argument("--viewpoint-splitter-config", default=DEFAULT_VIEWPOINT_SPLITTER_CONFIG)
    parser.add_argument("--grad-start", type=float, default=1e-3)
    parser.add_argument("--grad-end", type=float, default=1e-4)
    parser.add_argument("--grad-steps", type=int, default=4)
    parser.add_argument(
        "--schedules",
        default=",".join(f"{naive}:{densify}" for naive, densify in DEFAULT_SCHEDULES),
        help=(
            "Comma-separated NAIVE_LOD_STAGE_ITERATIONS:DENSIFY_UNTIL_ITER pairs. "
            "Each pair must satisfy naive_lod_stage_iterations + post_lod_densify_margin < densify_until."
        ),
    )
    parser.add_argument(
        "--post-lod-densify-margin",
        type=int,
        default=DEFAULT_POST_LOD_DENSIFY_MARGIN,
        help=(
            "Minimum extra densification iterations required after the last coarse LoD stage. "
            "Schedules must satisfy naive_lod_stage_iterations + margin < densify_until_iter."
        ),
    )
    parser.add_argument("--baseline-resolution-scales", type=parse_int_list, default=[2])
    parser.add_argument("--lod-resolution-scales", type=parse_int_list, default=[2, 4, 8])
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--edgs_matches_per_ref", type=int, default=200)
    parser.add_argument("--edgs_num_refs", type=int, default=500)
    parser.add_argument("--edgs_proj_err_tolerance", type=float, default=0.01)
    parser.add_argument("--edgs_roma_model", default="outdoors")
    parser.add_argument("--edgs_skip_frames", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue launching later jobs after a failed training process.",
    )
    args = parser.parse_args()

    jobs = build_jobs(args)
    print(f"Prepared {len(jobs)} jobs.")
    print(f"Cluster count: {args.cluster_count} (xtend={max(args.cluster_count - 1, 0)})")
    print("Non-split control: enabled")
    print(f"Clustered split iterations: {args.splitter_itrs}")
    print(f"Viewpoint splitter: {args.viewpoint_splitter}")
    print(f"Viewpoint splitter config: {args.viewpoint_splitter_config}")
    print(f"Baseline resolution scales: {args.baseline_resolution_scales}")
    print(f"LoD resolution scales: {args.lod_resolution_scales}")
    print(f"Post-LoD densify margin: {args.post_lod_densify_margin}")
    print(f"W&B project/group: {args.wandb_project}/{args.wandb_group}")

    for index, command in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] {' '.join(command)}")
        if args.dry_run:
            continue
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            if not args.continue_on_error:
                raise
            print(f"[WARN] job {index} failed; continuing because --continue-on-error was set")


if __name__ == "__main__":
    main()
