#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import read_model, qvec2rotmat


def _quat_xyzw_to_rotmat(q_xyzw):
    x, y, z, w = [float(value) for value in q_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _camera_center_from_colmap(image):
    rotation = qvec2rotmat(image.qvec)
    translation = np.asarray(image.tvec, dtype=np.float64)
    return -rotation.T @ translation


def _load_colmap_trajectory(folder):
    _, images, _ = read_model(str(folder))
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if not ordered_images:
        raise ValueError(f"No registered COLMAP images found in {folder}")

    points = np.stack([_camera_center_from_colmap(image) for image in ordered_images], axis=0)
    labels = [image.name for image in ordered_images]
    return points, labels, None, "colmap"


def _camera_center_from_packet(packet, camera_id):
    body_to_world = packet["body_to_world"]
    camera_models = packet["camera_models"]
    if camera_id >= len(camera_models):
        raise IndexError(
            f"Packet {packet.get('packet_index', '?')} has no camera_id={camera_id}; "
            f"available ids: 0..{len(camera_models) - 1}"
        )

    camera_to_body = camera_models[camera_id]["camera_to_body"]
    R_ItoG = _quat_xyzw_to_rotmat(body_to_world["q_xyzw"])
    p_IinG = np.asarray(body_to_world["p_xyz"], dtype=np.float64)
    R_CtoI = _quat_xyzw_to_rotmat(camera_to_body["q_xyzw"])
    p_CinI = np.asarray(camera_to_body["p_xyz"], dtype=np.float64)
    return R_ItoG @ p_CinI + p_IinG, float(packet["timestamp_sec"]), packet.get("frame_id", "")


def _load_packet_trajectory(folder, camera_id):
    packets_path = folder / "packets.jsonl"
    if not packets_path.exists():
        raise FileNotFoundError(f"Could not find {packets_path}")

    entries = []
    with packets_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                packet = json.loads(stripped)
            except json.JSONDecodeError:
                if line_number == 1:
                    raise
                continue
            center, timestamp_sec, frame_id = _camera_center_from_packet(packet, camera_id)
            entries.append((timestamp_sec, center, frame_id or f"packet_{len(entries):06d}"))

    if not entries:
        raise ValueError(f"No valid packets found in {packets_path}")

    entries.sort(key=lambda item: item[0])
    points = np.stack([item[1] for item in entries], axis=0)
    labels = [item[2] for item in entries]
    timestamps = np.asarray([item[0] for item in entries], dtype=np.float64)
    return points, labels, timestamps, "packets"


def _load_gt_csv(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No GT rows found in {path}")

    timestamps = np.asarray([float(row[0]) * 1e-9 for row in rows], dtype=np.float64)
    points = np.asarray([[float(row[1]), float(row[2]), float(row[3])] for row in rows], dtype=np.float64)
    return timestamps, points


def _resample_gt_points(gt_timestamps, gt_points, sample_timestamps):
    if len(gt_timestamps) == 0 or len(gt_points) == 0:
        raise ValueError("Ground-truth trajectory is empty.")
    if len(gt_timestamps) == 1:
        return np.repeat(gt_points[:1], len(sample_timestamps), axis=0)

    sample_timestamps = np.asarray(sample_timestamps, dtype=np.float64)
    x = np.interp(sample_timestamps, gt_timestamps, gt_points[:, 0])
    y = np.interp(sample_timestamps, gt_timestamps, gt_points[:, 1])
    z = np.interp(sample_timestamps, gt_timestamps, gt_points[:, 2])
    return np.stack([x, y, z], axis=1)


def _detect_input_type(folder):
    if (folder / "packets.jsonl").exists():
        return "packets"
    for extension in (".bin", ".txt"):
        if (folder / f"images{extension}").exists():
            return "colmap"
    return None


def _apply_equal_aspect(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = np.max(maxs - mins)
    radius = max(extent / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _build_segments(points):
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.stack([points[:-1], points[1:]], axis=1)


def _plot_trajectory(points, title, cmap_name, point_size, gt_points=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    progress = np.linspace(0.0, 1.0, len(points))
    cmap = plt.get_cmap(cmap_name)

    segments = _build_segments(points)
    if len(segments) > 0:
        line_collection = Line3DCollection(segments, cmap=cmap, linewidths=2.5)
        line_collection.set_array(progress[:-1])
        ax.add_collection(line_collection)

    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=progress,
        cmap=cmap,
        s=point_size,
        depthshade=False,
    )
    fig.colorbar(scatter, ax=ax, pad=0.08, label="Start → End")

    ax.scatter(points[0, 0], points[0, 1], points[0, 2], c="limegreen", s=80, label="Start")
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="crimson", s=80, label="End")

    if gt_points is not None and len(gt_points) > 0:
        ax.plot(
            gt_points[:, 0],
            gt_points[:, 1],
            gt_points[:, 2],
            color="black",
            linewidth=1.6,
            alpha=0.8,
            label="Ground Truth",
        )
        ax.scatter(
            gt_points[0, 0],
            gt_points[0, 1],
            gt_points[0, 2],
            c="black",
            s=28,
            marker="x",
        )

    all_points = points if gt_points is None else np.concatenate([points, gt_points], axis=0)
    _apply_equal_aspect(ax, all_points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=28, azim=-62)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot a COLMAP-style 3D camera trajectory from either a packet export root or a COLMAP sparse model folder."
    )
    parser.add_argument("folder", type=Path, help="Packet export root or COLMAP sparse model folder.")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index for packet exports.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth pose.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap for the start→end gradient.")
    parser.add_argument("--point-size", type=float, default=18.0, help="Scatter marker size.")
    parser.add_argument("--gt-csv", type=Path, help="Optional ground-truth CSV with #time(ns),px,py,pz,...")
    parser.add_argument("--output", type=Path, help="Write the plot to this file instead of only showing it.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    args = parser.parse_args()

    folder = args.folder.resolve()
    input_type = _detect_input_type(folder)
    if input_type is None:
        raise SystemExit(
            f"Unsupported input folder: {folder}\n"
            "Expected either packets.jsonl or a COLMAP sparse model with images.bin/images.txt."
        )

    if input_type == "packets":
        points, labels, timestamps, source_label = _load_packet_trajectory(folder, args.camera_id)
    else:
        points, labels, timestamps, source_label = _load_colmap_trajectory(folder)

    stride = max(1, int(args.stride))
    points = points[::stride]
    labels = labels[::stride]
    if timestamps is not None:
        timestamps = timestamps[::stride]

    if len(points) == 0:
        raise SystemExit("No trajectory points remain after applying --stride.")

    gt_points = None
    if args.gt_csv is not None:
        gt_timestamps, gt_all_points = _load_gt_csv(args.gt_csv.resolve())
        if timestamps is not None:
            gt_points = _resample_gt_points(gt_timestamps, gt_all_points, timestamps)
        else:
            gt_points = gt_all_points[::stride]

    title = f"Camera Trajectory ({source_label})\n{folder.name} [{len(points)} poses]"
    fig = _plot_trajectory(
        points,
        title=title,
        cmap_name=args.cmap,
        point_size=args.point_size,
        gt_points=gt_points,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=220, bbox_inches="tight")
        print(f"Saved trajectory plot to {args.output}")

    start_label = labels[0] if labels else "n/a"
    end_label = labels[-1] if labels else "n/a"
    print(f"Loaded {len(points)} poses from {source_label} input: {folder}")
    print(f"Start label: {start_label}")
    print(f"End label:   {end_label}")
    if gt_points is not None:
        print(f"Loaded {len(gt_points)} GT poses from {args.gt_csv.resolve()}")

    if args.show or args.output is None:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
