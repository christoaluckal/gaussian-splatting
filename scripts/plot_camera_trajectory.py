#!/usr/bin/env python3
import argparse
import csv
import json
import math
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


def _camera_forward_from_colmap(image):
    rotation = qvec2rotmat(image.qvec)
    camera_to_world = rotation.T
    forward = camera_to_world[:, 2]
    forward_norm = np.linalg.norm(forward)
    if forward_norm > 1e-8:
        forward = forward / forward_norm
    return forward.astype(np.float64)


def _load_colmap_trajectory(folder):
    _, images, _ = read_model(str(folder))
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if not ordered_images:
        raise ValueError(f"No registered COLMAP images found in {folder}")

    points = np.stack([_camera_center_from_colmap(image) for image in ordered_images], axis=0)
    forwards = np.stack([_camera_forward_from_colmap(image) for image in ordered_images], axis=0)
    labels = [image.name for image in ordered_images]
    return points, forwards, labels, None, "colmap"


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
    center = R_ItoG @ p_CinI + p_IinG
    camera_to_world = R_ItoG @ R_CtoI
    forward = camera_to_world[:, 2]
    forward_norm = np.linalg.norm(forward)
    if forward_norm > 1e-8:
        forward = forward / forward_norm
    return center, forward.astype(np.float64), float(packet["timestamp_sec"]), packet.get("frame_id", "")


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
            center, forward, timestamp_sec, frame_id = _camera_center_from_packet(packet, camera_id)
            entries.append((timestamp_sec, center, forward, frame_id or f"packet_{len(entries):06d}"))

    if not entries:
        raise ValueError(f"No valid packets found in {packets_path}")

    entries.sort(key=lambda item: item[0])
    points = np.stack([item[1] for item in entries], axis=0)
    forwards = np.stack([item[2] for item in entries], axis=0)
    labels = [item[3] for item in entries]
    timestamps = np.asarray([item[0] for item in entries], dtype=np.float64)
    return points, forwards, labels, timestamps, "packets"


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


def _plot_progress_trajectory(ax, points, title, cmap_name, point_size, gt_points=None):
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
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=28, azim=-62)
    return scatter


def _build_cluster_features(points, forwards, position_scale, forward_scale):
    return np.concatenate(
        [
            points * position_scale,
            forwards * forward_scale,
        ],
        axis=1,
    )


def _initialize_centroids(features, num_partitions):
    centroids = [features[0]]
    min_dist2 = np.full((features.shape[0],), np.inf, dtype=np.float64)
    for _ in range(1, num_partitions):
        latest = centroids[-1]
        dist2 = np.sum((features - latest[None, :]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        next_idx = int(np.argmax(min_dist2))
        centroids.append(features[next_idx])
    return np.stack(centroids, axis=0)


def _repair_empty_clusters(assignments, distances, num_partitions):
    counts = np.bincount(assignments, minlength=num_partitions)
    for cluster_idx in range(num_partitions):
        if counts[cluster_idx] > 0:
            continue
        donor_idx = int(np.argmax(counts))
        donor_members = np.where(assignments == donor_idx)[0]
        if donor_members.size <= 1:
            continue
        donor_distances = distances[donor_members, donor_idx]
        moved_member = int(donor_members[np.argmax(donor_distances)])
        assignments[moved_member] = cluster_idx
        counts[donor_idx] -= 1
        counts[cluster_idx] += 1
    return assignments


def _kmeans(features, num_partitions, max_iterations):
    centroids = _initialize_centroids(features, num_partitions)
    assignments = np.zeros((features.shape[0],), dtype=np.int64)
    for _ in range(max_iterations):
        distances = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        next_assignments = np.argmin(distances, axis=1)
        next_assignments = _repair_empty_clusters(next_assignments, distances, num_partitions)

        next_centroids = centroids.copy()
        for cluster_idx in range(num_partitions):
            members = features[next_assignments == cluster_idx]
            if members.size == 0:
                continue
            next_centroids[cluster_idx] = members.mean(axis=0)

        if np.array_equal(assignments, next_assignments) and np.allclose(centroids, next_centroids):
            assignments = next_assignments
            break

        assignments = next_assignments
        centroids = next_centroids

    return assignments


def _compute_cluster_assignments(points, forwards, cluster_count, position_scale, forward_scale, max_iterations):
    effective_cluster_count = min(max(1, int(cluster_count)), len(points))
    if effective_cluster_count <= 1:
        return np.zeros((len(points),), dtype=np.int64), 1
    features = _build_cluster_features(points, forwards, position_scale, forward_scale)
    assignments = _kmeans(features, effective_cluster_count, max_iterations=max_iterations)
    return assignments, effective_cluster_count


def _plot_cluster_trajectory(ax, points, assignments, cluster_count, point_size, gt_points=None):
    cmap = plt.get_cmap("tab10" if cluster_count <= 10 else "tab20")
    colors = np.asarray([cmap(idx % cmap.N) for idx in assignments])

    segments = _build_segments(points)
    if len(segments) > 0:
        line_colors = colors[:-1]
        line_collection = Line3DCollection(segments, colors=line_colors, linewidths=2.5)
        ax.add_collection(line_collection)

    for cluster_idx in range(cluster_count):
        cluster_mask = assignments == cluster_idx
        if not np.any(cluster_mask):
            continue
        ax.scatter(
            points[cluster_mask, 0],
            points[cluster_mask, 1],
            points[cluster_mask, 2],
            color=colors[cluster_mask][0],
            s=point_size,
            depthshade=False,
            label=f"Cluster {cluster_idx}",
        )

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
    ax.set_title(f"Pose Clusters [{cluster_count}]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=28, azim=-62)


def _plot_trajectory(points, forwards, title, cmap_name, point_size, gt_points=None, cluster_count=3, position_scale=1.0, forward_scale=0.5, max_cluster_iterations=32):
    fig = plt.figure(figsize=(18, 8))
    left_ax = fig.add_subplot(121, projection="3d")
    right_ax = fig.add_subplot(122, projection="3d")

    scatter = _plot_progress_trajectory(left_ax, points, title, cmap_name, point_size, gt_points=gt_points)
    fig.colorbar(scatter, ax=left_ax, pad=0.08, label="Start → End")

    assignments, effective_cluster_count = _compute_cluster_assignments(
        points,
        forwards,
        cluster_count,
        position_scale=position_scale,
        forward_scale=forward_scale,
        max_iterations=max_cluster_iterations,
    )
    _plot_cluster_trajectory(
        right_ax,
        points,
        assignments,
        effective_cluster_count,
        point_size,
        gt_points=gt_points,
    )
    left_ax.set_title(title)
    fig.tight_layout()
    return fig, assignments, effective_cluster_count


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
    parser.add_argument("--cluster-count", type=int, default=3, help="Number of pose clusters for the right-hand subplot.")
    parser.add_argument("--cluster-position-scale", type=float, default=1.0, help="Position feature scale for pose clustering.")
    parser.add_argument("--cluster-forward-scale", type=float, default=0.5, help="Forward-direction feature scale for pose clustering.")
    parser.add_argument("--cluster-max-iterations", type=int, default=32, help="Maximum iterations for pose clustering.")
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
        points, forwards, labels, timestamps, source_label = _load_packet_trajectory(folder, args.camera_id)
    else:
        points, forwards, labels, timestamps, source_label = _load_colmap_trajectory(folder)

    stride = max(1, int(args.stride))
    points = points[::stride]
    forwards = forwards[::stride]
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
    fig, assignments, effective_cluster_count = _plot_trajectory(
        points,
        forwards,
        title=title,
        cmap_name=args.cmap,
        point_size=args.point_size,
        gt_points=gt_points,
        cluster_count=args.cluster_count,
        position_scale=args.cluster_position_scale,
        forward_scale=args.cluster_forward_scale,
        max_cluster_iterations=max(1, args.cluster_max_iterations),
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
    cluster_sizes = np.bincount(assignments, minlength=effective_cluster_count).tolist()
    print(f"Computed {effective_cluster_count} pose clusters with sizes: {cluster_sizes}")
    if gt_points is not None:
        print(f"Loaded {len(gt_points)} GT poses from {args.gt_csv.resolve()}")

    if args.show or args.output is None:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
