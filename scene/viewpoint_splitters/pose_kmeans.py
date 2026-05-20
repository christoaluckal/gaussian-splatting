from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from scene.viewpoint_splitters.base import ViewpointPartition
from utils.graphics_utils import getWorld2View2


def _camera_center_and_forward(cam_info):
    world_to_camera = getWorld2View2(cam_info.R, cam_info.T)
    camera_to_world = np.linalg.inv(world_to_camera)
    center = camera_to_world[:3, 3]
    forward = camera_to_world[:3, 2]
    forward_norm = np.linalg.norm(forward)
    if forward_norm > 1e-8:
        forward = forward / forward_norm
    return center.astype(np.float64), forward.astype(np.float64)


def _build_features(train_cameras, position_scale: float, forward_scale: float):
    features = []
    for cam_info in train_cameras:
        center, forward = _camera_center_and_forward(cam_info)
        features.append(
            np.concatenate(
                [
                    center * position_scale,
                    forward * forward_scale,
                ]
            )
        )
    return np.asarray(features, dtype=np.float64)


def _initialize_centroids(features: np.ndarray, num_partitions: int) -> np.ndarray:
    centroids = [features[0]]
    min_dist2 = np.full((features.shape[0],), np.inf, dtype=np.float64)
    for _ in range(1, num_partitions):
        latest = centroids[-1]
        dist2 = np.sum((features - latest[None, :]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        next_idx = int(np.argmax(min_dist2))
        centroids.append(features[next_idx])
    return np.stack(centroids, axis=0)


def _repair_empty_clusters(assignments: np.ndarray, distances: np.ndarray, num_partitions: int) -> np.ndarray:
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


def _kmeans(features: np.ndarray, num_partitions: int, max_iterations: int) -> np.ndarray:
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


def partition_viewpoints(
    train_cameras: Sequence,
    num_partitions: int,
    config: dict | None = None,
):
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive.")
    if not train_cameras:
        return []

    config = config or {}
    position_scale = float(config.get("position_scale", 1.0))
    forward_scale = float(config.get("forward_scale", 0.5))
    max_iterations = max(int(config.get("max_iterations", 32)), 1)

    effective_partitions = min(num_partitions, len(train_cameras))
    if effective_partitions == 1:
        return [
            ViewpointPartition(
                name="cluster_0",
                camera_indices=list(range(len(train_cameras))),
                metadata={"size": len(train_cameras)},
            )
        ]

    features = _build_features(train_cameras, position_scale=position_scale, forward_scale=forward_scale)
    assignments = _kmeans(features, effective_partitions, max_iterations=max_iterations)

    partitions = []
    for cluster_idx in range(effective_partitions):
        camera_indices = np.where(assignments == cluster_idx)[0].tolist()
        if not camera_indices:
            continue
        centroid = features[camera_indices].mean(axis=0)
        partitions.append(
            ViewpointPartition(
                name=f"cluster_{cluster_idx}",
                camera_indices=camera_indices,
                metadata={
                    "size": len(camera_indices),
                    "feature_centroid": centroid.tolist(),
                },
            )
        )

    partitions.sort(
        key=lambda partition: (
            -len(partition.camera_indices),
            min(partition.camera_indices) if partition.camera_indices else math.inf,
        )
    )
    return partitions
