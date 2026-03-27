import argparse
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from common import ColmapSplitterBase


class Splitter(ColmapSplitterBase):
    def __init__(self, scene_path: str = None, new_scene_path: str = None, is_default: bool = False):
        super().__init__(scene_path=scene_path, new_scene_path=new_scene_path)
        self.is_default = is_default

    def _camera_centers(self, images):
        xyz = []
        for image in images:
            qw, qx, qy, qz = image["quat"]
            tx, ty, tz = image["tvec"]
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_vec = np.array([tx, ty, tz], dtype=float)
            xyz.append(-rotation.T @ t_vec)
        return np.array(xyz)

    def _build_clusters(self, images, dist):
        if self.is_default or len(images) <= 1:
            return [list(range(len(images)))]

        xyz = self._camera_centers(images)
        tree = cKDTree(xyz)
        neighbors = tree.query_ball_tree(tree, r=dist)

        indexed = np.zeros(len(neighbors), dtype=bool)
        anchor_groups = {}
        for idx, entries in enumerate(neighbors):
            if indexed[idx]:
                continue

            cluster = []
            for neighbor_idx in entries:
                if neighbor_idx == idx or indexed[neighbor_idx]:
                    continue
                cluster.append(neighbor_idx)
                indexed[neighbor_idx] = True

            if cluster:
                anchor_groups[idx] = cluster

        groups = []
        if anchor_groups:
            groups.append(sorted(anchor_groups.keys()))
            for cluster in anchor_groups.values():
                groups.append(sorted(cluster))
        else:
            groups.append(list(range(len(images))))

        return groups

    def build_model(self, dist=0.5, num_test=0):
        images = self._read_images()
        all_points = self._read_points3D()

        if not images:
            raise ValueError("No images found in sparse_txt/images.txt")

        groups = self._build_clusters(images, dist)
        image_groups = {}
        for group_idx, image_indices in enumerate(groups):
            for image_idx in image_indices:
                image_groups[images[image_idx]["name"]] = group_idx

        num_groups = len(groups)
        group_images, group_image_ids = self._build_group_images(image_groups, num_groups)
        group_points = self._assign_unique_points(group_images, all_points, num_groups)
        group_images, group_points = self._prune_to_group_consistency(
            group_images, group_points, group_image_ids
        )

        for group_idx in range(num_groups):
            self.write_model(
                f"model{group_idx}",
                group_images[group_idx],
                group_points[group_idx],
                num_test=num_test,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, required=True, help="source scene path")
    parser.add_argument("-m", "--output", type=str, required=True, help="destination root path")
    parser.add_argument("--dist", type=float, default=0.1, help="camera-center neighborhood radius")
    parser.add_argument("--default", action="store_true", help="emit only model0")
    parser.add_argument("--num-test", type=int, default=0, help="number of held-out test images per model")
    args = parser.parse_args()

    splitter = Splitter(
        scene_path=os.path.abspath(args.source),
        new_scene_path=os.path.abspath(args.output),
        is_default=args.default,
    )
    splitter.build_model(dist=args.dist, num_test=args.num_test)
