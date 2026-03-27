import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from common import ColmapSplitterBase


class Splitter(ColmapSplitterBase):
    def split_points_radial(self, proj_xy, image_names, num_splits=4):
        mean = np.mean(proj_xy, axis=0)
        centered = proj_xy - mean

        angles = np.arctan2(centered[:, 1], centered[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        bin_edges = np.linspace(0, 2 * np.pi, num_splits + 1)

        image_groups = {}
        bin_indices = np.digitize(angles, bin_edges, right=False)
        bin_indices[bin_indices > num_splits] = num_splits

        for name, group in zip(image_names, bin_indices):
            image_groups[name] = int(group - 1)

        return image_groups, bin_edges, mean

    def _camera_centers_from_images(self, images):
        xyz = []
        image_names = []
        for image in images:
            qw, qx, qy, qz = image["quat"]
            tx, ty, tz = image["tvec"]
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_vec = np.array([tx, ty, tz], dtype=float)
            xyz.append(-rotation.T @ t_vec)
            image_names.append(image["name"])
        return np.array(xyz), image_names

    def build_model(self, split_num=2, num_test=0):
        images = self._read_images()
        all_points = self._read_points3D()

        if not images:
            raise ValueError("No images found in sparse_txt/images.txt")

        xyz, image_names = self._camera_centers_from_images(images)
        mean = np.mean(xyz, axis=0)
        xyz_centered = xyz - mean

        cov = np.cov(xyz_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1

        rotation_align = eigvecs.T
        xyz_aligned = (rotation_align @ xyz_centered.T).T
        proj_xy = xyz_aligned[:, :2]

        image_groups, _, _ = self.split_points_radial(proj_xy, image_names, num_splits=split_num)

        group_images, group_image_ids = self._build_group_images(image_groups, split_num)
        group_points = self._assign_unique_points(group_images, all_points, split_num)
        group_images, group_points = self._prune_to_group_consistency(
            group_images, group_points, group_image_ids
        )

        for group_idx in range(split_num):
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
    parser.add_argument("--split-num", type=int, default=1, help="number of radial output splits")
    parser.add_argument("--num-test", type=int, default=0, help="number of held-out test images per model")
    args = parser.parse_args()

    splitter = Splitter(
        scene_path=os.path.abspath(args.source),
        new_scene_path=os.path.abspath(args.output),
    )
    splitter.build_model(split_num=args.split_num, num_test=args.num_test)
