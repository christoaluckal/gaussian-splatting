import os
import random
import shutil

import numpy as np


class ColmapSplitterBase:
    def __init__(self, scene_path: str, new_scene_path: str):
        self.scene_base = scene_path
        sparse_dir = "sparse_txt"
        self.cameras = os.path.join(self.scene_base, sparse_dir, "cameras.txt")
        self.images = os.path.join(self.scene_base, sparse_dir, "images.txt")
        self.points3D = os.path.join(self.scene_base, sparse_dir, "points3D.txt")
        self.new_scene_path = os.path.join(new_scene_path)

    def copy_and_remove(self, keys, dst_dir):
        src_dir = os.path.join(self.scene_base, "images")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

        for root, _, files in os.walk(dst_dir):
            for filename in files:
                rel_path = os.path.relpath(os.path.join(root, filename), dst_dir)
                if rel_path not in keys:
                    os.remove(os.path.join(root, filename))

    def _image_header(self, num_images, mean_obs):
        return (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {num_images}, mean observations per image: {mean_obs:.6f}\n"
        )

    def _points_header(self, num_points, mean_track_len):
        return (
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            f"# Number of points: {num_points}, mean track length: {mean_track_len:.6f}\n"
        )

    def _read_images(self):
        images = []
        with open(self.images, "r") as handle:
            lines = handle.readlines()[4:]

        index = 0
        while index < len(lines):
            header = lines[index].strip().split()
            p2d_line = lines[index + 1].strip()

            if p2d_line:
                points2d = np.array(p2d_line.split(), dtype=float).reshape(-1, 3)
            else:
                points2d = np.zeros((0, 3), dtype=float)

            images.append(
                {
                    "image_id": int(header[0]),
                    "header": header,
                    "name": header[9],
                    "camera_id": int(header[8]),
                    "quat": list(map(float, header[1:5])),
                    "tvec": list(map(float, header[5:8])),
                    "points2D": points2d,
                }
            )
            index += 2

        return images

    def _read_points3D(self):
        points = {}
        with open(self.points3D, "r") as handle:
            lines = handle.readlines()[3:]

        for line in lines:
            tokens = line.strip().split()
            if not tokens:
                continue

            point_id = tokens[0]
            track_tokens = tokens[8:]
            track = []
            for idx in range(0, len(track_tokens), 2):
                track.append((int(track_tokens[idx]), int(track_tokens[idx + 1])))

            points[point_id] = {
                "xyzrgberr": tokens[1:8],
                "track": track,
            }

        return points

    def _build_group_images(self, image_groups, num_groups):
        images = self._read_images()
        group_images = {i: {} for i in range(num_groups)}
        group_image_ids = {i: set() for i in range(num_groups)}

        for image in images:
            group_idx = image_groups[image["name"]]
            group_images[group_idx][image["name"]] = {
                "header": list(image["header"]),
                "points2D": image["points2D"].copy(),
                "image_id": image["image_id"],
            }
            group_image_ids[group_idx].add(image["image_id"])

        return group_images, group_image_ids

    def _assign_unique_points(self, group_images, all_points, num_groups):
        group_points = {i: {} for i in range(num_groups)}
        seen = set()

        for group_idx in range(num_groups):
            candidate_ids = set()
            for image in group_images[group_idx].values():
                points2d = image["points2D"]
                if len(points2d) == 0:
                    continue
                point_ids = points2d[:, 2].astype(int)
                point_ids = point_ids[point_ids != -1]
                candidate_ids.update(str(point_id) for point_id in np.unique(point_ids))

            for point_id in candidate_ids:
                if point_id in seen or point_id not in all_points:
                    continue
                group_points[group_idx][point_id] = {
                    "xyzrgberr": list(all_points[point_id]["xyzrgberr"]),
                    "track": list(all_points[point_id]["track"]),
                }
                seen.add(point_id)

        return group_points

    def _prune_to_group_consistency(self, group_images, group_points, group_image_ids):
        for group_idx in group_images.keys():
            valid_point_ids = set(group_points[group_idx].keys())
            valid_image_ids = set(group_image_ids[group_idx])

            for image in group_images[group_idx].values():
                points2d = image["points2D"].copy()
                if len(points2d) > 0:
                    point_ids = points2d[:, 2].astype(int)
                    for idx, point_id in enumerate(point_ids):
                        if point_id != -1 and str(point_id) not in valid_point_ids:
                            points2d[idx, 2] = -1
                image["points2D"] = points2d

            new_group_points = {}
            for point_id, point in group_points[group_idx].items():
                filtered_track = [
                    (image_id, p2d_idx)
                    for image_id, p2d_idx in point["track"]
                    if image_id in valid_image_ids
                ]
                if filtered_track:
                    new_group_points[point_id] = {
                        "xyzrgberr": list(point["xyzrgberr"]),
                        "track": filtered_track,
                    }
            group_points[group_idx] = new_group_points

            valid_point_ids = set(group_points[group_idx].keys())
            for image in group_images[group_idx].values():
                points2d = image["points2D"].copy()
                if len(points2d) > 0:
                    point_ids = points2d[:, 2].astype(int)
                    for idx, point_id in enumerate(point_ids):
                        if point_id != -1 and str(point_id) not in valid_point_ids:
                            points2d[idx, 2] = -1
                image["points2D"] = points2d

        return group_images, group_points

    def write_model(self, name, img_dict, point_dict, num_test=0):
        out_sparse = os.path.join(self.new_scene_path, name, "sparse", "0")
        os.makedirs(out_sparse, exist_ok=True)

        images_path = os.path.join(out_sparse, "images.txt")
        test_path = os.path.join(out_sparse, "test.txt")
        points_path = os.path.join(out_sparse, "points3D.txt")

        items = sorted(img_dict.items(), key=lambda item: item[1]["image_id"])

        if num_test > 0:
            num_test = min(num_test, len(items))
            test_names = set(random.sample([key for key, _ in items], num_test))
            train_items = [(key, value) for key, value in items if key not in test_names]
            test_items = [(key, value) for key, value in items if key in test_names]
        else:
            train_items = items
            test_items = []

        train_image_ids = {value["image_id"] for _, value in train_items}
        pruned_point_dict = {}
        valid_point_ids = set()

        for point_id, point in point_dict.items():
            filtered_track = [
                (image_id, p2d_idx)
                for image_id, p2d_idx in point["track"]
                if image_id in train_image_ids
            ]
            if filtered_track:
                pruned_point_dict[point_id] = {
                    "xyzrgberr": list(point["xyzrgberr"]),
                    "track": filtered_track,
                }
                valid_point_ids.add(point_id)

        pruned_train_items = []
        for key, value in train_items:
            points2d = value["points2D"].copy()
            if len(points2d) > 0:
                for idx in range(len(points2d)):
                    point_id = int(points2d[idx, 2])
                    if point_id != -1 and str(point_id) not in valid_point_ids:
                        points2d[idx, 2] = -1
            new_value = dict(value)
            new_value["points2D"] = points2d
            pruned_train_items.append((key, new_value))

        pruned_test_items = []
        for key, value in test_items:
            points2d = value["points2D"].copy()
            if len(points2d) > 0:
                for idx in range(len(points2d)):
                    points2d[idx, 2] = -1
            new_value = dict(value)
            new_value["points2D"] = points2d
            pruned_test_items.append((key, new_value))

        def count_obs(sub_items):
            obs = 0
            for _, value in sub_items:
                if len(value["points2D"]) == 0:
                    continue
                obs += int(np.sum(value["points2D"][:, 2].astype(int) != -1))
            return obs

        mean_obs = count_obs(pruned_train_items) / max(len(pruned_train_items), 1)
        with open(images_path, "w") as handle:
            handle.write(self._image_header(len(pruned_train_items), mean_obs))
            for _, value in pruned_train_items:
                handle.write(" ".join(str(item) for item in value["header"]) + "\n")
                points2d = value["points2D"]
                if len(points2d) > 0:
                    triplets = [f"{x:.6f} {y:.6f} {int(z)}" for x, y, z in points2d]
                    handle.write(" ".join(triplets))
                handle.write("\n")

        if num_test > 0:
            mean_obs_test = count_obs(pruned_test_items) / max(len(pruned_test_items), 1)
            with open(test_path, "w") as handle:
                handle.write(self._image_header(len(pruned_test_items), mean_obs_test))
                for _, value in pruned_test_items:
                    handle.write(" ".join(str(item) for item in value["header"]) + "\n")
                    points2d = value["points2D"]
                    if len(points2d) > 0:
                        triplets = [f"{x:.6f} {y:.6f} {int(z)}" for x, y, z in points2d]
                        handle.write(" ".join(triplets))
                    handle.write("\n")

        mean_track = (
            np.mean([len(value["track"]) for value in pruned_point_dict.values()])
            if pruned_point_dict
            else 0.0
        )
        with open(points_path, "w") as handle:
            handle.write(self._points_header(len(pruned_point_dict), mean_track))
            for point_id in sorted(pruned_point_dict.keys(), key=lambda value: int(value)):
                point = pruned_point_dict[point_id]
                track_tokens = []
                for image_id, p2d_idx in point["track"]:
                    track_tokens.extend([str(image_id), str(p2d_idx)])
                handle.write(f"{point_id} " + " ".join(point["xyzrgberr"] + track_tokens) + "\n")

        shutil.copy2(self.cameras, os.path.join(out_sparse, "cameras.txt"))

        valid_files = set(img_dict.keys())
        for dirname in ["images", "images_2", "images_4", "images_8"]:
            src_dir = os.path.join(self.scene_base, dirname)
            dst_dir = os.path.join(self.new_scene_path, name, dirname)
            if os.path.exists(src_dir):
                self.copy_and_remove(valid_files, dst_dir)
