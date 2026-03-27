import argparse
import os

from common import ColmapSplitterBase


class Splitter(ColmapSplitterBase):
    def __init__(self, scene_path: str = None, new_scene_path: str = None, is_default: bool = False):
        super().__init__(scene_path=scene_path, new_scene_path=new_scene_path)
        self.is_default = is_default

    def build_model(self, split_num=3, num_test=0):
        images = self._read_images()
        all_points = self._read_points3D()

        if not images:
            raise ValueError("No images found in sparse_txt/images.txt")

        if self.is_default:
            split_num = 1

        split_num = max(int(split_num), 1)
        image_count = len(images)

        split_sizes = [image_count // split_num] * split_num
        for idx in range(image_count % split_num):
            split_sizes[idx] += 1

        image_groups = {}
        cursor = 0
        for group_idx, group_size in enumerate(split_sizes):
            for image in images[cursor : cursor + group_size]:
                image_groups[image["name"]] = group_idx
            cursor += group_size

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
    parser.add_argument("--split-num", type=int, default=1, help="number of contiguous output splits")
    parser.add_argument("--default", action="store_true", help="emit only model0 regardless of split count")
    parser.add_argument("--num-test", type=int, default=0, help="number of held-out test images per model")
    args = parser.parse_args()

    splitter = Splitter(
        scene_path=os.path.abspath(args.source),
        new_scene_path=os.path.abspath(args.output),
        is_default=args.default,
    )
    splitter.build_model(split_num=args.split_num, num_test=args.num_test)
