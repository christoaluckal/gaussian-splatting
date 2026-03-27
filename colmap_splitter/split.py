import argparse
import os

from common import ColmapSplitterBase


class Splitter(ColmapSplitterBase):
    def build_model(
        self,
        first_name="model0",
        second_name="model1",
        split_frame=None,
        num_test=0,
    ):
        images = self._read_images()
        all_points = self._read_points3D()

        if not images:
            raise ValueError("No images found in sparse_txt/images.txt")

        if split_frame is None:
            split_frame = images[-1]["name"]

        image_groups = {}
        current_group = 0
        found_split_frame = False
        for image in images:
            image_groups[image["name"]] = current_group
            if image["name"] == split_frame:
                current_group = 1
                found_split_frame = True

        if not found_split_frame:
            raise ValueError(f"Split frame '{split_frame}' was not found in {self.images}")

        group_images, group_image_ids = self._build_group_images(image_groups, 2)
        group_points = self._assign_unique_points(group_images, all_points, 2)
        group_images, group_points = self._prune_to_group_consistency(
            group_images, group_points, group_image_ids
        )

        self.write_model(first_name, group_images[0], group_points[0], num_test=num_test)
        self.write_model(second_name, group_images[1], group_points[1], num_test=num_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, required=True, help="source scene path")
    parser.add_argument("-m", "--output", type=str, required=True, help="destination root path")
    parser.add_argument("--first-name", type=str, default="model0", help="first output model name")
    parser.add_argument("--second-name", type=str, default="model1", help="second output model name")
    parser.add_argument(
        "-f",
        "--split-frame",
        type=str,
        default=None,
        help="image name where the second split begins after inclusion in the first split",
    )
    parser.add_argument("--num-test", type=int, default=0, help="number of held-out test images per model")
    args = parser.parse_args()

    splitter = Splitter(
        scene_path=os.path.abspath(args.source),
        new_scene_path=os.path.abspath(args.output),
    )
    splitter.build_model(
        first_name=args.first_name,
        second_name=args.second_name,
        split_frame=args.split_frame,
        num_test=args.num_test,
    )
