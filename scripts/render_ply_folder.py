#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state


def _discover_scene_info(args):
    source_path = Path(args.source_path)
    loader_args = (
        str(source_path),
        args.images,
        args.depths,
        args.eval,
        args.train_test_exp,
    )
    if (source_path / "packets.jsonl").exists():
        return sceneLoadTypeCallbacks["OpenVINSPackets"](
            *loader_args,
            packet_stride=args.packet_stride,
            packet_offset=args.packet_offset,
            packet_flip_lr=args.packet_flip_lr,
            packet_flip_ud=args.packet_flip_ud,
        )
    if (source_path / "sparse").exists():
        return sceneLoadTypeCallbacks["Colmap"](*loader_args)
    if (source_path / "transforms_train.json").exists():
        return sceneLoadTypeCallbacks["Blender"](
            str(source_path),
            args.white_background,
            args.depths,
            args.eval,
        )
    raise ValueError(
        "Could not recognize scene source. Expected packets.jsonl, sparse/, or transforms_train.json."
    )


def _build_cameras(args):
    scene_info = _discover_scene_info(args)
    camera_args = SimpleNamespace(
        resolution=args.resolution,
        data_device="cuda",
        train_test_exp=args.train_test_exp,
        match_resolution=args.match_resolution,
        packet_flip_lr=args.packet_flip_lr,
        packet_flip_ud=args.packet_flip_ud,
    )
    cameras = []
    if args.split in {"train", "all"}:
        cameras.extend(
            cameraList_from_camInfos(
                scene_info.train_cameras,
                args.resolution_scale,
                camera_args,
                scene_info.is_nerf_synthetic,
                False,
                args.resolution_scale,
            )
        )
    if args.split in {"test", "all"}:
        cameras.extend(
            cameraList_from_camInfos(
                scene_info.test_cameras,
                args.resolution_scale,
                camera_args,
                scene_info.is_nerf_synthetic,
                True,
                args.resolution_scale,
            )
        )
    if args.view_stride > 1:
        cameras = cameras[:: args.view_stride]
    if args.max_views > 0:
        cameras = cameras[: args.max_views]
    if not cameras:
        raise ValueError(f"No cameras selected for split '{args.split}'.")
    return cameras


def _iter_ply_paths(ply_dir):
    return sorted(Path(ply_dir).glob("*.ply"))


def _make_pipeline_args(args):
    return SimpleNamespace(
        convert_SHs_python=args.convert_SHs_python,
        compute_cov3D_python=args.compute_cov3D_python,
        debug=args.debug,
        antialiasing=args.antialiasing,
    )


def _render_ply(ply_path, cameras, args, pipeline_args):
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(ply_path), use_train_test_exp=args.train_test_exp)

    render_dir = Path(args.output_dir) / ply_path.stem / args.split / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {ply_path.name}")):
        rendering = render(
            camera,
            gaussians,
            pipeline_args,
            background,
            use_trained_exp=args.train_test_exp,
            separate_sh=args.separate_sh,
        )["render"]
        torchvision.utils.save_image(rendering, render_dir / f"{idx:05d}.png")

    del gaussians
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Render every PLY in a folder from cameras loaded from a source scene."
    )
    parser.add_argument("--ply_dir", required=True, help="Folder containing standalone .ply splat files.")
    parser.add_argument("-s", "--source_path", required=True, help="Scene source used only for cameras.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output folder for rendered PNGs.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--images", default="images")
    parser.add_argument("--depths", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resolution", "-r", default=-1, type=int)
    parser.add_argument("--resolution_scale", default=1.0, type=float)
    parser.add_argument("--match_resolution", action="store_true")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--train_test_exp", action="store_true")
    parser.add_argument("--packet_stride", default=1, type=int)
    parser.add_argument("--packet_offset", default=0, type=int)
    parser.add_argument("--packet_flip_lr", action="store_true")
    parser.add_argument("--packet_flip_ud", action="store_true")
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--antialiasing", action="store_true")
    parser.add_argument("--separate_sh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    ply_paths = _iter_ply_paths(args.ply_dir)
    if not ply_paths:
        raise ValueError(f"No .ply files found in {args.ply_dir}")

    cameras = _build_cameras(args)
    pipeline_args = _make_pipeline_args(args)
    print(f"Rendering {len(ply_paths)} PLY files over {len(cameras)} {args.split} cameras.")
    for ply_path in ply_paths:
        _render_ply(ply_path, cameras, args, pipeline_args)


if __name__ == "__main__":
    main()
