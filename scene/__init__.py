#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import random
import time
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from arguments import ModelParams
from edgs_init import apply_edgs_initialization
from scene.dataset_readers import getNerfppNorm, sceneLoadTypeCallbacks
from scene.gaussian_model import BasicPointCloud, GaussianModel
from scene.viewpoint_splitters import load_viewpoint_splitter
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View2
from utils.system_utils import searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        edgs_init_cfg=None,
        training_args=None,
        device="cuda",
    ):
        self.model_path = args.model_path
        path = Path(args.source_path)
        self.source_path = path
        self.model_paths = path.parent.absolute()
        self.loaded_iter = None
        self.gaussians = gaussians
        self.xtend = args.xtend
        self.edgs_init_cfg = edgs_init_cfg
        self.training_args = training_args
        self.device = device
        self.viewpoint_splitter = getattr(args, "viewpoint_splitter", "")
        self.viewpoint_splitter_config = self._parse_viewpoint_splitter_config(
            getattr(args, "viewpoint_splitter_config", "{}")
        )
        self.edgs_cache_root = getattr(args, "edgs_cache_root", "")
        self.runtime_stats = {
            "edgs_base_init_time_sec": 0.0,
            "edgs_base_init_gpu_memory_mb": 0.0,
            "edgs_base_init_peak_gpu_memory_mb": 0.0,
            "edgs_extensions_init_time_sec": 0.0,
            "edgs_extensions_init_gpu_memory_mb": 0.0,
            "edgs_extensions_init_peak_gpu_memory_mb": 0.0,
            "edgs_extensions_init_count": 0,
        }

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}
        self.extension_set = []
        self.x_gauss = []
        self.current_xidx = 1

        scene_info = self._load_scene_info(args)

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"),
                "wb",
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for cam_idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(cam_idx, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        scene_info, extension_scene_infos = self._build_extension_scene_infos(scene_info, args)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
            for extension_scene_info in extension_scene_infos:
                random.shuffle(extension_scene_info["train_cameras"])
                random.shuffle(extension_scene_info["test_cameras"])

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        reference_resolution_scale = resolution_scales[0]
        self._reference_resolution_scale = reference_resolution_scale
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras,
                resolution_scale,
                args,
                scene_info.is_nerf_synthetic,
                False,
                reference_resolution_scale,
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras,
                resolution_scale,
                args,
                scene_info.is_nerf_synthetic,
                True,
                reference_resolution_scale,
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                ),
                args.train_test_exp,
            )
        else:
            base_cache_path = self._edgs_cache_file("base", 0)
            if (
                self._should_apply_edgs_init_to_base()
                and base_cache_path is not None
                and self._load_cached_gaussian(
                    self.gaussians,
                    base_cache_path,
                    args.train_test_exp,
                    scene_info.train_cameras,
                )
            ):
                print(f"Loaded cached EDGS base Gaussian from {base_cache_path}")
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
                if self._should_apply_edgs_init_to_base():
                    self.gaussians.training_setup(self.training_args)
                    self._apply_timed_edgs_initialization(
                        self.gaussians,
                        self.train_cameras[reference_resolution_scale],
                        phase="base",
                    )
                    self._save_cached_gaussian(self.gaussians, base_cache_path)

        for extension_idx, extension_scene_info in enumerate(extension_scene_infos, start=1):
            extension_set, extension_gaussian = self._create_extension_set(
                extension_idx,
                extension_scene_info,
                resolution_scales,
                args,
            )
            self.extension_set.append(extension_set)
            self.x_gauss.append(extension_gaussian)

    def _load_scene_info(self, args):
        if os.path.exists(os.path.join(args.source_path, "packets.jsonl")):
            print("Found packets.jsonl, assuming Phase 2 OpenVINS packet export!")
            return sceneLoadTypeCallbacks["OpenVINSPackets"](
                args.source_path,
                args.images,
                args.depths,
                args.eval,
                args.train_test_exp,
                packet_stride=args.packet_stride,
                packet_offset=args.packet_offset,
                packet_flip_lr=args.packet_flip_lr,
                packet_flip_ud=args.packet_flip_ud,
            )
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            return sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.depths,
                args.eval,
                args.train_test_exp,
            )
        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            return sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.depths,
                args.eval,
            )
        raise AssertionError("Could not recognize scene type!")

    def _parse_viewpoint_splitter_config(self, raw_config):
        if raw_config in (None, ""):
            return {}
        if isinstance(raw_config, dict):
            return raw_config
        try:
            parsed = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "viewpoint_splitter_config must be valid JSON."
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("viewpoint_splitter_config must decode to a JSON object.")
        return parsed

    def _build_edgs_cache_key(self):
        if not self.edgs_cache_root:
            return None
        payload = {
            "source_path": self.source_path.as_posix(),
            "xtend": self.xtend,
            "viewpoint_splitter": self.viewpoint_splitter,
            "viewpoint_splitter_config": self.viewpoint_splitter_config,
            "reference_resolution_scale": getattr(self, "_reference_resolution_scale", None),
            "edgs_cfg": vars(self.edgs_init_cfg) if self.edgs_init_cfg is not None else None,
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return digest

    def _edgs_cache_dir(self):
        cache_key = self._build_edgs_cache_key()
        if cache_key is None:
            return None
        return Path(self.edgs_cache_root) / cache_key

    def _edgs_cache_file(self, phase, index):
        cache_dir = self._edgs_cache_dir()
        if cache_dir is None:
            return None
        if phase == "base":
            return cache_dir / "base_gaussian.ply"
        if phase == "extension":
            return cache_dir / f"extension_{index}_gaussian.ply"
        raise ValueError(f"Unsupported EDGS cache phase: {phase}")

    def _load_cached_gaussian(self, gaussians, cache_path, use_train_test_exp, cam_infos):
        if cache_path is None or not cache_path.exists():
            return False
        gaussians.load_ply(str(cache_path), use_train_test_exp)
        gaussians.rebuild_exposure_from_cam_infos(cam_infos)
        return True

    def _save_cached_gaussian(self, gaussians, cache_path):
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        gaussians.save_ply(str(cache_path))

    def _legacy_extension_exists(self, index):
        return os.path.exists(os.path.join(self.model_paths, f"model{index}", "sparse"))

    def _build_extension_scene_infos(self, scene_info, args):
        if self.viewpoint_splitter:
            return self._build_dynamic_extension_scene_infos(scene_info)

        extension_scene_infos = []
        for extension_idx in range(1, self.xtend + 1):
            if not self._legacy_extension_exists(extension_idx):
                print(
                    f"Legacy extension folder model{extension_idx} not found; "
                    "skipping remaining legacy extension blocks."
                )
                break
            extension_scene_infos.append(
                self._load_legacy_extension_scene_info(extension_idx)
            )
        return scene_info, extension_scene_infos

    def _build_dynamic_extension_scene_infos(self, scene_info):
        total_partitions = self.xtend + 1
        if total_partitions <= 1:
            return scene_info, []

        splitter = load_viewpoint_splitter(self.viewpoint_splitter)
        partitions = splitter(
            scene_info.train_cameras,
            total_partitions,
            self.viewpoint_splitter_config,
        )
        if not partitions:
            return scene_info, []

        partition_cameras = [
            [scene_info.train_cameras[idx] for idx in partition.camera_indices]
            for partition in partitions
            if partition.camera_indices
        ]
        if not partition_cameras:
            return scene_info, []

        cluster_point_clouds = self._split_point_cloud_by_view_clusters(
            scene_info.point_cloud,
            partition_cameras,
            scene_info.nerf_normalization["radius"],
        )

        base_train_cameras = partition_cameras[0]
        base_scene_info = scene_info._replace(
            train_cameras=base_train_cameras,
            point_cloud=cluster_point_clouds[0],
            nerf_normalization=getNerfppNorm(base_train_cameras),
        )

        extension_scene_infos = []
        for partition_idx in range(1, len(partition_cameras)):
            extension_train_cameras = partition_cameras[partition_idx]
            extension_scene_infos.append(
                {
                    "name": f"dynamic_partition_{partition_idx}",
                    "train_cameras": extension_train_cameras,
                    "test_cameras": [],
                    "point_cloud": cluster_point_clouds[partition_idx],
                    "nerf_normalization": getNerfppNorm(extension_train_cameras),
                    "is_nerf_synthetic": scene_info.is_nerf_synthetic,
                }
            )

        print(
            "Dynamic viewpoint splitter "
            f"'{self.viewpoint_splitter}' created {1 + len(extension_scene_infos)} partitions "
            f"from {len(partitions)} requested groups."
        )
        return base_scene_info, extension_scene_infos

    def _camera_center_and_forward(self, cam_info):
        world_to_camera = getWorld2View2(cam_info.R, cam_info.T)
        camera_to_world = np.linalg.inv(world_to_camera)
        center = camera_to_world[:3, 3]
        forward = camera_to_world[:3, 2]
        norm = np.linalg.norm(forward)
        if norm > 1e-8:
            forward = forward / norm
        return center.astype(np.float32), forward.astype(np.float32)

    def _split_point_cloud_by_view_clusters(self, point_cloud, camera_clusters, scene_radius):
        if len(camera_clusters) == 1:
            return [point_cloud]

        points = np.asarray(point_cloud.points, dtype=np.float32)
        colors = np.asarray(point_cloud.colors, dtype=np.float32)
        normals = np.asarray(point_cloud.normals, dtype=np.float32)
        centers = []
        for cameras in camera_clusters:
            cluster_centers = np.stack([self._camera_center_and_forward(cam)[0] for cam in cameras], axis=0)
            centers.append(cluster_centers.mean(axis=0))
        centers = np.stack(centers, axis=0)

        if points.size == 0:
            return [
                self._build_fallback_point_cloud(cameras, scene_radius)
                for cameras in camera_clusters
            ]

        point_assignments = np.argmin(
            np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2),
            axis=1,
        )

        cluster_point_clouds = []
        for cluster_idx, cameras in enumerate(camera_clusters):
            cluster_mask = point_assignments == cluster_idx
            if np.any(cluster_mask):
                cluster_points = points[cluster_mask]
                cluster_colors = colors[cluster_mask]
                cluster_normals = normals[cluster_mask]
                cluster_point_clouds.append(
                    BasicPointCloud(
                        points=cluster_points,
                        colors=cluster_colors,
                        normals=cluster_normals,
                    )
                )
            else:
                cluster_point_clouds.append(
                    self._build_fallback_point_cloud(cameras, scene_radius)
                )
        return cluster_point_clouds

    def _build_fallback_point_cloud(self, cameras, scene_radius):
        fallback_points = []
        fallback_colors = []
        fallback_normals = []
        fallback_depth = max(float(scene_radius) * 0.05, 0.25)

        for cam_info in cameras[: min(len(cameras), 128)]:
            center, forward = self._camera_center_and_forward(cam_info)
            fallback_points.append(center + forward * fallback_depth)
            fallback_normals.append(np.zeros(3, dtype=np.float32))

            with Image.open(cam_info.image_path).convert("RGB") as image:
                center_pixel = np.asarray(
                    image.getpixel((cam_info.width // 2, cam_info.height // 2)),
                    dtype=np.float32,
                )
            fallback_colors.append(center_pixel / 255.0)

        if not fallback_points:
            fallback_points = [np.zeros(3, dtype=np.float32)]
            fallback_colors = [np.array([0.5, 0.5, 0.5], dtype=np.float32)]
            fallback_normals = [np.zeros(3, dtype=np.float32)]

        return BasicPointCloud(
            points=np.asarray(fallback_points, dtype=np.float32),
            colors=np.asarray(fallback_colors, dtype=np.float32),
            normals=np.asarray(fallback_normals, dtype=np.float32),
        )

    def _load_legacy_extension_scene_info(self, index):
        model_path = os.path.join(self.model_paths, f"model{index}")
        scene_info = sceneLoadTypeCallbacks["Colmap"](model_path, "images", "", False, False)
        return {
            "name": f"legacy_model{index}",
            "train_cameras": scene_info.train_cameras,
            "test_cameras": scene_info.test_cameras,
            "point_cloud": scene_info.point_cloud,
            "nerf_normalization": scene_info.nerf_normalization,
            "is_nerf_synthetic": scene_info.is_nerf_synthetic,
        }

    def _should_apply_edgs_init_to_base(self):
        return (
            self.training_args is not None
            and self.edgs_init_cfg is not None
            and self.edgs_init_cfg.use
        )

    def _should_apply_edgs_init_to_extensions(self):
        return (
            self.training_args is not None
            and self.edgs_init_cfg is not None
            and self.edgs_init_cfg.use
            and self.edgs_init_cfg.init_extensions
        )

    def _synchronize_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_gpu_memory_mb(self):
        if not torch.cuda.is_available():
            return 0.0

        device = torch.cuda.current_device()
        bytes_per_mb = 1024.0 * 1024.0
        return torch.cuda.memory_reserved(device) / bytes_per_mb

    def _get_gpu_peak_memory_mb(self):
        if not torch.cuda.is_available():
            return 0.0

        device = torch.cuda.current_device()
        bytes_per_mb = 1024.0 * 1024.0
        return torch.cuda.max_memory_reserved(device) / bytes_per_mb

    def _apply_timed_edgs_initialization(self, gaussians, train_cameras, phase):
        self._synchronize_cuda()
        start_time = time.perf_counter()
        applied = apply_edgs_initialization(
            gaussians,
            train_cameras,
            self.edgs_init_cfg,
            device=self.device,
        )
        self._synchronize_cuda()
        elapsed_time_sec = time.perf_counter() - start_time
        gpu_memory_mb = self._get_gpu_memory_mb()
        peak_gpu_memory_mb = self._get_gpu_peak_memory_mb()
        if not applied:
            return False

        if phase == "base":
            self.runtime_stats["edgs_base_init_time_sec"] += elapsed_time_sec
            self.runtime_stats["edgs_base_init_gpu_memory_mb"] = gpu_memory_mb
            self.runtime_stats["edgs_base_init_peak_gpu_memory_mb"] = peak_gpu_memory_mb
        elif phase == "extension":
            self.runtime_stats["edgs_extensions_init_time_sec"] += elapsed_time_sec
            self.runtime_stats["edgs_extensions_init_gpu_memory_mb"] = max(
                self.runtime_stats["edgs_extensions_init_gpu_memory_mb"],
                gpu_memory_mb,
            )
            self.runtime_stats["edgs_extensions_init_peak_gpu_memory_mb"] = max(
                self.runtime_stats["edgs_extensions_init_peak_gpu_memory_mb"],
                peak_gpu_memory_mb,
            )
            self.runtime_stats["edgs_extensions_init_count"] += 1
        else:
            raise ValueError(f"Unsupported EDGS init phase: {phase}")

        return True

    def _create_extension_set(self, index, extension_scene_info, res_scales, args):
        new_train_cameras = {}
        new_test_cameras = {}
        reference_resolution_scale = res_scales[0]

        for resolution_scale in res_scales:
            print("Loading Training Cameras")
            new_train_cameras[resolution_scale] = cameraList_from_camInfos(
                extension_scene_info["train_cameras"],
                resolution_scale,
                args,
                extension_scene_info["is_nerf_synthetic"],
                False,
                reference_resolution_scale,
            )
            print("Loading Test Cameras")
            new_test_cameras[resolution_scale] = cameraList_from_camInfos(
                extension_scene_info["test_cameras"],
                resolution_scale,
                args,
                extension_scene_info["is_nerf_synthetic"],
                True,
                reference_resolution_scale,
            )

        extension_gaussian = GaussianModel(self.gaussians.max_sh_degree, self.gaussians.optimizer_type)
        extension_cache_path = self._edgs_cache_file("extension", index)
        if (
            self._should_apply_edgs_init_to_extensions()
            and extension_cache_path is not None
            and self._load_cached_gaussian(
                extension_gaussian,
                extension_cache_path,
                args.train_test_exp,
                extension_scene_info["train_cameras"],
            )
        ):
            print(f"Loaded cached EDGS extension Gaussian from {extension_cache_path}")
        else:
            extension_gaussian.create_from_pcd(
                extension_scene_info["point_cloud"],
                extension_scene_info["train_cameras"],
                extension_scene_info["nerf_normalization"]["radius"],
            )
            if self._should_apply_edgs_init_to_extensions():
                extension_gaussian.training_setup(self.training_args)
                self._apply_timed_edgs_initialization(
                    extension_gaussian,
                    new_train_cameras[reference_resolution_scale],
                    phase="extension",
                )
                self._save_cached_gaussian(extension_gaussian, extension_cache_path)

        return [new_train_cameras, new_test_cameras], extension_gaussian

    def extend(self):
        print(f"Number of Gaussians before extending: {self.gaussians._xyz.shape}")
        if self.current_xidx <= len(self.extension_set):
            print(f"Extension number {self.current_xidx}")
            for scale in self.train_cameras:
                self.train_cameras[scale] = self.train_cameras[scale] + self.extension_set[self.current_xidx - 1][0][scale]
            for scale in self.test_cameras:
                self.test_cameras[scale] = self.test_cameras[scale] + self.extension_set[self.current_xidx - 1][1][scale]
            self.gaussians.concat_new_gaussian(self.x_gauss[self.current_xidx - 1])
            self.gaussians.extend_exposure_mapping(
                self.extension_set[self.current_xidx - 1][0][min(self.train_cameras.keys())]
            )
            self.current_xidx += 1
            print(f"Number of Gaussians after extending: {self.gaussians._xyz.shape}")
        else:
            print("No extensions available")

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
