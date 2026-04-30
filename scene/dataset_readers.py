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

import math
import os
import sys
from collections import defaultdict
from PIL import Image
from typing import NamedTuple
import cv2
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    packet_metadata: dict = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list, packet_metadata=None)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test, packet_metadata=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


def _quat_xyzw_to_rotmat(q_xyzw):
    x, y, z, w = q_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _resolve_phase2_intrinsics(camera_model, width, height):
    intrinsics = camera_model.get("intrinsics", [])
    if len(intrinsics) < 4:
        raise ValueError("Packet camera model is missing fx, fy, cx, cy intrinsics.")

    fx, fy, cx, cy = [float(value) for value in intrinsics[:4]]
    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    distortion_coeffs = camera_model.get("distortion_coeffs", [])
    if camera_model.get("model") == "radtan" and len(distortion_coeffs) >= 4:
        k1, k2, p1, p2 = [float(value) for value in distortion_coeffs[:4]]
        distortion = np.array([k1, k2, p1, p2], dtype=np.float32)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (width, height),
            0.0,
            (width, height),
        )
    else:
        distortion = None
        new_camera_matrix = camera_matrix

    return camera_matrix, distortion, new_camera_matrix


def _flip_phase2_uv(uv, width, height, flip_lr=False, flip_ud=False):
    u, v = float(uv[0]), float(uv[1])
    if flip_lr:
        u = (width - 1) - u
    if flip_ud:
        v = (height - 1) - v
    return np.array([u, v], dtype=np.float64)


def _phase2_uv_to_norm(uv, intrinsics):
    fx, fy, cx, cy = [float(value) for value in intrinsics[:4]]
    return np.array(
        [
            (float(uv[0]) - cx) / fx,
            (float(uv[1]) - cy) / fy,
        ],
        dtype=np.float64,
    )


def _build_phase2_camera_info(packet, image_entry, camera_model, image_index, is_test):
    width = int(image_entry["width"])
    height = int(image_entry["height"])
    _, _, new_camera_matrix = _resolve_phase2_intrinsics(camera_model, width, height)
    fx = float(new_camera_matrix[0, 0])
    fy = float(new_camera_matrix[1, 1])
    cx = float(new_camera_matrix[0, 2])
    cy = float(new_camera_matrix[1, 2])

    body_to_world = packet["body_to_world"]
    camera_to_body = camera_model["camera_to_body"]

    R_ItoG = _quat_xyzw_to_rotmat(body_to_world["q_xyzw"])
    p_IinG = np.asarray(body_to_world["p_xyz"], dtype=np.float64)
    R_CtoI = _quat_xyzw_to_rotmat(camera_to_body["q_xyzw"])
    p_CinI = np.asarray(camera_to_body["p_xyz"], dtype=np.float64)

    R_CtoG = R_ItoG @ R_CtoI
    p_CinG = R_ItoG @ p_CinI + p_IinG
    R_GtoC = R_CtoG.T
    T = -R_GtoC @ p_CinG

    image_name = Path(image_entry["path"]).stem
    image_path = os.path.join(packet["_root_path"], image_entry["path"])

    return CameraInfo(
        uid=image_index,
        R=R_CtoG,
        T=T,
        FovY=focal2fov(fy, height),
        FovX=focal2fov(fx, width),
        depth_params=None,
        image_path=image_path,
        image_name=image_name,
        depth_path="",
        width=width,
        height=height,
        is_test=is_test,
        packet_metadata={
            "packet_index": packet["packet_index"],
            "timestamp_sec": packet["timestamp_sec"],
            "camera_id": image_entry["camera_id"],
            "frame_id": packet["frame_id"],
            "camera_model": camera_model.get("model"),
            "intrinsics": list(camera_model.get("intrinsics", [])),
            "distortion_coeffs": list(camera_model.get("distortion_coeffs", [])),
            "rectified_intrinsics": [fx, fy, cx, cy],
        },
    )


def _sample_phase2_color(image_cache, image_path, uv, width, height):
    image = image_cache.get(image_path)
    if image is None:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        image_cache[image_path] = image

    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return image[y, x].astype(np.float32) / 255.0


def _triangulate_track(observations):
    if len(observations) < 2:
        return None

    best_angle = 0.0
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            dot = float(np.clip(np.dot(observations[i]["ray_world"], observations[j]["ray_world"]), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
            best_angle = max(best_angle, angle)

    if best_angle < 1.0:
        return None

    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for obs in observations:
        direction = obs["ray_world"]
        center = obs["camera_center"]
        proj = identity - np.outer(direction, direction)
        A += proj
        b += proj @ center

    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        point, *_ = np.linalg.lstsq(A, b, rcond=None)

    valid_observations = 0
    total_error = 0.0
    for obs in observations:
        point_cam = obs["R_GtoC"] @ point + obs["T_GtoC"]
        if point_cam[2] <= 1e-4:
            continue
        reproj = point_cam[:2] / point_cam[2]
        total_error += float(np.linalg.norm(reproj - obs["uv_norm"]))
        valid_observations += 1

    if valid_observations < 2:
        return None

    mean_error = total_error / valid_observations
    if mean_error > 0.02:
        return None

    return point


def _build_phase2_seed_point_cloud(train_cam_infos, track_observations, fallback_depth_scale=0.5):
    points = []
    colors = []
    image_cache = {}

    for _, observations in track_observations.items():
        point = _triangulate_track(observations)
        if point is None:
            continue
        first_obs = observations[0]
        color = _sample_phase2_color(
            image_cache,
            first_obs["image_path"],
            first_obs["uv"],
            first_obs["width"],
            first_obs["height"],
        )
        points.append(point)
        colors.append(color)

    if not points:
        fallback_points = []
        fallback_colors = []
        for cam_info in train_cam_infos[: min(len(train_cam_infos), 128)]:
            Rt = np.eye(4, dtype=np.float64)
            Rt[:3, :3] = cam_info.R.T
            Rt[:3, 3] = cam_info.T
            c2w = np.linalg.inv(Rt)
            cam_center = c2w[:3, 3]
            cam_forward = c2w[:3, 2]
            fallback_points.append(cam_center + cam_forward * fallback_depth_scale)
            fallback_colors.append(
                _sample_phase2_color(
                    image_cache,
                    cam_info.image_path,
                    np.array([cam_info.width / 2.0, cam_info.height / 2.0]),
                    cam_info.width,
                    cam_info.height,
                )
            )

        points = fallback_points
        colors = fallback_colors

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    normals = np.zeros_like(points, dtype=np.float32)
    return BasicPointCloud(points=points, colors=colors, normals=normals)


def readOpenVINSPacketSceneInfo(
    path,
    images,
    depths,
    eval,
    train_test_exp,
    llffhold=8,
    packet_stride=1,
    packet_offset=0,
    packet_flip_lr=False,
    packet_flip_ud=False,
):
    packets_path = os.path.join(path, "packets.jsonl")
    if not os.path.exists(packets_path):
        raise FileNotFoundError(f"Expected packet export at '{packets_path}'.")

    packet_stride = max(int(packet_stride or 1), 1)
    packet_offset = max(int(packet_offset or 0), 0)

    packet_entries = []
    skipped_packet_lines = 0
    with open(packets_path, "r") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                skipped_packet_lines += 1
                if skipped_packet_lines <= 5:
                    print(
                        f"[Phase2] Skipping malformed packet line {line_number} in "
                        f"'{packets_path}'."
                    )
                continue
            packet["_root_path"] = path
            packet["_flip_lr"] = bool(packet_flip_lr)
            packet["_flip_ud"] = bool(packet_flip_ud)
            packet_entries.append(packet)

    if not packet_entries:
        raise RuntimeError(f"No packet entries found in '{packets_path}'.")

    if packet_stride > 1 or packet_offset > 0:
        packet_entries = [
            packet
            for packet_index, packet in enumerate(packet_entries)
            if packet_index >= packet_offset and (packet_index - packet_offset) % packet_stride == 0
        ]
        if not packet_entries:
            raise RuntimeError(
                "Packet subsampling removed every packet entry. "
                f"stride={packet_stride}, offset={packet_offset}."
            )
        print(
            f"[Phase2] Packet subsampling active: keeping {len(packet_entries)} packets "
            f"with stride={packet_stride}, offset={packet_offset}."
        )

    cam_infos = []
    train_track_observations = defaultdict(list)
    global_image_index = 0

    for packet in packet_entries:
        camera_models = {model["camera_id"]: model for model in packet["camera_models"]}
        sparse_tracks_by_camera = defaultdict(list)
        for sparse_track in packet.get("sparse_tracks", []):
            sparse_tracks_by_camera[sparse_track["camera_id"]].append(sparse_track)

        for image_entry in packet["images"]:
            cam_id = image_entry["camera_id"]
            camera_model = camera_models[cam_id]
            camera_matrix, distortion, new_camera_matrix = _resolve_phase2_intrinsics(
                camera_model,
                int(image_entry["width"]),
                int(image_entry["height"]),
            )
            is_test = bool(eval and llffhold and global_image_index % llffhold == 0)
            cam_info = _build_phase2_camera_info(packet, image_entry, camera_model, global_image_index, is_test)
            cam_infos.append(cam_info)

            if not is_test or train_test_exp:
                Rt = np.eye(4, dtype=np.float64)
                Rt[:3, :3] = cam_info.R.T
                Rt[:3, 3] = cam_info.T
                c2w = np.linalg.inv(Rt)
                camera_center = c2w[:3, 3]
                R_CtoG = c2w[:3, :3]
                R_GtoC = R_CtoG.T
                T_GtoC = cam_info.T

                for sparse_track in sparse_tracks_by_camera.get(cam_id, []):
                    uv = np.asarray(sparse_track["uv"], dtype=np.float64)
                    uv_norm = np.asarray(sparse_track["uv_norm"], dtype=np.float64)
                    if distortion is not None:
                        undistorted_uv = cv2.undistortPoints(
                            uv.reshape(1, 1, 2).astype(np.float32),
                            camera_matrix,
                            distortion,
                            P=new_camera_matrix,
                        ).reshape(2)
                        undistorted_uv_norm = cv2.undistortPoints(
                            uv.reshape(1, 1, 2).astype(np.float32),
                            camera_matrix,
                            distortion,
                        ).reshape(2)
                        uv = undistorted_uv.astype(np.float64)
                        uv_norm = undistorted_uv_norm.astype(np.float64)
                    if packet_flip_lr or packet_flip_ud:
                        uv = _flip_phase2_uv(
                            uv,
                            cam_info.width,
                            cam_info.height,
                            flip_lr=packet_flip_lr,
                            flip_ud=packet_flip_ud,
                        )
                        uv_norm = _phase2_uv_to_norm(
                            uv,
                            cam_info.packet_metadata["rectified_intrinsics"],
                        )
                    ray_camera = np.array([uv_norm[0], uv_norm[1], 1.0], dtype=np.float64)
                    ray_camera /= np.linalg.norm(ray_camera)
                    train_track_observations[sparse_track["feature_id"]].append({
                        "camera_center": camera_center,
                        "ray_world": R_CtoG @ ray_camera,
                        "R_GtoC": R_GtoC,
                        "T_GtoC": T_GtoC,
                        "uv_norm": uv_norm,
                        "uv": uv,
                        "image_path": cam_info.image_path,
                        "width": cam_info.width,
                        "height": cam_info.height,
                    })

            global_image_index += 1

    cam_infos = sorted(
        cam_infos,
        key=lambda x: (
            ((x.packet_metadata or {}).get("packet_index", float("inf"))),
            ((x.packet_metadata or {}).get("camera_id", float("inf"))),
            ((x.packet_metadata or {}).get("timestamp_sec", float("inf"))),
            x.image_name,
        ),
    )
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd = _build_phase2_seed_point_cloud(
        train_cam_infos,
        train_track_observations,
        fallback_depth_scale=max(nerf_normalization["radius"] * 0.05, 0.25),
    )

    ply_name = "phase2_points3d.ply"
    if packet_stride > 1 or packet_offset > 0:
        ply_name = f"phase2_points3d_stride{packet_stride}_offset{packet_offset}.ply"
    ply_path = os.path.join(path, ply_name)
    storePly(ply_path, pcd.points, np.clip(pcd.colors * 255.0, 0.0, 255.0).astype(np.uint8))
    pcd = fetchPly(ply_path)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "OpenVINSPackets": readOpenVINSPacketSceneInfo,
}
