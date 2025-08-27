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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import copy
from pathlib import Path
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        path = Path(args.source_path)
        self.model_paths = path.parent.absolute()
        self.loaded_iter = None
        self.gaussians = gaussians
        self.xtend = args.xtend

        print(f"Creating additional {self.xtend} gaussians")
        self.x_gauss = [copy.deepcopy(self.gaussians) for _ in range(self.xtend)]

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

        self.extension_set = []
        
        for idx in range(self.xtend):
            self.extension_set.append(self.create_2nd_set(idx+1,resolution_scales, args))


        self.current_xidx = 1

    def create_2nd_set(self,index,res_scales, args):
        new_train_cameras = {}
        new_test_cameras = {}
        if os.path.exists(os.path.join(self.model_paths,f'model{index}', "sparse")):
            new_scene_info = sceneLoadTypeCallbacks["Colmap"](os.path.join(self.model_paths,f'model{index}'), "images", "", False, False)
            
        camlist = []
        json_cams = []
        camlist.extend(new_scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        random.shuffle(new_scene_info.train_cameras) 
        new_cameras_extent = new_scene_info.nerf_normalization["radius"]

        for resolution_scale in res_scales:
            print("Loading Training Cameras")
            new_train_cameras[resolution_scale] = cameraList_from_camInfos(new_scene_info.train_cameras, resolution_scale, args, new_scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            new_test_cameras[resolution_scale] = cameraList_from_camInfos(new_scene_info.test_cameras, resolution_scale, args, new_scene_info.is_nerf_synthetic, True)


        self.x_gauss[index-1].create_from_pcd(new_scene_info.point_cloud, new_scene_info.train_cameras, new_cameras_extent)

        xset = [new_train_cameras,new_test_cameras]
        return xset

    def extend(self):
        print(f"Number of Gaussians before extending: {self.gaussians._xyz.shape}")
        if self.current_xidx <= len(self.extension_set):
            print(f'Extension number {self.current_xidx}')
            for k,v in self.train_cameras.items():
                self.train_cameras[k] = self.train_cameras[k] + self.extension_set[self.current_xidx-1][0][k]
            for k,v in self.test_cameras.items():
                self.test_cameras[k] = self.test_cameras[k] + self.extension_set[self.current_xidx-1][1][k]
            self.gaussians.concat_new_gaussian(self.x_gauss[self.current_xidx-1])
            self.current_xidx += 1
            print(f"Number of Gaussians after extending: {self.gaussians._xyz.shape}")
        else:
            print(f"No extensions available")
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
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
