import numpy as np
import os
import pprint
import shutil
import argparse
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Splitter:
    def __init__(self,
                 scene_path:str=None,
                 new_scene_path: str = None
                 ):
        self.scene_base = scene_path
        # sparse_0 = os.path.join('sparse','0')
        sparse_0 = 'sparse_txt'
        self.cameras = os.path.join(self.scene_base, sparse_0, 'cameras.txt')
        self.images = os.path.join(self.scene_base, sparse_0, 'images.txt')
        self.points3D = os.path.join(self.scene_base, sparse_0, 'points3D.txt')
        self.new_scene_path = os.path.join(new_scene_path)
        self.image_3d_dict = {}

        self.img_HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            0, 0
        )
        )

        self.points_HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            0, 0
        )
        )

    def copy_and_remove(self, keys, dir):
        shutil.copytree(os.path.join(self.scene_base, 'images'),dir)
        for root, _, files in os.walk(dir):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), dir)
                if rel_path not in keys:
                    os.remove(os.path.join(root, f))

    def write_model(self, name, img_dict, point_dict, num_test=0):
        print("Writing")
        m1_im = os.path.join(self.new_scene_path, name, 'sparse','0','images.txt')
        m1_test = os.path.join(self.new_scene_path, name, 'sparse','0','test.txt')
        m1_pt = os.path.join(self.new_scene_path, name, 'sparse','0','points3D.txt')

        if num_test > 0:
            keys = random.sample(list(img_dict.keys()),num_test)
            with open(m1_test,'a') as f:
                f.write(self.img_HEADER)
                for ke,v in img_dict.items():
                    if ke in keys:
                        s = " ".join(str(x) for x in v[0])
                        f.write(s)
                        f.write('\n')
                        arr_data = v[1]
                        arr_data[:, 2] = arr_data[:, 2].astype(int)
                        for x, y, z in arr_data:
                            f.write(f"{x:.6f} {y:.6f} {int(z)} ")
                        f.write('\n')

            with open(m1_im,'a') as f:
                f.write(self.img_HEADER)
                for ke,v in img_dict.items():
                    if ke not in keys:
                        s = " ".join(str(x) for x in v[0])
                        f.write(s)
                        f.write('\n')
                        arr_data = v[1]
                        arr_data[:, 2] = arr_data[:, 2].astype(int)
                        for x, y, z in arr_data:
                            f.write(f"{x:.6f} {y:.6f} {int(z)} ")
                        f.write('\n')
        else:
            print("Skipping test")
            with open(m1_im,'a') as f:
                f.write(self.img_HEADER)
                for ke,v in img_dict.items():
                    s = " ".join(str(x) for x in v[0])
                    f.write(s)
                    f.write('\n')
                    arr_data = v[1]
                    arr_data[:, 2] = arr_data[:, 2].astype(int)
                    for x, y, z in arr_data:
                        f.write(f"{x:.6f} {y:.6f} {int(z)} ")
                    f.write('\n')

        with open(m1_pt,'a') as f:
            f.write(self.points_HEADER)
            for k,v in point_dict.items():
                f.write(f'{k} ')
                f.write(" ".join(str(x) for x in v))
                f.write('\n')

        shutil.copy2(self.cameras, os.path.join(self.new_scene_path, name, 'sparse','0','cameras.txt'))
        valid_files = set(img_dict.keys())
        base_images = os.path.join(self.new_scene_path, name, 'images')
        res2 = os.path.join(self.new_scene_path, name, 'images_2')
        res4 = os.path.join(self.new_scene_path, name, 'images_4')
        res8 = os.path.join(self.new_scene_path, name, 'images_8')
        self.copy_and_remove(valid_files,base_images)
        self.copy_and_remove(valid_files,res2)
        self.copy_and_remove(valid_files,res4)
        self.copy_and_remove(valid_files,res8)

    def split_points_radial(self, proj_xy, image_names, num_splits=4):
        """
        Split projected 2D points into angular (radial) wedges.
        Returns:
            image_groups: dict mapping image_name -> group_id (1..num_splits)
            bin_edges: array of wedge boundary angles (radians)
        """
        # Center points
        mean = np.mean(proj_xy, axis=0)
        centered = proj_xy - mean

        # Compute angle in radians
        angles = np.arctan2(centered[:, 1], centered[:, 0])  # range [-π, π]
        angles = (angles + 2 * np.pi) % (2 * np.pi)          # normalize to [0, 2π)

        # Define angular bins
        bin_edges = np.linspace(0, 2 * np.pi, num_splits + 1)

        # Assign group by which wedge the angle falls in
        image_groups = {}
        bin_indices = np.digitize(angles, bin_edges, right=False)
        bin_indices[bin_indices > num_splits] = num_splits  # fix edge case

        for name, group in zip(image_names, bin_indices):
            image_groups[name] = int(group-1)

        return image_groups, bin_edges, mean


    def build_model(self, num_split=2, num_test=0):
        i = 0
        xyz = []
        image_names = []
        xyz = []
        with open(self.images, 'r') as f:
            image_data = f.readlines()[4:]  # skip header
            i = 0
            while i < len(image_data):
                iv_row_data = image_data[i].split()
                qw, qx, qy, qz = map(float, iv_row_data[1:5])
                image_names.append(iv_row_data[9])
                tx, ty, tz = map(float, iv_row_data[5:8])
                R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
                t_vec = np.array([tx, ty, tz])
                C = -R_mat.T @ t_vec
                xyz.append(C)
                i += 2  

        xyz = np.array(xyz)
        mean = np.mean(xyz, axis=0)
        xyz_centered = xyz - mean

        # covariance + eigen decomposition
        cov = np.cov(xyz_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # sort eigenvectors by eigenvalue (largest first)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # make a right-handed coordinate system
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1

        R_align = eigvecs.T  # rotation matrix to align to XYZ

        # -------------------------
        # Step 2: Apply alignment
        # -------------------------
        xyz = (R_align @ xyz_centered.T).T

        proj_xy = xyz[:, :2]  # project to XY plane

        image_groups, bin_edges, center = self.split_points_radial(proj_xy, image_names, num_splits=num_split)

        group_image_p2d = {i: {} for i in range(num_split)}
        group_p3d = {i: {} for i in range(num_split)}
        id_dict = {i: {} for i in range(num_split)}

        with open(self.images, 'r') as f:
            image_data = f.readlines()[4:]
            i = 0
            while i < len(image_data):
                iv_row = i
                p2d_row = i + 1
                iv_row_data = image_data[iv_row].split()
                image_name = iv_row_data[-1]
                p2d_data = np.array(image_data[p2d_row].split(), dtype=float).reshape(-1, 3)

                group_idx = image_groups[image_name]
                p2d_ids = np.unique(p2d_data[:, 2].astype(int).astype(str))
                id_dict[group_idx].update({str(id_): True for id_ in p2d_ids})
                group_image_p2d[group_idx][image_name] = [iv_row_data, p2d_data]

                i += 2

        # assign 3D points
        with open(self.points3D, 'r') as f:
            points_data = f.readlines()[3:]
            for line in points_data:
                point_idx = line.split()[0]
                assigned = False
                for g, ids in id_dict.items():
                    if point_idx in ids:
                        group_p3d[g][point_idx] = line.split()[1:]
                        assigned = True
                        break
                if not assigned:
                    # optionally assign to last group
                    group_p3d[num_split][point_idx] = line.split()[1:]

        # plt.scatter(proj_xy[:,0], proj_xy[:,1], c=[image_groups[n]-1 for n in image_names], cmap='tab10')
        # for b in bin_edges:
        #     plt.axvline(b, color='k', linestyle='--')
        # plt.xlabel('X'); plt.ylabel('Y')
        # plt.show()


        try:
            for i in range(num_split):
                os.makedirs(os.path.join(self.new_scene_path, f'model{i}', 'sparse','0'),exist_ok=True)
                self.write_model(f'model{i}', group_image_p2d[i], group_p3d[i],num_test)
        except Exception as e:
            print(e)
            pass

        




    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',type=str,required=True)
    parser.add_argument('-m',type=str,required=True)
    parser.add_argument('--split_num',type=int,default=1)
    parser.add_argument('--num_test',type=int,default=0)
    args = parser.parse_args()
    for arg_name, arg_value in sorted(vars(args).items()):
        print(f"  {arg_name}: {arg_value}")
    src_scene = os.path.abspath(args.s)
    dst_scene = os.path.abspath(args.m)
    s = Splitter(scene_path=src_scene,
                 new_scene_path=dst_scene)
    s.build_model(num_split=args.split_num, num_test=args.num_test)