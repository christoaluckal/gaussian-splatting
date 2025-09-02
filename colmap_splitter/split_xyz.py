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



    def build_model(self, model1_name=None, model2_name=None, split_frame=None, num_test=0):
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

        minx, miny = np.min(proj_xy, axis=0)
        maxx, maxy = np.max(proj_xy, axis=0)

        # --- choose one diagonal ---
        # (1) bottom-left to top-right
        x1, y1 = minx, miny
        x2, y2 = maxx, maxy

        # # (2) top-left to bottom-right (uncomment to use)
        # x1, y1 = minx, maxy
        # x2, y2 = maxx, miny

        # line coefficients: ax + by + c = 0
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1

        # signed distance
        d = proj_xy @ np.array([a, b]) + c

        group1 = proj_xy[d > 0]
        group2 = proj_xy[d < 0]
        on_line = proj_xy[np.isclose(d, 0)]

        print("Group 1:", group1.shape[0], "points")
        print("Group 2:", group2.shape[0], "points")
        print("On line:", on_line.shape[0], "points")


        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        image_groups = {}  # image_name -> 1 or 2

        # after you compute proj_xy and d
        for name, dist in zip(image_names, d):
            if dist >= 0:
                image_groups[name] = 1
            else:
                image_groups[name] = 2


        m1_image_p2d = {}
        m2_image_p2d = {}
        self.id_dict = {}

        with open(self.images, 'r') as f:
            image_data = f.readlines()[4:]
            i = 0
            while i < len(image_data):
                iv_row = i
                p2d_row = i+1

                iv_row_data = image_data[iv_row].split()
                image_name = iv_row_data[-1]
                p2d_data = np.array(image_data[p2d_row].split(), dtype=float).reshape(-1, 3)

                if image_groups[image_name] == 1:
                    # keep IDs so we can filter points3D later
                    p2d_ids = np.unique(p2d_data[:, 2].astype(int).astype(str))
                    self.id_dict.update({str(id_): True for id_ in p2d_ids})
                    m1_image_p2d[image_name] = [iv_row_data, p2d_data]
                else:
                    curr_ids = p2d_data[:, 2].astype(int).astype(str)
                    # remove IDs that were already assigned to group1
                    mask = ~np.isin(curr_ids, list(self.id_dict.keys()))
                    filtered_arr = p2d_data[mask]
                    m2_image_p2d[image_name] = [iv_row_data, filtered_arr]

                i += 2

        m1_p3d = {}
        m2_p3d = {}
        with open(self.points3D, 'r') as f:
            points_data = f.readlines()[3:]
            i = 0

            while i < len(points_data):
                point_row = points_data[i]
                point_idx = point_row.split()[0]
                if str(point_idx) in self.id_dict:
                    m1_p3d[point_idx] = point_row.split()[1:]
                else:
                    m2_p3d[point_idx] = point_row.split()[1:]
                i+=1


        try:
            os.makedirs(os.path.join(self.new_scene_path, model1_name, 'sparse','0'),exist_ok=True)
            os.makedirs(os.path.join(self.new_scene_path, model2_name, 'sparse','0'),exist_ok=True)
        except:
            pass

        self.write_model(model1_name, m1_image_p2d, m1_p3d,num_test)
        self.write_model(model2_name, m2_image_p2d, m2_p3d,num_test)



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',type=str,required=True)
    parser.add_argument('-m',type=str,required=True)
    parser.add_argument('-m1',type=str,default='model0')
    parser.add_argument('-m2',type=str,default='model1')
    parser.add_argument('-f',type=str,default=None)
    parser.add_argument('--num_test',type=int,default=0)
    args = parser.parse_args()
    src_scene = os.path.abspath(args.s)
    dst_scene = os.path.abspath(args.m)
    s = Splitter(scene_path=src_scene,
                 new_scene_path=dst_scene)
    s.build_model(model1_name=args.m1,model2_name=args.m2,split_frame=args.f,num_test=args.num_test)