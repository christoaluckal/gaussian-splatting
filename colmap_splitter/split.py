import numpy as np
import os
import pprint
import shutil
import argparse

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

    def write_model(self, name, img_dict, point_dict):
        m1_im = os.path.join(self.new_scene_path, name, 'sparse','0','images.txt')
        m1_pt = os.path.join(self.new_scene_path, name, 'sparse','0','points3D.txt')

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



    def build_model(self, model1_name=None, model2_name=None, split_frame=None):
        if not split_frame:
            i = 0
            with open(self.images, 'r') as f:
                image_data = f.readlines()[4:]
                while i < len(image_data)//2:
                    iv_row = i
                    iv_row_data = image_data[iv_row].split()
                    image_name = iv_row_data[-1]
                    i+=2

                split_frame = image_name

        print(f"Splitting at {split_frame}")
        m1_image_p2d = {}
        m2_image_p2d = {}
        self.id_dict = {}

        with open(self.images, 'r') as f:
            image_data = f.readlines()[4:]
            i = 0
            split = False
            while i < len(image_data):
                iv_row = i
                p2d_row = i+1

                iv_row_data = image_data[iv_row].split()
                image_name = iv_row_data[-1]

                p2d_data = np.array(image_data[p2d_row].split(), dtype=float).reshape(-1, 3)
                if not split:
                    # Unique IDs as strings
                    p2d_ids = np.unique(p2d_data[:, 2].astype(int).astype(str))
                    self.id_dict.update({str(id_): True for id_ in p2d_ids})
                    m1_image_p2d[image_name] = [iv_row_data, p2d_data]
                else:
                    filter_ids = np.array(list(self.id_dict.keys()), dtype=str)
                    curr_ids = p2d_data[:, 2].astype(int).astype(str)
                    mask = ~np.isin(curr_ids, filter_ids)
                    filtered_arr = p2d_data[mask]
                    m2_image_p2d[image_name] = [iv_row_data, filtered_arr]

                if image_name == split_frame:
                    split = True
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

        self.write_model(model1_name, m1_image_p2d, m1_p3d)
        self.write_model(model2_name, m2_image_p2d, m2_p3d)



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',type=str,required=True)
    parser.add_argument('-m',type=str,required=True)
    parser.add_argument('-m1',type=str,default='model1')
    parser.add_argument('-m2',type=str,default='model2')
    parser.add_argument('-f',type=str,default=None)
    args = parser.parse_args()
    src_scene = os.path.abspath(args.s)
    dst_scene = os.path.abspath(args.m)
    s = Splitter(scene_path=src_scene,
                 new_scene_path=dst_scene)
    s.build_model(model1_name=args.m1,model2_name=args.m2,split_frame=args.f)