import numpy as np
import os
import pprint
import shutil
import argparse
import random

class Splitter:
    def __init__(self,
                 scene_path:str=None,
                 new_scene_path: str = None,
                 is_default: bool = None
                 ):
        self.scene_base = scene_path
        self.is_default = is_default
        # sparse_0 = os.path.join('sparse','0')
        sparse_0 = 'sparse_txt'
        self.cameras = os.path.join(self.scene_base, sparse_0, 'cameras.txt')
        self.images = os.path.join(self.scene_base, sparse_0, 'images.txt')
        self.points3D = os.path.join(self.scene_base, sparse_0, 'points3D.txt')
        self.new_scene_path = os.path.join(new_scene_path)

        self.id_dict = {}
        self.p3_dict = {}

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

    def build_composite_idlist(self,large_dict, skip_idx):
        idlist = []
        for i,kv in enumerate(large_dict.items()):
            k = kv[0]
            v = kv[1]
            if i==skip_idx:
                continue
            else:
                idlist += list(v.keys())

        return idlist

    def build_model(self, split_frac=[0.33,0.33,0.34], num_test=None):
        # if not split_frame:
        #     i = 0
        #     with open(self.images, 'r') as f:
        #         image_data = f.readlines()[4:]
        #         while i < len(image_data)//2:
        #             iv_row = i
        #             iv_row_data = image_data[iv_row].split()
        #             image_name = iv_row_data[-1]
        #             i+=2

        #         split_frame = image_name

        if self.is_default:
            split_frac = [1.0]


        model_image_p2d = {}
        for i in range(len(split_frac)):
            model_image_p2d[f'm{i}_image_p2d'] = {}
            self.id_dict[f'm{i}'] = {}
            self.p3_dict[f'm{i}'] = {}

        split_frac_idxs = []


        with open(self.images, 'r') as f:
            image_data = f.readlines()[4:]
            image_count = len(image_data)//2
            split_counts = [int(image_count * frac) for frac in split_frac]
            split_counts[-1] = image_count - sum(split_counts[:-1])
            split_frac_idxs = [0]
            for count in split_counts:
                split_frac_idxs.append(split_frac_idxs[-1] + count)

            print(split_frac_idxs)
            ans = input('Good?')
            if ans == 'n':
                exit(-1)

            image_row = 0
            curr_block = 0
            icount = 0
            while image_row < len(image_data):
                iv_row = image_row
                p2d_row = image_row+1

                iv_row_data = image_data[iv_row].split()
                image_name = iv_row_data[-1]

                p2d_data = np.array(image_data[p2d_row].split(), dtype=float).reshape(-1, 3)

                p2d_ids = np.unique(p2d_data[:, 2].astype(int).astype(str))
                self.id_dict[f'm{curr_block}'].update({str(id_): True for id_ in p2d_ids})
        
                filter_ids = self.build_composite_idlist(self.id_dict, curr_block)
                curr_ids = p2d_data[:, 2].astype(int).astype(str)
                mask = ~np.isin(curr_ids, filter_ids)
                filtered_arr = p2d_data[mask]

                model_image_p2d[f'm{curr_block}_image_p2d'][image_name] = [iv_row_data, filtered_arr]


                if icount <= split_frac_idxs[curr_block+1]:
                    icount +=1
                else:
                    curr_block +=1


                image_row += 2


        with open(self.points3D, 'r') as f:
            points_data = f.readlines()[3:]
            i = 0
            curr_block = 0
            while i < len(points_data):
                point_row = points_data[i]
                point_idx = point_row.split()[0]

                for idx, kv in enumerate(self.id_dict.items()):
                    k = kv[0]
                    v = kv[1]
                    if str(point_idx) in self.id_dict[k]:
                        self.p3_dict[f'm{idx}'][point_idx] = point_row.split()[1:]
                        break

                i+=1


        # try:
            # os.makedirs(os.path.join(self.new_scene_path, model1_name, 'sparse','0'),exist_ok=True)
            # os.makedirs(os.path.join(self.new_scene_path, model2_name, 'sparse','0'),exist_ok=True)
        num_dirs = len(split_frac_idxs)-1
        for i in range(num_dirs):    
            os.makedirs(os.path.join(self.new_scene_path, f'model{i}', 'sparse','0'),exist_ok=True)
            self.write_model(f'model{i}', model_image_p2d[f'm{i}_image_p2d'], self.p3_dict[f'm{i}'],num_test)
        # except Exception as e:
        #     print(e)
        #     pass

        # self.write_model(model1_name, m1_image_p2d, m1_p3d,self.num_test)
        # self.write_model(model2_name, m2_image_p2d, m2_p3d,self.num_test)



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',type=str,required=True,help='source scene path')
    parser.add_argument('-m',type=str,required=True,help='destination scene path')
    parser.add_argument('--default',action='store_true')
    parser.add_argument('--num_test',type=int,default=0,help='number of test images per model')
    args = parser.parse_args()
    src_scene = os.path.abspath(args.s)
    dst_scene = os.path.abspath(args.m)
    s = Splitter(scene_path=src_scene,
                 new_scene_path=dst_scene,
                 is_default=args.default
                 )
    s.build_model(num_test=args.num_test)