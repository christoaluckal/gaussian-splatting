import subprocess

res = [8,4,2]
base_name = 'kitchen'
base_scene = f'{base_name}_base'
split_scene = f'{base_name}_split'

for i in res:
    for j in range(5):
        if j == 0:
            exp_str = f'python train_nomask.py -s input/{base_scene}/model0 -m output/{base_scene}_{i} -r {i} --default --eval --pkl_name output/{base_scene}_{i}/result.pkl -x 0'
        elif j == 1:
            continue
        else:
            exp_str = f'python train_nomask.py -s input/{split_scene}{j}/model0 -m output/{split_scene}{j}_{i} -r {i} --eval --pkl_name output/{split_scene}{j}_{i}/result.pkl -x {j-1} --splitter_itr {10000//(j-1)}'
        # elif j == 1:
        #     exp_str = f'python train.py -s {split2_scene}/model0 -m split2_{i} -r {i} --eval --pkl_name split2_r{i}.pkl -x 1'
        # elif j == 2:
        #     exp_str = f'python train.py -s {split3_scene}/model0 -m split3_{i} -r {i} --eval --pkl_name split3_r{i}.pkl -x 2'

        subprocess.call(exp_str,shell=True)


# for i in range(0,5):
#     if i == 0:
#         exp_str = f'python colmap_splitter/split_xyz.py -s ~/Downloads/360_v2/{base_name} -m ./input/{base_scene} --num_test 10 --split_num 1'
#     elif i == 1:
#         continue
#     else:
#         exp_str = f'python colmap_splitter/split_xyz.py -s ~/Downloads/360_v2/{base_name} -m ./input/{split_scene}{i} --num_test {10//i} --split_num {i}'

#     subprocess.call(exp_str,shell=True)