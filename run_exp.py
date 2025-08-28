import subprocess

res = [8,4,2,1]
base_scene = 'base_scene'
split2_scene = 'split2_scene'
split3_scene = 'split3_scene'

for i in res:
    for j in range(3):
        if j == 0:
            exp_str = f'python train.py -s {base_scene}/model0 -m base_{i} -r {i} --default --eval --pkl_name base_r{i}.pkl -x 0'
        elif j == 1:
            exp_str = f'python train.py -s {split2_scene}/model0 -m split2_{i} -r {i} --eval --pkl_name split2_r{i}.pkl -x 1'
        elif j == 2:
            exp_str = f'python train.py -s {split3_scene}/model0 -m split3_{i} -r {i} --eval --pkl_name split3_r{i}.pkl -x 2'

        subprocess.call(exp_str,shell=True)