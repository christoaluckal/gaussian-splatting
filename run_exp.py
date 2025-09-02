import subprocess

res = [8,4,2]
base_scene = 'base_scene'
split2_scene = 'split2_scene'
split3_scene = 'split3_scene'

for i in res:
    for j in range(11):
        if j == 0:
            exp_str = f'python train.py -s {base_scene}/model0 -m base_{i} -r {i} --default --eval --pkl_name base_r{i}.pkl -x 0'
            # continue
        elif j == 1:
            continue
        else:
            exp_str = f'python train.py -s split{j}_scene/model0 -m split{j}_{i} -r {i} --eval --pkl_name split{j}_r{i}.pkl -x {j-1} --splitter_itr {10000//(j-1)}'
        # elif j == 1:
        #     exp_str = f'python train.py -s {split2_scene}/model0 -m split2_{i} -r {i} --eval --pkl_name split2_r{i}.pkl -x 1'
        # elif j == 2:
        #     exp_str = f'python train.py -s {split3_scene}/model0 -m split3_{i} -r {i} --eval --pkl_name split3_r{i}.pkl -x 2'

        subprocess.call(exp_str,shell=True)

# exp_str = f'python colmap_splitter/split_list.py -s ~/Downloads/Splat-20250821T013834Z-1-001/Splat/scene -m ./base_scene --num_test 10 --default'
# subprocess.call(exp_str,shell=True)
# for i in range(2,11):
#     exp_str = f'python colmap_splitter/split_list.py -s ~/Downloads/Splat-20250821T013834Z-1-001/Splat/scene -m ./split{i}_scene --num_test {10//i} --split_num {i}'

#     subprocess.call(exp_str,shell=True)