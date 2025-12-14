import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

plt.rcParams['font.size'] = 32
args = sys.argv[1:]
name = args[0]
resolution = args[1]
count = int(args[2])

base_pkl = os.path.join('output',f'{name}_base_{resolution}','result.pkl')
base_name = f'{name} B'

pairs = [[base_pkl,base_name]]
for i in range(2,count):
    pkl_path = os.path.join('output',f'{name}_split{i}_{resolution}','result.pkl')
    pkl_name = f'{name} {i}'
    pairs.append([pkl_path,pkl_name])

# # Check even number of arguments
# if len(args) % 2 != 0:
#     print("Usage: python plot_compare.py file1.pkl name1 file2.pkl name2 ...")
#     sys.exit(1)

# # Group pickle paths and names
# pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]



def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def extract(data):
    base_time = np.array(data['times']) / 1e9
    base_time[:] = base_time[:] - base_time[0]

    base_loss = data['losses']
    base_loss_sm = smooth(data['losses'], 0.98)
    base_l1 = data['l1s']
    base_numg = data['num_gaussians']

    # Convert tensors to floats
    base_l1p = [l.item() if not isinstance(l, float) else l for l in base_l1]
    base_psnrp = [l.item() if not isinstance(l, float) else l for l in data['psnrs']]
    
    idxs = np.arange(1000, 31000, 5000)
    thirty = np.array([30000])
    idxs = np.concatenate((idxs,thirty)).reshape(-1,1)
    base_l1p = np.array(base_l1p).reshape(-1, 1)
    base_psnrp = np.array(base_psnrp).reshape(-1, 1)
    # Try to align shapes
    try:
        base_l1p = np.concatenate((idxs, base_l1p), axis=1)
    except:
        idxs = np.arange(len(base_l1p)).reshape(-1, 1)
        base_l1p = np.concatenate((idxs, base_l1p), axis=1)

    try:
        base_psnrp = np.concatenate((idxs, base_psnrp), axis=1)
    except:
        idxs = np.arange(len(base_psnrp)).reshape(-1, 1)
        base_psnrp = np.concatenate((idxs, base_psnrp), axis=1)

    return base_time, base_loss, base_loss_sm, base_l1p, base_psnrp, base_numg, idxs


# Load all pickles and extract data
datasets = []
for pkl_path, name in pairs:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    datasets.append((name, *extract(data)))


# Plot setup
fig, axs = plt.subplots(2, 2, figsize=(25, 18))

# Build title summary
summary_lines = []
for name, t, loss, *_ in datasets:
    summary_lines.append(f"|{name}| Time: {t[-1]:0.2f} Loss Sum:{np.sum(loss):0.2f}")
fig.suptitle("\n".join(summary_lines))


# ---- Plot sections ----

# (1,0) Loss
axs[1, 0].set_title('Loss per iteration')
# for name, t, loss, loss_sm, *_ in datasets:
for name, t, loss, loss_sm, l1loss, psnr, numg, idxs in datasets:
    # axs[1, 0].plot(loss, alpha=0.3, label=f'{name} (raw)')
    axs[1, 0].plot(loss_sm, label=f'{name} (smooth)', linewidth=3)
axs[1, 0].legend()
axs[1,0].grid()

# (0,0) Eval L1
axs[0, 0].set_title('Eval L1 Loss')
for name, t, loss, loss_sm, l1loss, psnr, numg, idxs in datasets:
    axs[0, 0].plot(l1loss[:, 0], l1loss[:, 1], label=name, linewidth=5)
    axs[0, 0].scatter(l1loss[:, 0], l1loss[:, 1], s=80)
    axs[0,0].set_xticks(idxs.flatten())
    axs[0,0].axvline(10000,color='black')
axs[0, 0].legend()
axs[0,0].grid()

# (0,1) Eval PSNR
axs[0, 1].set_title('Eval PSNR')
for name, t, loss, loss_sm, l1loss, psnr, numg, idxs in datasets:
    axs[0, 1].plot(psnr[:, 0], psnr[:, 1], label=name, linewidth=5)
    axs[0, 1].scatter(psnr[:, 0], psnr[:, 1], s=80)
    axs[0,1].set_xticks(idxs.flatten())
    axs[0,1].axvline(10000,color='black')
axs[0, 1].legend()
axs[0,1].grid()

# (1,1) Number of Gaussians
axs[1, 1].set_title('Number of Gaussians')
for name, t, loss, loss_sm, l1loss, psnr, numg, idxs in datasets:
    axs[1, 1].plot(numg, label=name, linewidth=5)
axs[1, 1].legend()
axs[1,1].grid()

plt.tight_layout()
plt.show()
