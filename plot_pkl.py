import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

res = sys.argv[1:][0]

plt.rcParams['font.size'] = 18

base_pkl = f'base_r{res}.pkl'

# load base data
with open(base_pkl,'rb') as f:
    base_data = pickle.load(f)

# Load split data up to 10
split_data = {}
for i in range(2, 11):
    pkl_file = f'split{i}_r{res}.pkl'
    with open(pkl_file, 'rb') as f:
        split_data[i] = pickle.load(f)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def extract(data):
    base_time = np.array(data['times'])/1e9
    base_time[:] = base_time[:] - base_time[0]
    base_loss = data['losses']
    base_loss_sm = smooth(data['losses'],0.9)
    base_l1 = data['l1s']
    base_l1p = []
    base_numg = data['num_gaussians']
    for l in base_l1:
        if not isinstance(l,float):
            base_l1p.append(l.item())
    base_psnr = data['psnrs']
    base_psnrp = []
    for l in base_psnr:
        if not isinstance(l,float):
            base_psnrp.append(l.item())
    
    idxs = np.arange(1000,31000,5000).reshape(-1,1)
    base_l1p = np.array(base_l1p).reshape(-1,1)
    base_l1p = np.concatenate((idxs,base_l1p),axis=1)

    base_psnrp = np.array(base_psnrp).reshape(-1,1)
    base_psnrp = np.concatenate((idxs,base_psnrp),axis=1)

    return base_time, base_loss, base_loss_sm, base_l1p, base_psnrp, base_numg

# Extract base
t_base,loss_base,loss_base_sm,l1loss_base,psnr_base,ng_base = extract(base_data)

# Extract splits
split_results = {}
for i, data in split_data.items():
    split_results[i] = extract(data)

# Figure
fig, axs = plt.subplots(2,2,figsize=(16,10))

base_t = t_base[-1]
base_loss_sum = np.sum(loss_base)

# compute suptitle string
summary_str = f"Base Time: {base_t:0.2f} Base Loss Sum:{base_loss_sum:0.2f}\n"
for i,(t,loss,_,_,_,_) in split_results.items():
    summary_str += f" Aug{i} Time: {t[-1]:0.2f} Aug{i} Loss Sum:{np.sum(loss):0.2f}\n"

fig.suptitle(summary_str)

# Loss per iteration
axs[1,0].set_title('Loss per iteration')
axs[1,0].plot(loss_base,c='tab:red',alpha=0.5)
axs[1,0].plot(loss_base_sm,label="Base",c='tab:red',alpha=1,linewidth=3)

colors = plt.cm.tab10.colors  # color palette
for idx,(i,(t,loss,loss_sm,_,_,_)) in enumerate(split_results.items()):
    color = colors[(i-2)%10]
    axs[1,0].plot(loss,c=color,alpha=0.5)
    axs[1,0].plot(loss_sm,label=f"Augmented {i}",c=color,alpha=1,linewidth=3)

axs[1,0].legend()

# Eval L1 Loss
axs[0,0].set_title('Eval L1 Loss')
axs[0,0].plot(l1loss_base[:,0],l1loss_base[:,1],label='Base',c='tab:red',linewidth=5)
axs[0,0].scatter(l1loss_base[:,0],l1loss_base[:,1],c='tab:red',s=100)

for idx,(i,(t,_,_,l1loss,_,_)) in enumerate(split_results.items()):
    color = colors[(i-2)%10]
    axs[0,0].plot(l1loss[:,0],l1loss[:,1],label=f'Augmented {i}',c=color,linewidth=5)
    axs[0,0].scatter(l1loss[:,0],l1loss[:,1],c=color,s=100)

axs[0,0].legend()

# Eval PSNR
axs[0,1].set_title('Eval PSNR')
axs[0,1].plot(psnr_base[:,0],psnr_base[:,1],label='Base',c='tab:red',linewidth=5)
axs[0,1].scatter(psnr_base[:,0],psnr_base[:,1],c='tab:red',s=100)

for idx,(i,(t,_,_,_,psnr,_)) in enumerate(split_results.items()):
    color = colors[(i-2)%10]
    axs[0,1].plot(psnr[:,0],psnr[:,1],label=f'Augmented {i}',c=color,linewidth=5)
    axs[0,1].scatter(psnr[:,0],psnr[:,1],c=color,s=100)

axs[0,1].legend()

# Number of Gaussians
axs[1,1].set_title('Number of Gaussians')
axs[1,1].plot(ng_base,label='Base',c='tab:red',linewidth=5)
for idx,(i,(t,_,_,_,_,ng)) in enumerate(split_results.items()):
    color = colors[(i-2)%10]
    axs[1,1].plot(ng,label=f'Augmented {i}',c=color,linewidth=5)

axs[1,1].legend()

plt.show()
