import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.rcParams['font.size'] = 32
args = sys.argv[1:]
base_pkl = args[0]
base_name = args[1]
new_pkl = args[2]
new_name = args[3]

base_data = None
new_data = None

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed


with open(base_pkl,'rb') as f:
    base_data = pickle.load(f)

with open(new_pkl,'rb') as f:
    new_data = pickle.load(f)


fig, axs = plt.subplots(2,2)

def extract(data):
    base_time = np.array(data['times'])/1e9
    
    base_time[:] = base_time[:] - base_time[0]
    base_loss = data['losses']
    base_loss_sm = smooth(data['losses'],0.9)
    base_l1 = data['l1s']
    base_l1p = []
    base_numg = data['num_gaussians']
    for idx,l in enumerate(base_l1):
        if type(l) != float:
            base_l1p.append(l.item())
    base_psnr = data['psnrs']
    base_psnrp = []
    for idx,l in enumerate(base_psnr):
        if type(l) != float:
            base_psnrp.append(l.item())
    
    idxs = np.arange(200,35000,500).reshape(-1,1)
    base_l1p = np.array(base_l1p).reshape(-1,1)
    try:
        base_l1p = np.concatenate((idxs,base_l1p),axis=1)
    except:
        base_l1p = np.concatenate((np.arange(len(base_l1p)).reshape(-1,1),base_l1p),axis=1)

    base_psnrp = np.array(base_psnrp).reshape(-1,1)
    try:
        base_psnrp = np.concatenate((idxs,base_psnrp),axis=1)
    except:
        base_psnrp = np.concatenate((np.arange(len(base_psnrp)).reshape(-1,1),base_psnrp),axis=1)

    return base_time, base_loss, base_loss_sm, base_l1p, base_psnrp, base_numg

t1,loss1,loss1sm,l1loss1,psnr1,ng1 = extract(base_data)
t2,loss2,loss2sm,l1loss2,psnr2,ng2 = extract(new_data)

base_t = t1[-1]
aug_t = t2[-1]
base_loss_sum = np.sum(loss1)
aug_loss_sum = np.sum(loss2)

fig.suptitle(f"|{base_name}| Time: {base_t:0.2f} {base_name} Loss Sum:{base_loss_sum:0.2f}\n|{new_name}| Time: {aug_t:0.2f} {new_name} Loss Sum:{aug_loss_sum:0.2f}")

axs[1,0].set_title('Loss per iteration')
axs[1,0].plot(loss1,c='tab:red',alpha=0.5)
axs[1,0].plot(loss2,c='tab:blue',alpha=0.5)
axs[1,0].plot(loss1sm,label=f"{base_name}",c='tab:red',alpha=1,linewidth=3)
axs[1,0].plot(loss2sm,label=f"{new_name}",c='tab:blue',alpha=1,linewidth=3)
axs[1,0].legend()

axs[0,0].set_title('Eval L1 Loss')
axs[0,0].plot(l1loss1[:,0],l1loss1[:,1],label=f'{base_name}',c='tab:red',linewidth=5)
axs[0,0].plot(l1loss2[:,0],l1loss2[:,1],label=f'{new_name}',c='tab:blue',linewidth=5)
axs[0,0].scatter(l1loss1[:,0],l1loss1[:,1],c='tab:red',s=100)
axs[0,0].scatter(l1loss2[:,0],l1loss2[:,1],c='tab:blue',s=100)
axs[0,0].legend()

axs[0,1].set_title('Eval PSNR')
axs[0,1].plot(psnr1[:,0],psnr1[:,1],label=f'{base_name}',c='tab:red',linewidth=5)
axs[0,1].plot(psnr2[:,0],psnr2[:,1],label=f'{new_name}',c='tab:blue',linewidth=5)
axs[0,1].scatter(psnr1[:,0],psnr1[:,1],c='tab:red',s=100)
axs[0,1].scatter(psnr2[:,0],psnr2[:,1],c='tab:blue',s=100)
axs[0,1].legend()

axs[1,1].set_title('Number of Gaussians')
axs[1,1].plot(ng1,label=f'{base_name}',c='tab:red',linewidth=5)
axs[1,1].plot(ng2,label=f'{new_name}',c='tab:blue',linewidth=5)
axs[1,1].legend()
plt.show()