import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 18

base_pkl = 'base_r2.pkl'
new_pkl = 'jug_trash_r2.pkl'

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


fig, axs = plt.subplots(2,1)

base_time = np.array(base_data['times'])/1e9
base_time[:] = base_time[:] - base_time[0]

new_time = np.array(new_data['times'])/1e9
new_time[:] = new_time[:] - new_time[0]

base_loss_sm = smooth(base_data['losses'],0.9)
comb_loss_sm = smooth(new_data['losses'],0.9)

axs[0].set_title('Loss per iteration')
axs[0].plot(base_data['losses'],c='tab:red',alpha=0.5)
axs[0].plot(new_data['losses'],c='tab:blue',alpha=0.5)
axs[0].plot(base_loss_sm,label=f"Base T:{base_time[-1]:0.2f} Loss sum:{np.sum(base_data['losses']):0.4f}",c='tab:red',alpha=1,linewidth=3)
axs[0].plot(comb_loss_sm,label=f"Combined T:{new_time[-1]:0.2f}  Loss sum:{np.sum(new_data['losses']):0.4f}",c='tab:blue',alpha=1,linewidth=3)
axs[0].legend()

axs[1].set_title('Number of Gaussians')
axs[1].plot(base_data['num_gaussians'],label='Base Gaussians',c='tab:red',alpha=0.5,linewidth=5)
axs[1].plot(new_data['num_gaussians'],label='Combined Gaussians',c='tab:blue',alpha=0.5,linewidth=5)
axs[1].legend()

plt.show()