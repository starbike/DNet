import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.font_manager as font_manager


#plt.rc('font',family='Times New Roman')
f_ids = range(252)
fig, ax = plt.subplots(1,dpi=300,figsize=(5,2))
#mean_scale = np.load('mean_scale.npy')
#t_div = np.load("AirSim/gt_norms_div_AirSim.npy")
#gt_norms = np.load("gt_norms00.npy")
#pred_norms = np.load("pred_norms00.npy")
#gt_mean_depths = np.load("gt_mean_depths.npy")
#pred_mean_depths = np.load("pred_mean_depths.npy")
#gt_td = gt_norms/gt_mean_depths
#pred_td = pred_norms/pred_mean_depths

ratio_med = np.load('AirSim/median_ratios_AirSim_nc.npy')
ratio_dgc = np.load('AirSim/dgc_ratios_AirSim_nc.npy')
ideal_scale = np.load('AirSim/ideal_scale_AirSim_nc.npy')
#print(t_div,ratio_dgc)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
#ax.plot(f_ids, t_div, linewidth=1)
ax.plot(f_ids, ratio_med[:-1], linewidth=1)
ax.plot(f_ids, ratio_dgc[:-1],linewidth=1)
ax.plot(f_ids, ideal_scale[:-1], linewidth=1)
plt.xlim(0,252)
plt.xlabel('Frame')
plt.ylabel('Scale')
#plt.legend(['t scale','gt median', 'dgc', 'ideal scale'])
plt.legend(['gt_median', 'dgc', 'ideal scale'])
#plt.legend(['GT_Div', 'DGC_Div'])
plt.savefig('scale_nc.png',bbox_inches='tight')
plt.close()



