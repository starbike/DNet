import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.font_manager as font_manager
import argparse
#plt.rc('font',family='Times New Roman')
parser = argparse.ArgumentParser(description='Options for trajectory visualization.')
parser.add_argument('--type', type=str, default='min', help='min/max/median gradient point to plot loss')
parser.add_argument('--gt_div', type=float, help='the corresponding gt_div of the point')
parser.add_argument('--id', type=int, help='selected frame to plot')
parser.add_argument('--range', nargs="+", type=float, help='lower and upper limit for x axis')
args = parser.parse_args()

f_ids = np.arange(args.range[0], args.range[1], step = 0.01)
fig, ax = plt.subplots(1,dpi=300,figsize=(5,2))

'''
depth_min = 0.2255
depth_max = 0.2525
depth_median = 0.6833

depth_min = 0.1995
depth_max = 0.1996
depth_median = 0.4401
'''

l1_losses = np.load('./{}_grad/l1_losses_{}.npy'.format(args.type, args.id))
ssim_losses = np.load("./{}_grad/ssim_losses_{}.npy".format(args.type, args.id))
reprojection_losses = np.load("./{}_grad/reprojection_losses_{}.npy".format(args.type, args.id))

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.plot(f_ids, l1_losses, linewidth=1)
ax.plot(f_ids, ssim_losses, linewidth=1)
ax.plot(f_ids, reprojection_losses,linewidth=1)
x0 = args.gt_div
plt.annotate('gt_div={}'.format(x0), xy=(x0, 0), xytext=(x0, 0.1), arrowprops=dict(arrowstyle='->'))  # 添加注释
#ax.plot(f_ids, ideal_scale, linewidth=1)
plt.xlim(args.range[0], args.range[1])
plt.ylim(0,0.9)
plt.xlabel('Depth')
plt.ylabel('Error')
plt.legend(['L1_Loss','SSIM', 'Reprojection'],loc='upper center', bbox_to_anchor=(0.8,1))
#plt.legend(['GT_Div', 'DGC_Div'])
plt.savefig('./{}_grad/depth_plot_{}_{}.png'.format(args.type, args.id, args.type),bbox_inches='tight')
