import numpy as np
import matplotlib.pyplot as plt 

pred_disp = np.load('pred_disp28.npy')
surface_normal = np.load('surface_normal28.npy')
ground_mask = np.load('ground_mask28.npy')


surface_normal_inv=surface_normal.transpose(1,2,0)
fig, ax = plt.subplots(1,dpi=300)
height, width = pred_disp.shape
ax.set_ylim(height, 0)
ax.set_xlim(0, width )
ax.axis('off')
fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0,0)

#ax.scatter(y,x,s=0.1,c=values,alpha=0.5,cmap='jet',vmin=-1,vmax=10)
#ax.imshow(surface_normal_inv)
########################################################################
#position=fig.add_axes([0.9, 0.9, 0.1, 0.5])#位置[左,下,右,上]
#cb=plt.colorbar(im,orientation='vertical')
#plt.savefig(os.path.join(os.path.dirname(__file__), "blend_imgs","{:010d}.png".format(i)))




#28:min depth0.14825621 max depth5.4407187 ratio27.899173736572266