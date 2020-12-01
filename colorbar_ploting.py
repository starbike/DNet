import matplotlib.pyplot as plt
import matplotlib as mpl

# Make a figure and axes with dimensions as desired.
fig,ax = plt.subplots(1,dpi=300)
# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.plasma_r
norm = mpl.colors.Normalize(vmin=5, vmax=10)
ax.axis('off')
fig.set_size_inches(0.5, 1.25)
# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm)
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0,0)
plt.savefig('color_bar.svg',bbox_inches='tight')
