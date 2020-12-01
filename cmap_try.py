import matplotlib.pyplot as plt
import numpy as np
import matplotlib as m


ones = np.ones((192,640))
half = ones*(0.7)
fig, ax = plt.subplots(1,dpi=300)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0,0)
X=range(0,640,1)
Y=range(0,640,1)
ax.imshow(half,cmap='plasma',vmin=-1,vmax=1)
plt.show()
plt.close()