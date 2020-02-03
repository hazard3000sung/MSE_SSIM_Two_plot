import numbers
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from skimage import io
from skimage import img_as_float
from skimage.measure import compare_ssim as ssim

try:
    import scipy as sp
    import cv2

except:
    pass

img = cv2.imread('') 
# <---file read
img1 = cv2.imread('')   
#<----file2 read
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#<--- make file to gray color
img1  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img_1 = img_as_float(io.imread(''))
img_2 = img_as_float(io.imread(''))
# <--- this just bring the original file! just for showing

def mse(x,y):
    return np.linalg.norm(x-y)
    #mse norm


mse_none = mse(img,img)
ssim_none = ssim(img,img,multichannel=False,sigma=0.5,gaussian_weights=True,use_sample_covariance=False)

mse_1 = mse(img,img1)
ssim_1 = ssim(img,img1,multichannel=False,sigma=50,gaussian_weights=True,use_sample_covariance=False)

fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(5,5),sharex=True,sharey=True)
ax = axes.ravel()


label = 'MSE :{:.2f}, SSIM :{:.2f}'

ax[0].imshow(img_1,cmap=plt.cm.gray,vmin=0,vmax=1)
ax[0].set_xlabel(label.format(mse_none,ssim_none))
ax[0].set_title('original')


ax[1].imshow(img_2,cmap=plt.cm.gray,vmin=0,vmax=1)
ax[1].set_xlabel(label.format(mse_1,ssim_1))
ax[1].set_title('contrast')


plt.tight_layout()
#plt.axis('off') <--- if you don't want axis 
#plt.savefig() <---- if you want to save pic
plt.show()
