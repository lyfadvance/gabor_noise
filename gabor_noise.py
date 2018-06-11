import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random

from sparse_white_noise import sparse_white_noise
from gabor_kernel import gabor_kernel

def gabor_noise(x,y,impluse,point_num=100):
    sample_x,sample_y,sample_w,oritination=impluse
    sum=0
    for i in range(point_num):
        sum=sum+sample_w[i]*gabor_kernel(x-sample_x[i],y-sample_y[i],w=oritination[i])
    return sum
def gabor_noise_fft(point_num=100):
    sample_x,sample_y,sample_w,oritination=sparse_white_noise(point_num)
    impluse=[sample_x,sample_y,sample_w,oritination]
    X=np.arange(-60,60,0.1)
    Y=np.arange(-60,60,0.1)
    X,Y=np.meshgrid(X,Y)
    Z=gabor_noise(X,Y,impluse,point_num)
    Z_fft2 = np.fft.fft2(Z)
    Z_fft2_sh = abs(np.fft.fftshift(Z_fft2))

    plt.subplot(221)
    plt.imshow(Z,cmap='gray')
    plt.title('Original')

    plt.subplot(222)
    plt.imshow(abs(Z_fft2),cmap=plt.cm.gray)
    plt.title('fft2')

    plt.subplot(223)
    plt.imshow(Z_fft2_sh,cmap=plt.cm.gray_r)
    plt.title('fft2-shift')

    plt.subplot(224)
    plt.plot(Z_fft2_sh[128,:])
    plt.title('x=128')

    plt.show()
#gabor_noise_fft(gabor_noise)
