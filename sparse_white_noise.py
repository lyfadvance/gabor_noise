import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random

def sparse_white_noise(point_num):
    #x=random.sample(range(0,399),point_num)
    #y=random.sample(range(0,399),point_num)
    x=[random.uniform(-40,40) for i in range(point_num)]
    y=[random.uniform(-40,40) for i in range(point_num)]
    w=[random.uniform(-1,1) for i in range(point_num)]
    oritination=[random.uniform(0,2*np.pi) for i in range(point_num)]
    '''
    print(x)
    print(w)
    image=np.zeros((400,400))
    image[x,y]=255
    Z_fft2=np.fft.fft2(image)
    Z_fft2_sh=abs(np.fft.fftshift(Z_fft2))
    
    plt.subplot(221)
    plt.imshow(image,cmap='gray')
    plt.title('sparse white noise')

    plt.subplot(222)
    plt.imshow(abs(Z_fft2),cmap=plt.cm.gray)
    plt.title('fft2')

    plt.subplot(223)
    plt.imshow(Z_fft2_sh,cmap=plt.cm.gray_r)
    plt.title('fft2-shift')

    plt.show()
    '''
    return x,y,w,oritination
x,y,w,oritination=sparse_white_noise(5)
