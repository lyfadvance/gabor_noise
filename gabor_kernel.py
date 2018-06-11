import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def gabor_kernel(x,y,K=1,a=1/(2*np.pi),F=1/np.pi,w=2/4*np.pi):
    return K*np.exp(-np.pi*a*a*(x*x+y*y))*np.cos(2*np.pi*F*(x*np.cos(w)+y*np.sin(w)))
def plot3d(gabor_kernel):
    fig=plt.figure()
    ax=Axes3D(fig)
    X=np.arange(-10,10,0.05)
    Y=np.arange(-10,10,0.05)
    X,Y=np.meshgrid(X,Y)
    Z=gabor_kernel(X,Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
def functionToImage(w):
    x=np.linspace(-10,10,500)
    y=np.linspace(-10,10,500)
    data=np.zeros((500,500))
    for i in range(500):
        for j in range(500):
            data[i,j]=gabor_kernel(x[i],y[j],w=w)
    data=data*100+100
    new_im=Image.fromarray(data.astype(np.uint8))
    return new_im
def fft_function(x,y,K=1,a=1/(2*np.pi),F=1/np.pi,w=1/4*np.pi):
    return K/(2*a*a)*(np.exp(-np.pi/a/a*(np.square(x-F*np.cos(w))+np.square(y-F*np.sin(w))))+np.exp(-np.pi/a/a*(np.square(x+F*np.cos(w))+np.square(y+F*np.sin(w)))))
def fft_functionToImage(w):
    x=np.linspace(-1,1,500)
    y=np.linspace(-1,1,500)
    data=np.zeros((500,500))
    for i in range(500):
        for j in range(500):
            data[i,j]=fft_function(x[i],y[j],w=w)
    data=data*6+100
    new_im=Image.fromarray(data.astype(np.uint8))
    return new_im
def fft(gabor_kernel):
    X=np.arange(-10,10,0.05)
    Y=np.arange(-10,10,0.05)
    X,Y=np.meshgrid(X,Y)
    Z=gabor_kernel(X,Y)
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
'''
new_im=functionToImage(gabor_kernel)
#plt.imshow(data,cmap=plt.cm.gray,interpolation='nearest')
new_im.show()
new_im.save('gabor.jpg')
#plot3d(gabor_kernel)
print("start")
fft(gabor_kernel)
'''
'''
new_im=fft_functionToImage(fft_function)
new_im.show()
'''
