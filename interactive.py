import gabor_kernel as gk
import gabor_noise as gn
while True:
    choice=input("请输入序号;1 展示gabor_kernel;2 展示gabor_noise;3 退出")
    if int(choice)==1:
        w=input("请输入方向:")
        new_im=gk.fft_functionToImage(float(w))
        new_im.show()
        new_im2=gk.functionToImage(float(w))
        new_im2.show()
    elif int(choice)==2:
        
        impluse_num=input("impluse_num:")

        gn.gabor_noise_fft(int(impluse_num))
    else :
        break
