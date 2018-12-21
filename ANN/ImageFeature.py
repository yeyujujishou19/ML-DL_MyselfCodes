
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pylab as pl
import random

#字符图像的特征提取方法
#要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
def SimpleGridFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:RenHui
    '''
    #图像大小归一化
    image = cv2.resize(image,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    #计算网格大小
    grid_size=16
    grid_h = binary.shape[0]/grid_size
    grid_w = binary.shape[1]/grid_size
    #定义特征向量
    feature = np.zeros(grid_size*grid_size)
    for j in range(grid_size):
        for i in range(grid_size):
            grid = binary[int(j*grid_h):int((j+1)*grid_h),int(i*grid_w):int((i+1)*grid_w)]
            feature[j*grid_size+i] = grid[grid==0].size
    return feature

#字符图像的特征提取方法
#要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
def GetImageFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:SXL
    '''
    #图像大小归一化
    image = cv2.resize(image,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)

    #统计二值图中黑点的个数
    Globalgrid = binary[int(0):int(img_h), int(0):int(img_w)]
    iGlobalBlackNum = Globalgrid[Globalgrid == 0].size

    #计算网格大小
    grid_h = binary.shape[0]/8
    grid_w = binary.shape[1]/8
    #定义特征向量
    feature = np.zeros(64)
    for j in range(8):
        for i in range(8):
            #统计每个网格中黑点的个数
            grid = binary[int(j*grid_h):int((j+1)*grid_h),int(i*grid_w):int((i+1)*grid_w)]
            feature[j*8+i] = grid[grid==0].size/iGlobalBlackNum    #每个网格黑点数占整幅图黑点数的比例
    return feature


###########################################################################################################
def BuildGaborKernels(ksize = 5,lamda = 1.5,sigma = 1.0):
    '''
    @description:生成多尺度，多方向的gabor特征
    @参数参考opencv
    @return:多个gabor卷积核所组成的
    @author:SXL
    '''
    filters = []
    for theta in np.array([0,np.pi/4, np.pi/2,np.pi*3/4]):
        kern = cv2.getGaborKernel((ksize,ksize),sigma,
                theta,lamda,0.5,0,ktype=cv2.CV_32F)
        #kern /= 1.5*kern.sum()
        filters.append(kern)

    # pl.figure(1)
    # for temp in range(len(filters)):
    #     pl.subplot(4, 4, temp + 1)
    #     pl.imshow(filters[temp], cmap='gray')
    # pl.show()
    return filters

def GaborFeature(image):
    '''
    @description:提取字符图像的gabor特征
    @image:灰度字符图像
    @return:
    @author:SXL
    '''
    # retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    kernels = BuildGaborKernels(ksize = 7,lamda = 8,sigma = 4)
    dst_imgs = []
    for kernel in kernels:
        img = np.zeros_like(image)
        tmp = cv2.filter2D(image,cv2.CV_8UC3,kernel)
        img = np.maximum(img,tmp,img)
        dst_imgs.append(img)

    pl.figure(2)
    for temp in range(len(dst_imgs)):
        pl.subplot(4,1,temp+1) #第一个4为4个方向，第二个4为4个尺寸
        pl.imshow(dst_imgs[temp], cmap='gray' )
    pl.show()
    return dst_imgs


#字符图像的特征提取方法
#要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
def GetImageFeatureGabor(image):
    '''
    @description:提取经过Gabor滤波后字符图像的网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:SXL
    '''
    #----------------------------------------
    #图像大小归一化
    image = cv2.resize(image,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]
    #----------------------------------------

    #-----Gabor滤波--------------------------
    resImg=GaborFeature(image)
    #-----Gabor滤波--------------------------

    #-----对滤波后的图逐个网格化提取特征-------
    feature = np.zeros(64)        # 定义特征向量
    grid_size=4
    imgcount=0
    for img in resImg:
        # 二值化
        retval, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        imgcount+=1
        # pl.figure("dog")
        # pl.imshow(binary)
        # pl.show()

        # 计算网格大小
        grid_h = binary.shape[0] / grid_size
        grid_w = binary.shape[1] / grid_size
        for j in range(grid_size):
            for i in range(grid_size):
                # 统计每个网格中黑点的个数
                grid = binary[int(j * grid_h):int((j + 1) * grid_h), int(i * grid_w):int((i + 1) * grid_w)]
                feature[j * grid_size + i+(imgcount-1)*grid_size*grid_size] = grid[grid == 0].size

    return feature

#可以读取带中文路径的图
def cv_imread(file_path,type=0):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    if(type==0):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img

###########################################################################################################
#
# # 读图
# path = 'E:/sxl_Programs/Python/data/脆/1.jpg'
# image = cv_imread(path,0)
#
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# pl.figure("dog")
# pl.imshow(image)
# pl.show()
# feature=GetImageFeatureGabor(image)
# print(feature)

#--------------批量读图------------------------------------------------------------------------------------
#扫描输出图片列表
def eachFile(filepath):
    list = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list.append(child)
    return list

list=eachFile(r"D:\sxl\处理图片\汉字分类\testImage/")

count1=0
# 读图
ImgFeatures=[]
ImgLabels=[]
for filename in list:
    # image = cv2.imread(filename, 0)
    image = cv_imread(filename,0)
    # 获取特征向量，传入灰度图
    feature = GetImageFeatureGabor(image)
    ImgFeatures.append(feature)
    ImgLabels.append([0,0,1,0,0,0,0,0,0,0])

# print(ImgFeatures)
# print(ImgLabels)
# np.save("ImgFeatures.npy",ImgFeatures)
# np.save("ImgLabels.npy",ImgLabels)