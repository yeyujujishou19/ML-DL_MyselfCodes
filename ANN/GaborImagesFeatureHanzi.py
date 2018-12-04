
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pylab as pl  #画图
import random
import re   #查找字符串   re.finditer(word, path)]


#---------------------1--------------------------------------------------
#字符图像的特征提取方法
#要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
def ComplexGetImageFeature(image):
    '''
    @description:综合多种方法提取的特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:SXL
    '''
    feature = []
    feature1=GetImageFeatureGabor(image)
    feature2=SimpleGridFeature(image)
    for i in range(len(feature1)):
        feature.append(feature1[i])
    for i in range(len(feature1)):
        feature.append(feature2[i])
    return feature

#---------------------2--------------------------------------------------
#剪切白边
def CutWhiteEdge(img):
    #----剪切白边代码---

    return img

#字符图像的特征提取方法
#要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
def SimpleGridFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:RenHui
    '''
    new_img=CutWhiteEdge(image)

    #图像大小归一化
    image = cv2.resize(new_img,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    #计算网格大小
    grid_size=8
    grid_h = binary.shape[0]/grid_size
    grid_w = binary.shape[1]/grid_size
    #定义特征向量
    feature = np.zeros(grid_size*grid_size)
    for j in range(grid_size):
        for i in range(grid_size):
            grid = binary[int(j*grid_h):int((j+1)*grid_h),int(i*grid_w):int((i+1)*grid_w)]
            feature[j*grid_size+i] = grid[grid==0].size
    return feature
###########################################################################################################
def BuildGaborKernels(ksize = 5,lamda = 1.5,sigma = 1.0):
    '''
    @description:生成多尺度，多方向的gabor特征
    @参数参考opencv
    @return:多个gabor卷积核所组成的
    @author:RenHui
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
    @return: 四个方向滤波后的图
    @author:RenHui
    '''
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    kernels = BuildGaborKernels(ksize = 7,lamda = 8,sigma = 4)
    dst_imgs = []
    for kernel in kernels:
        img = np.zeros_like(image)
        tmp = cv2.filter2D(image,cv2.CV_8UC3,kernel)
        img = np.maximum(img,tmp,img)
        dst_imgs.append(img)

    # pl.figure(2)
    # for temp in range(len(dst_imgs)):
    #     pl.subplot(4,1,temp+1) #第一个4为4个方向，第二个4为4个尺寸
    #     pl.imshow(dst_imgs[temp], cmap='gray' )
    # pl.show()
    return dst_imgs

#---------------------3--------------------------------------------------
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
    resImg = GaborFeature(image)
    #-----Gabor滤波--------------------------

    #-----对滤波后的图逐个网格化提取特征-------
    grid_size=4
    feature = np.zeros(grid_size*grid_size*4)  # 定义特征向量
    imgcount=0
    for img in resImg:
        # 二值化
        retval, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        imgcount+=1
        # pl.figure("binary")
        # pl.imshow(binary)
        # pl.show()

        # 统计二值图中黑点的个数
        Globalgrid = binary[int(0):int(img_h), int(0):int(img_w)]
        iGlobalBlackNum = Globalgrid[Globalgrid == 0].size
        if(iGlobalBlackNum==0):
            iGlobalBlackNum=1
        # 计算网格大小
        grid_h = binary.shape[0] / grid_size
        grid_w = binary.shape[1] / grid_size
        for j in range(grid_size):
            for i in range(grid_size):
                # 统计每个网格中黑点的个数
                grid = binary[int(j * grid_h):int((j + 1) * grid_h), int(i * grid_w):int((i + 1) * grid_w)]
                #当前网格黑点占整个图中黑点的比重
                feature[j * grid_size + i+(imgcount-1)*grid_size*grid_size] = grid[grid == 0].size
    return feature

#可以读取带中文路径的图
def cv_imread(file_path,type=0):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    # print(file_path)
    # print(cv_img.shape)
    # print(len(cv_img.shape))
    if(type==0):
        if(len(cv_img.shape)==3):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img

#输入index范围0-84
def InitImagesLabels(index):
    aa=np.zeros(85)
    aa[index]=1
    return aa

#遍历文件夹
list = []
def TraverFolders(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        list.append(path)
        if os.path.isdir(path):
            TraverFolders(path)
    return list

#多个数组按同一规则打乱数据
def ShuffledData(features,labels):
    '''
    @description:随机打乱数据与标签，但保持数据与标签一一对应
    @author:RenHui
    '''
    permutation = np.random.permutation(features.shape[0])
    shuffled_features = features[permutation,:]
    shuffled_labels = labels[permutation,:]
    return shuffled_features,shuffled_labels

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

path=r"D:\sxl\处理图片\汉字分类\汉字"
list=TraverFolders(path)

count1=0
# 读图
ImgFeatures=[]
ImgLabels=[]
# strcharName=['0','1','2','3','4','5','6','7','8','9','A','a_小','B','b_小','C','c_小','D','d_小',
#              'E','e_小','F','f_小','G','g_小','H','h_小','I','i_小','J','j_小','K','k_小','L','l_小',
#              'M','m_小','N','n_小','O','o_小','P','p_小','Q','q_小','R','r_小','S','s_小','T','t_小',
#              'U', 'u_小', 'V', 'v_小', 'W', 'w_小', 'X', 'x_小', 'Y', 'y_小', 'Z', 'z_小']

# strcharName=['嫒','扳','板','苯','笨','币','厕','侧','厂','憧','脆','打','大','钉','丢','懂',
#              '儿','非','干','古','诂','诡','乎','话','几','己','减','減','韭','钜','决','決',
#              '扛','钌','昧','孟','末','沐','钼','蓬','篷','平','亓','千','去','犬','壬','晌',
#              '舌','十','士','市','拭','栻','適','沭','耍','巳','趟','天','土','王','枉','未',
#              '味','文','兀','淅','晰','响','写','要','已','盂','与','元','媛','丈','趙','谪',
#              '锺','鍾','柱','状','狀']
strcharName= np.load( "ImgHanZiName653.npy" )
ImgLabelsTemp=np.zeros(653)
print("共%d个文件" % (len(list)))
count=0
for filename in list:
    count+=1
    if(count%100==0):
       print("共%d个文件，正在执行第%d个..." % (len(list),count))
    #-----确定子文件夹名称------------------
    word = r'\\'
    a = [m.start() for m in re.finditer(word, filename)]
    if(len(a)==5):   #字文件夹
        strtemp=filename[a[-1]+1:]  #文件夹名称-字符名称
        index=strcharName.index(strtemp)  #在文件数组中的位置
        ImgLabelsTemp=InitImagesLabels(index)  #创建标签
        # print(ImgLabelsTemp)
    # -----确定子文件夹名称------------------

    # -----子文件夹图片特征提取--------------
    if (len(a) == 6):   #子文件夹下图片
       image = cv_imread(filename,0)
       # 获取特征向量，传入灰度图
       feature = SimpleGridFeature(image)
       # feature = GetImageFeatureGabor(image)
       # feature = ComplexGetImageFeature(image)
       ImgFeatures.append(feature)
       ImgLabels.append(ImgLabelsTemp)
       # print(feature)
    # -----子文件夹图片特征提取--------------

#打乱数组
ImgFeatures,ImgLabels=ShuffledData(np.array(ImgFeatures),np.array(ImgLabels))
# print(ImgFeatures)
# print(ImgLabels)
np.save("ImgFeaturesHanZi.npy",ImgFeatures)
np.save("ImgLabelsHanZi.npy",ImgLabels)

print ("运行结束！")