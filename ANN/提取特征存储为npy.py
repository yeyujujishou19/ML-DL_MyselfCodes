
# -*- coding: utf-8 -*-
import time
import os
import cv2
import numpy as np
import pylab as pl  #画图
import random
import math
import re   #查找字符串   re.finditer(word, path)]
from time import sleep, ctime
# import threading        # 线程
import multiprocessing  # 进程

n_label = 1             # 标签维度
iDisPlay=1000           # 显示间隔
ithreadNum=1            # 线程数量
inpySize=5000          # 每个npy大小，确保npy数量为偶数！

#---------------------1--------------------------------------------------
#函数功能：综合方法
#函数要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
#返回数据：64*n维特征
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
#叠加两张图片，输入皆是黑白图，img1是底层图片，img2是上层图片，返回叠加后的图片
def ImageOverlay(img1,img2):
    # 把logo放在左上角，所以我们只关心这一块区域
    h = img1.shape[0]
    w = img1.shape[1]
    rows = img2.shape[0]
    cols = img2.shape[1]
    roi = img1[int((h - rows) / 2):rows + int((h - rows) / 2), int((w - cols) / 2):cols + int((w - cols) / 2)]
    # 创建掩膜
    img2gray = img2.copy()
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    dst = cv2.add(img1_bg, img2)  # 进行融合
    img1[int((h - rows) / 2):rows + int((h - rows) / 2),int((w - cols) / 2):cols + int((w - cols) / 2)] = dst  # 融合后放在原图上
    return img1

#函数功能：处理白边
#找到上下左右的白边位置
#剪切掉白边
#二值化
#将图像放到64*64的白底图像中心
def HandWhiteEdges(img):
    ret, thresh1 = cv2.threshold(img, 249, 255, cv2.THRESH_BINARY)
    # OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 膨胀图像
    thresh1 = cv2.dilate(thresh1, kernel)
    row= img.shape[0]
    col = img.shape[1]
    tempr0 = 0    #横上
    tempr1 = 0    #横下
    tempc0 = 0    #竖左
    tempc1 = 0    #竖右
    # 765 是255+255+255,如果是黑色背景就是0+0+0，彩色的背景，将765替换成其他颜色的RGB之和，这个会有一点问题，因为三个和相同但颜色不一定同
    for r in range(0, row):
        if thresh1.sum(axis=1)[r] != 255 * col:
            tempr0 = r
            break

    for r in range(row - 1, 0, -1):
        if thresh1.sum(axis=1)[r] != 255 * col:
            tempr1 = r
            break

    for c in range(0, col):
        if thresh1.sum(axis=0)[c] != 255 * row:
            tempc0 = c
            break

    for c in range(col - 1, 0, -1):
        if thresh1.sum(axis=0)[c] != 255 * row:
            tempc1 = c
            break

    # 创建全白图片
    imageTemp = np.zeros((64, 64, 3), dtype=np.uint8)
    imageTemp = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2GRAY)
    imageTemp.fill(255)

    if(tempr1-tempr0==0 or tempc1-tempc0==0):   #空图
        return imageTemp    #返回全白图

    new_img = img[tempr0:tempr1, tempc0:tempc1]
    #二值化
    retval,binary = cv2.threshold(new_img,0,255,cv2.THRESH_OTSU)

    #叠加两幅图像
    rstImg=ImageOverlay(imageTemp, binary)
    return rstImg

#函数功能：简单网格
#函数要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
#返回数据：64*1维特征
def SimpleGridFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:RenHui
    '''
    new_img=HandWhiteEdges(image)  #白边处理
    #new_img=image
    #图像大小归一化
    image = cv2.resize(new_img,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)

    #定义特征向量
    grid_size1 = 16
    grid_size2 = 8
    grid_size3 = 4
    feature = np.zeros(grid_size1*grid_size1+grid_size2*grid_size2+grid_size3*grid_size3)

    #计算网格大小1
    grid_h1 = binary.shape[0]/grid_size1
    grid_w1 = binary.shape[1]/grid_size1
    for j in range(grid_size1):
        for i in range(grid_size1):
            grid = binary[int(j*grid_h1):int((j+1)*grid_h1),int(i*grid_w1):int((i+1)*grid_w1)]
            feature[j*grid_size1+i] = grid[grid==0].size

    #计算网格大小2
    grid_h2 = binary.shape[0]/grid_size2
    grid_w2 = binary.shape[1]/grid_size2
    for j in range(grid_size2):
        for i in range(grid_size2):
            grid = binary[int(j*grid_h2):int((j+1)*grid_h2),int(i*grid_w2):int((i+1)*grid_w2)]
            feature[grid_size1*grid_size1+j*grid_size2+i] = grid[grid==0].size

    #计算网格大小3
    grid_h3 = binary.shape[0]/grid_size3
    grid_w3 = binary.shape[1]/grid_size3
    for j in range(grid_size3):
        for i in range(grid_size3):
            grid = binary[int(j*grid_h3):int((j+1)*grid_h3),int(i*grid_w3):int((i+1)*grid_w3)]
            feature[grid_size1*grid_size1+grid_size2*grid_size2+j*grid_size3+i] = grid[grid==0].size

    return feature


#---------------------3--------------------------------------------------
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

#函数功能：Gabor滤波
#函数要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
#返回数据：64*1维特征
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

#---------------------4--------------------------------------------------
#函数功能：粗外围特征
#函数要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
#返回数据：64*1维粗外围特征，其中一次粗外围特征32维，二次粗外围特征32维
def RoughPeriphery(image):
    #----------------------------------------
    #图像大小归一化
    image = cv2.resize(image,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]
    #----------------------------------------
    # 二值化
    retval, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    #计算网格大小
    grid_size=8
    #定义特征向量
    feature = np.zeros(grid_size*grid_size)
    iD=0
    # 从左到右  #从右到左  0-31
    for h in range(grid_size - 1, img_h-grid_size, grid_size - 1):
        iD += 4
        iLetfToRight = 0
        iRightToLetf = 0
        for w in range(img_w):             # 从左到右
            if (binary[h, w] == 0 and iLetfToRight==0):  #一次粗外围
                iLetfToRight+=1
                feature[iD - 4] = w
                break
            if (binary[h, w] == 0 and iLetfToRight==1):  #二次粗外围
                iLetfToRight+=1
                feature[iD - 3] = w
                break
        for w in range(img_w - 1, 0, -1):  # 从右到左
            if (binary[h, w] == 0 and iRightToLetf==0):  #一次粗外围
                iRightToLetf+=1
                feature[iD - 2] = w
                break
            if (binary[h, w] == 0 and iRightToLetf==1):  #二次粗外围
                iRightToLetf+=1
                feature[iD - 1] = w
                break
    # 从上到下   #从下到上  32-63
    for w in range(grid_size - 1, img_w-grid_size, grid_size - 1):
        iD += 4
        iUpToDown = 0
        iDownToUp = 0
        for h in range(img_h):  # 从上到下
            if (binary[h, w] == 0 and  iUpToDown==0):  #一次粗外围
                iUpToDown+=1
                feature[iD - 4] = h
                break
            if (binary[h, w] == 0 and  iUpToDown==1):  #二次粗外围
                iUpToDown+=1
                feature[iD - 3] = h
                break
        for h in range(img_h - 1, 0, -1):  # 从下到上
            if (binary[h, w] == 0 and iDownToUp==0):  #一次粗外围
                iDownToUp+=1
                feature[iD - 2] = h
                break
            if (binary[h, w] == 0 and iDownToUp==1):  #二次粗外围
                iDownToUp+=1
                feature[iD - 1] = h
                break
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
    aa=np.zeros(n_label)
    aa[index]=1
    return aa

#遍历文件夹
list = []
def TraverFolders(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        list.append(path)
        #----在此处直接处理----

        # ----在此处直接处理----

        if os.path.isdir(path):
            TraverFolders(path)
    return list

#扫描输出图片列表
def eachFile(filepath):
    list = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list.append(child)
    return list

#多个数组按同一规则打乱数据
def ShuffledData(features,labels):
    '''
    @description:随机打乱数据与标签，但保持数据与标签一一对应
    @author:RenHui
    '''
    permutation = np.random.permutation(features.shape[0])
    shuffled_features = features[permutation,:]  #多维
    shuffled_labels = labels[permutation]       #1维
    return shuffled_features,shuffled_labels

#随机移动图像，黑白图
def randomMoveImage(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    # 0 上，1 下，2 左，3 右
    idirection=random.randrange(0, 4) #随机产生0,1,2,3
    #随机移动距离
    iPixsNum=random.randrange(1, 3) #随机产生1,2

    if (idirection == 0): #上
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, 0], [0, 1, -iPixsNum]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for h in range(iPixsNum):              # 从上到下
            for w in range(img_w):             # 从左到右
                dst[img_h-h-1, w] = 255

    if (idirection == 1): #下
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, 0], [0, 1, iPixsNum]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for h in range(iPixsNum):              # 从上到下
            for w in range(img_w):             # 从左到右
                dst[h, w] = 255

    if (idirection == 2): #左
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, -iPixsNum], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for w in range(iPixsNum):  # 从左到右
            for h in range(img_h):  # 从上到下
                dst[h, img_w - w - 1] = 255

    if (idirection == 3): #右
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, iPixsNum], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for w in range(iPixsNum):  # 从左到右
            for h in range(img_h):  # 从上到下
                dst[h, w] = 255
    return dst

#综合图像处理
def complexProcImg(list,threadID):

    npybaseID=math.ceil( float( (len(list)/ithreadNum) *5 /inpySize) )  #每个线程会存多少个npy

    # 读图
    ImgFeatures = []
    ImgLabels = []

    ImgLabelsTemp = np.zeros(n_label)

    count = 0  # 文件计数
    countImg = 0  # 图片计数
    SaveNO = 0+npybaseID*threadID  # 存储npy序号

    for filename in list[threadID:-1:ithreadNum]:
        count += 1
        # -----确定子文件夹名称------------------
        word = r'\\'
        a = [m.start() for m in re.finditer(word, filename)]
        if (len(a) == 3):  # 字文件夹
            strtemp = filename[a[-1] + 1:]  # 文件夹名称-字符名称
            # print(filename)
        # -----确定子文件夹名称------------------

        # -----子文件夹图片特征提取--------------
        if (len(a) == 4):  # 子文件夹下图片
            if ('.jpg' in filename):
                countImg += 1
                if (countImg % iDisPlay == 0):
                    print("线程%d, 共%d个文件，正在处理第%d张图片..." % (threadID, len(list), countImg))
                image = cv_imread(filename, 0)

                # 获取特征向量，传入灰度图
                feature = SimpleGridFeature(image)  # 简单网格
                ImgFeatures.append(feature)
                ImgLabels.append(ImgLabelsTemp)

                # 随机移动4次
                for itime in range(4):
                    rMovedImage = randomMoveImage(image)
                    feature = SimpleGridFeature(rMovedImage)  # 简单网格
                    ImgFeatures.append(feature)
                    ImgLabels.append(ImgLabelsTemp)
                    countImg += 1

                if (countImg % inpySize == 0):
                    ImgFeatures = np.array(ImgFeatures)  # 转化成数组，方便拆分
                    ImgLabels = np.array(ImgLabels)  # 转化成数组，方便拆分
                    # 打乱数组
                    ImgFeatures, ImgLabels = ShuffledData(ImgFeatures, ImgLabels)
                    ImgFeatures = ImgFeatures.tolist()  # 转化成list，方便append()
                    ImgLabels = ImgLabels.tolist()  # 转化成list，方便append()

                    # 保存
                    SaveNO += 1
                    save_train_data = (r".\npy2\ImgFeatures21477_train_data_temp%d.npy" % (SaveNO))
                    save_train_labels = (r".\npy2\ImgLabels21477_train_labels_temp%d.npy" % (SaveNO))
                    print("进程%d,SaveNO：%d" % (threadID,SaveNO))
                    np.save(save_train_data, ImgFeatures)
                    np.save(save_train_labels, ImgLabels)

                    # 清空列表
                    ImgFeatures = []
                    ImgLabels = []
            else:
                continue

    #最后一组
    ImgFeatures = np.array(ImgFeatures)  # 转化成数组，方便拆分
    ImgLabels = np.array(ImgLabels)  # 转化成数组，方便拆分
    # 打乱数组
    ImgFeatures, ImgLabels = ShuffledData(ImgFeatures, ImgLabels)
    ImgFeatures = ImgFeatures.tolist()  # 转化成list，方便append()
    ImgLabels = ImgLabels.tolist()  # 转化成list，方便append()

    # 保存
    SaveNO += 1
    save_train_data = (r".\npy2\ImgFeatures21477_train_data_temp%d.npy" % (SaveNO))
    save_train_labels = (r".\npy2\ImgLabels21477_train_labels_temp%d.npy" % (SaveNO))

    print("进程%d,last SaveNO：%d" % (threadID,SaveNO))

    np.save(save_train_data, ImgFeatures)
    np.save(save_train_labels, ImgLabels)

    # 清空列表
    ImgFeatures = []
    ImgLabels = []

###########################################################################################################

#===================================================================================
#===================================================================================
if __name__ == "__main__":
    time_start=time.time()                         #开始计时
    if(os.path.exists("./npy2/List_21477.npy")):
        list = np.load("./npy2/List_21477.npy")
    else:
        path=r"E:\2万汉字分类\train21477"     #文件夹路径
        # path=r"D:\sxl\处理图片\汉字分类\train85"     #文件夹路径
        print("遍历图像中，可能需要花一些时间...")
        list=TraverFolders(path)
        np.save("./npy2/List_21477.npy", list)

    print("共%d个文件" % (len(list)*5))
    inpyNUM = int(math.ceil(len(list)*5/inpySize))  # npy文件数量  4=math.ceil(3.75)向上取整
    print("inpyNUM:",inpyNUM)

    strcharName= np.load( r"./npy2/ImgHanZiName21477.npy" )  #读取文件夹名称列表
    strcharName=strcharName.tolist()

    # 创建进程
    Process=[]
    print("创建进程...")
    for i in range(ithreadNum):
        p = multiprocessing.Process(target = complexProcImg, args = (list, i))
        Process.append(p)

    # 启动进程
    print("启动进程...")
    for i in range(ithreadNum):
        Process[i].start()

    for i in range(ithreadNum):
        Process[i].join()

    time_end=time.time()
    time_h=(time_end-time_start)/3600
    print('用时：%.6f 小时'% time_h)
    print ("读图提取特征存npy,运行结束！")


