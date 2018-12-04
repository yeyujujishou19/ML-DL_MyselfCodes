
# -*- coding: utf-8 -*-
import time
import os
import cv2
import numpy as np
import pylab as pl  #画图
import random
import re   #查找字符串   re.finditer(word, path)]

n_label = 1            #标签维度

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
    shuffled_features = features[permutation,:]  #多维
    shuffled_labels = labels[permutation]       #1维
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

def hb(list1,list2):
    result = []
    while list1 and list2:
        if list1[0] < list2[0]:
            result.append(list1[0])
            del list1[0]
        else:
            result.append(list2[0])
            del list2[0]
    if list1:
        result.extend(list1)
    if list2:
        result.extend(list2)
    print(result)
    return result

path=r"D:\sxl\处理图片\汉字分类\train3587"
print("遍历图像中，可能需要花一些时间...")
list=TraverFolders(path)

count1=0
# 读图
ImgFeatures=[]
ImgLabels=[]

time_start=time.time()

strcharName= np.load( r"./npy2/ImgHanZiName3587.npy" )
strcharName=strcharName.tolist()

ImgLabelsTemp=np.zeros(n_label)
print("共%d个文件" % (len(list)))
count=0
SaveNO=0   #存储npy序号
for filename in list:
    count+=1
    if(count%100==0):
       print("共%d个文件，正在处理第%d个..." % (len(list),count))
    #-----确定子文件夹名称------------------
    word = r'\\'
    a = [m.start() for m in re.finditer(word, filename)]
    if(len(a)==5):   #字文件夹
        strtemp=filename[a[-1]+1:]  #文件夹名称-字符名称
        # print(filename)

        iMyindex=strcharName.index(strtemp)  #在文件数组中的位置
        ImgLabelsTemp =iMyindex
        #ImgLabelsTemp=InitImagesLabels(iMyindex)  #创建标签
        # print(iMyindex)
        # print(ImgLabelsTemp)
        # print(ImgLabelsTemp)
    # -----确定子文件夹名称------------------

    # -----子文件夹图片特征提取--------------
    if (len(a) == 6):   #子文件夹下图片
       if('.jpg'in filename):
          image = cv_imread(filename,0)
       else:
           continue
       # 获取特征向量，传入灰度图
       feature = SimpleGridFeature(image)       #简单网格
       #feature = RoughPeriphery(image)                #粗外围
       # feature = GetImageFeatureGabor(image)    #Gabor滤波
       # feature = ComplexGetImageFeature(image)  #综合方法

       ImgFeatures.append(feature)
       ImgLabels.append(ImgLabelsTemp)

       if(count%100000==0):
           ImgFeatures=np.array(ImgFeatures)  #转化成数组，方便拆分
           ImgLabels = np.array(ImgLabels)    #转化成数组，方便拆分
           # 打乱数组
           ImgFeatures, ImgLabels = ShuffledData(ImgFeatures, ImgLabels)
           # 拆分数据为训练集，测试集
           #split_x = int(ImgFeatures.shape[0] * 0.8)
           #train_data, test_data = np.split(ImgFeatures, [split_x], axis=0)  # 拆分特征数据集
           #train_labels, test_labels = np.split(ImgLabels, [split_x], axis=0)  # 拆分标签数据集
           ImgFeatures = ImgFeatures.tolist()   #转化成list，方便append()
           ImgLabels = ImgLabels.tolist()       #转化成list，方便append()

           #保存
           SaveNO+=1
           save_train_data=(r".\npy2\ImgFeatures3587_train_data%d.npy" % (SaveNO))
           save_train_labels= (r".\npy2\ImgLabels3587_train_labels%d.npy" % (SaveNO))
           #save_test_data=(r".\npy\ImgFeatures653_test_data%d.npy" % (SaveNO))
           #save_test_labels= (r".\npy\ImgLabels653_test_labels%d.npy" % (SaveNO))

           np.save(save_train_data, ImgFeatures)
           np.save(save_train_labels, ImgLabels)
           #np.save(save_test_data, test_data)
           #np.save(save_test_labels, test_labels)
           # 清空列表
           ImgFeatures.clear()
           ImgLabels.clear()
           #train_data=[]
           #train_label=[]
           #test_data=[]
           #test_labels=[]
       if(count>len(list)-10):
           print(count)
       if(count==len(list)-1):  #最后一个
           ImgFeatures=np.array(ImgFeatures)  #转化成数组，方便拆分
           ImgLabels = np.array(ImgLabels)    #转化成数组，方便拆分
           # 打乱数组
           ImgFeatures, ImgLabels = ShuffledData(ImgFeatures, ImgLabels)
           # 拆分数据为训练集，测试集
           #split_x = int(ImgFeatures.shape[0] * 0.8)
           #train_data, test_data = np.split(ImgFeatures, [split_x], axis=0)  # 拆分特征数据集
           #train_labels, test_labels = np.split(ImgLabels, [split_x], axis=0)  # 拆分标签数据集
           ImgFeatures = ImgFeatures.tolist()   #转化成list，方便append()
           ImgLabels = ImgLabels.tolist()       #转化成list，方便append()

           #保存
           SaveNO+=1
           save_train_data=(r".\npy2\ImgFeatures3587_train_data%d.npy" % (SaveNO))
           save_train_labels= (r".\npy2\ImgLabels3587_train_labels%d.npy" % (SaveNO))
           #save_test_data=(r".\npy\ImgFeatures653_test_data%d.npy" % (SaveNO))
           #save_test_labels= (r".\npy\ImgLabels653_test_labels%d.npy" % (SaveNO))

           np.save(save_train_data, ImgFeatures)
           np.save(save_train_labels, ImgLabels)
           #np.save(save_test_data, test_data)
           #np.save(save_test_labels, test_labels)
           # 清空列表
           ImgFeatures.clear()
           ImgLabels.clear()
           #train_data=[]
           #rain_label=[]
           #test_data=[]
           #test_labels=[]
       # print(feature)
    # -----子文件夹图片特征提取--------------
time_end=time.time()
time_h=(time_end-time_start)/3600
print('用时：%.6f 小时'% time_h)
print ("第一次存npy,运行结束！")

print ("第一次打乱中...")
#1
###############读取再打乱#####################################################################################
inpyNUM = 18              # npy文件数量
SaveNO=0
for iNo in range(0,int(inpyNUM/2),1):  # npy文件数量
    print("第%d个npy..." % (iNo+1))
    # 读取数据
    train_data_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\ImgFeatures3587_train_data%d.npy" % (iNo + 1))  # npy路径
    train_labels_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\ImgLabels3587_train_labels%d.npy" % (iNo + 1))  # npy路径

    train_data_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\ImgFeatures3587_train_data%d.npy" % (inpyNUM - iNo))  # npy路径
    train_labels_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\ImgLabels3587_train_labels%d.npy" % (inpyNUM - iNo))  # npy路径
    # print("===============================")
    # print("共%d个训练数据集，加载第%d训练数据集中..." % (inpyNUM, iNo + 1))
    train_data1 = np.load(train_data_npyPath1).astype(np.float32)      # 加载数据
    train_labels1 = np.load(train_labels_npyPath1).astype(np.float32)  # 加载数据
    # print("data1:")
    # print(np.array(train_data1).shape)
    # print(np.array(train_labels1).shape)
    train_data2 = np.load(train_data_npyPath2).astype(np.float32)      # 加载数据
    train_labels2 = np.load(train_labels_npyPath2).astype(np.float32)  # 加载数据
    # print("data2:")
    # print(np.array(train_data2).shape)
    # print(np.array(train_labels2).shape)
    #两者合并
    train_data1 = train_data1.tolist()      # 转化成list
    train_labels1 = train_labels1.tolist()  # 转化成list
    train_data2 = train_data2.tolist()      # 转化成list
    train_labels2 = train_labels2.tolist()  # 转化成list
    new_train_data=train_data1+train_data2
    new_train_labels=train_labels1+train_labels2
    # print("data12:")
    # print(np.array(new_train_data).shape)
    # print(np.array(new_train_labels).shape)

    # 打乱数组
    new_train_data,new_train_labels = ShuffledData(np.array(new_train_data), np.array(new_train_labels))

    # 拆分数据为训练集1，训练集2
    split_x = int(new_train_labels.shape[0] * 0.5)
    train_data1, train_data2 = np.split(new_train_data, [split_x], axis=0)  # 拆分特征数据集
    train_labels1, train_labels2 = np.split(new_train_labels, [split_x], axis=0)  # 拆分标签数据集
    # print("train_data1,train_labels1:")
    # print(np.array(train_data1).shape)
    # print(np.array(train_labels1).shape)
    # print(np.array(train_data2).shape)
    # print(np.array(train_labels2).shape)
    # print("===============================")

    # 保存
    SaveNO =SaveNO+ 2
    save_train_data1 = (r".\npy3\ImgFeatures3587_train_data%d.npy" % (SaveNO-1))
    save_train_labels1 = (r".\npy3\ImgLabels3587_train_labels%d.npy" % (SaveNO-1))
    np.save(save_train_data1, train_data1)
    np.save(save_train_labels1, train_labels1)
    save_train_data2 = (r".\npy3\ImgFeatures3587_train_data%d.npy" % (SaveNO))
    save_train_labels2 = (r".\npy3\ImgLabels3587_train_labels%d.npy" % (SaveNO))
    np.save(save_train_data2, train_data2)
    np.save(save_train_labels2, train_labels2)

###############读取再打乱#####################################################################################
#2
###############读取再打乱#####################################################################################
print ("第二次打乱中...")
SaveNO=0
for iNo in range(0,int(inpyNUM/2),1):  # npy文件数量
    print("第%d个npy..." % (iNo+1))
    # 读取数据
    train_data_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy3\ImgFeatures3587_train_data%d.npy" % (iNo + 1))  # npy路径
    train_labels_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy3\ImgLabels653_train_labels%d.npy" % (iNo + 1))  # npy路径

    train_data_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy3\ImgFeatures3587_train_data%d.npy" % (inpyNUM - iNo))  # npy路径
    train_labels_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy3\ImgLabels3587_train_labels%d.npy" % (inpyNUM - iNo))  # npy路径
    # print("===============================")
    # print("共%d个训练数据集，加载第%d训练数据集中..." % (inpyNUM, iNo + 1))
    train_data1 = np.load(train_data_npyPath1).astype(np.float32)      # 加载数据
    train_labels1 = np.load(train_labels_npyPath1).astype(np.float32)  # 加载数据
    # print("data1:")
    # print(np.array(train_data1).shape)
    # print(np.array(train_labels1).shape)
    train_data2 = np.load(train_data_npyPath2).astype(np.float32)      # 加载数据
    train_labels2 = np.load(train_labels_npyPath2).astype(np.float32)  # 加载数据
    # print("data2:")
    # print(np.array(train_data2).shape)
    # print(np.array(train_labels2).shape)
    #两者合并
    train_data1 = train_data1.tolist()      # 转化成list
    train_labels1 = train_labels1.tolist()  # 转化成list
    train_data2 = train_data2.tolist()      # 转化成list
    train_labels2 = train_labels2.tolist()  # 转化成list
    new_train_data=train_data1+train_data2
    new_train_labels=train_labels1+train_labels2
    # print("data12:")
    # print(np.array(new_train_data).shape)
    # print(np.array(new_train_labels).shape)

    # 打乱数组
    new_train_data,new_train_labels = ShuffledData(np.array(new_train_data), np.array(new_train_labels))

    # 拆分数据为训练集1，训练集2
    split_x = int(new_train_labels.shape[0] * 0.5)
    train_data1, train_data2 = np.split(new_train_data, [split_x], axis=0)  # 拆分特征数据集
    train_labels1, train_labels2 = np.split(new_train_labels, [split_x], axis=0)  # 拆分标签数据集
    # print("train_data1,train_labels1:")
    # print(np.array(train_data1).shape)
    # print(np.array(train_labels1).shape)
    # print(np.array(train_data2).shape)
    # print(np.array(train_labels2).shape)
    # print("===============================")

    # 保存
    SaveNO =SaveNO+ 2
    save_train_data1 = (r".\npy4\ImgFeatures3587_train_data%d.npy" % (SaveNO-1))
    save_train_labels1 = (r".\npy4\ImgLabels3587_train_labels%d.npy" % (SaveNO-1))
    np.save(save_train_data1, train_data1)
    np.save(save_train_labels1, train_labels1)
    save_train_data2 = (r".\npy4\ImgFeatures3587_train_data%d.npy" % (SaveNO))
    save_train_labels2 = (r".\npy4\ImgLabels3587_train_labels%d.npy" % (SaveNO))
    np.save(save_train_data2, train_data2)
    np.save(save_train_labels2, train_labels2)

###############读取再打乱#####################################################################################

#3
###############读取再打乱#####################################################################################
print ("第三次打乱中...")
SaveNO=0
for iNo in range(0,int(inpyNUM/2),1):  # npy文件数量
    print("第%d个npy..." % (iNo + 1))
    # 读取数据
    train_data_npyPath1 = (r".\npy4\ImgFeatures3587_train_data%d.npy" % (iNo + 1))  # npy路径
    train_labels_npyPath1 = (r".\npy4\ImgLabels3587_train_labels%d.npy" % (iNo + 1))  # npy路径

    train_data_npyPath2 = (r".\npy4\ImgFeatures3587_train_data%d.npy" % (inpyNUM- iNo))  # npy路径
    train_labels_npyPath2 = (r".\npy4\ImgLabels3587_train_labels%d.npy" % (inpyNUM - iNo))  # npy路径

    # print("共%d个训练数据集，加载第%d训练数据集中..." % (inpyNUM, iNo + 1))
    train_data1 = np.load(train_data_npyPath1).astype(np.float32)      # 加载数据
    train_labels1 = np.load(train_labels_npyPath1).astype(np.float32)  # 加载数据

    train_data2 = np.load(train_data_npyPath2).astype(np.float32)      # 加载数据
    train_labels2 = np.load(train_labels_npyPath2).astype(np.float32)  # 加载数据

    # 两者合并
    train_data1 = train_data1.tolist()      # 转化成list
    train_labels1 = train_labels1.tolist()  # 转化成list
    train_data2 = train_data2.tolist()      # 转化成list
    train_labels2 = train_labels2.tolist()  # 转化成list
    new_train_data=train_data1+train_data2
    new_train_labels=train_labels1+train_labels2
    # print("new_train_data,new_train_labels:")
    # print(np.array(new_train_data).shape)
    # print(np.array(new_train_labels).shape)
    # 打乱数组
    new_train_data, new_train_labels = ShuffledData(np.array(new_train_data), np.array(new_train_labels))

    # 拆分数据为训练集1，训练集2
    split_x = int(new_train_data.shape[0] * 0.5)
    train_data1, train_data2 = np.split(new_train_data, [split_x], axis=0)  # 拆分特征数据集
    train_labels1, train_labels2 = np.split(new_train_labels, [split_x], axis=0)  # 拆分标签数据集
    # print("train_data1,train_labels1:")
    # print(np.array(train_data1).shape)
    # print(np.array(train_labels1).shape)
    #
    # print("train_data2,train_labels2:")
    # print(np.array(train_data2).shape)
    # print(np.array(train_labels2).shape)

    # 拆分数据为训练集1，测试集1
    split_x = int(train_data1.shape[0] * 0.8)
    train_data, test_data = np.split(train_data1, [split_x], axis=0)  # 拆分特征数据集
    train_labels, test_labels = np.split(train_labels1, [split_x], axis=0)  # 拆分标签数据集

    # print("train_data,train_labels:")
    # print(np.array(train_data).shape)
    # print(np.array(train_labels).shape)
    #
    # print("test_data,test_labels:")
    # print(np.array(test_data).shape)
    # print(np.array(test_labels).shape)
    # print("============111======================")
    # 保存
    SaveNO = SaveNO + 2
    save_train_data = (r".\npy5\ImgFeatures3587_train_data%d.npy" % (SaveNO-1))
    save_train_labels = (r".\npy5\ImgLabels3587_train_labels%d.npy" % (SaveNO-1))
    save_test_data=(r".\npy5\ImgFeatures3587_test_data%d.npy" % (SaveNO-1))
    save_test_labels= (r".\npy5\ImgLabels3587_test_labels%d.npy" % (SaveNO-1))

    np.save(save_train_data, train_data)
    np.save(save_train_labels, train_labels)
    np.save(save_test_data, test_data)
    np.save(save_test_labels, test_labels)
    # 清空列表
    train_data = []
    train_label = []
    test_data=[]
    test_labels=[]

    # 拆分数据为训练集2，测试集2
    split_x = int(train_data2.shape[0] * 0.8)
    train_data, test_data = np.split(train_data2, [split_x], axis=0)  # 拆分特征数据集
    train_labels, test_labels = np.split(train_labels2, [split_x], axis=0)  # 拆分标签数据集

    # print("train_data,train_labels:")
    # print(np.array(train_data).shape)
    # print(np.array(train_labels).shape)
    #
    # print("test_data,test_labels:")
    # print(np.array(test_data).shape)
    # print(np.array(test_labels).shape)
    # print("============222======================")
    # 保存
    save_train_data = (r".\npy5\ImgFeatures3587_train_data%d.npy" % (SaveNO))
    save_train_labels = (r".\npy5\ImgLabels3587_train_labels%d.npy" % (SaveNO))
    save_test_data=(r".\npy5\ImgFeatures3587_test_data%d.npy" % (SaveNO))
    save_test_labels= (r".\npy5\ImgLabels3587_test_labels%d.npy" % (SaveNO))

    np.save(save_train_data, train_data)
    np.save(save_train_labels, train_labels)
    np.save(save_test_data, test_data)
    np.save(save_test_labels, test_labels)
    # 清空列表
    train_data = []
    train_label = []
    test_data=[]
    test_labels=[]

###############读取再打乱#####################################################################################
print ("结束！")