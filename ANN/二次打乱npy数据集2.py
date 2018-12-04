
# -*- coding: utf-8 -*-
import time
import os
import cv2
import numpy as np
import pylab as pl  #画图
import random
import math
import re   #查找字符串   re.finditer(word, path)]

n_label = 1            #标签维度

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

#===================================================================================
#===================================================================================
ishuffledtimes=1                              #打乱次数

inpyNUM = 2                                  #npy文件数量

###############读取再打乱#####################################################################################

new_train_data=[]
time_start=time.time()   #开始计时
for itime in range(ishuffledtimes):
    print("第%d次打乱中..." % (itime+1))

    SaveNO = 0

    for iNo in range(0, int(inpyNUM), 1):  # npy文件数量
        print("1,第%d个npy..." % (iNo + 1))

        # 读取数据
        train_data_npyPath1 = ( r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_%d.npy" % (iNo + 1))         # npy路径
        train_labels_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_labels_train_%d.npy" % (iNo + 1))        # npy路径
        train_data1 = np.load(train_data_npyPath1).astype(np.int16)      # 加载数据1
        train_labels1 = np.load(train_labels_npyPath1).astype(np.int16)  # 加载标签1
        print(type(train_data1))
        train_data1=train_data1.tolist()
        print(type(train_data1))

        print("train_data1:%d"%(len(train_data1)))
        new_train_data=new_train_data+train_data1
        train_data1=[]
        print("new_train_data:%d" % (len(new_train_data)))
        # 打乱数组
        # new_train_data, new_train_labels = ShuffledData(np.array(new_train_data), np.array(new_train_labels))

