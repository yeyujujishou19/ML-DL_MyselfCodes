
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
ishuffledtimes=3                              #打乱次数

inpyNUM = 40                                  #npy文件数量

###############读取再打乱#####################################################################################


time_start=time.time()   #开始计时
for itime in range(ishuffledtimes):
    print("第%d次打乱中..." % (itime+1))

    SaveNO = 0
    for iNo in range(0, int(inpyNUM / 2), 1):  # npy文件数量
        print("1,第%d个npy..." % (iNo + 1))

        # 读取数据
        train_data_npyPath1 = ( r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_%d.npy" % (iNo + 1))         # npy路径
        train_labels_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_labels_train_%d.npy" % (iNo + 1))        # npy路径
        train_data_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_%d.npy" % (inpyNUM - iNo))    # npy路径
        train_labels_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_labels_train_%d.npy" % (inpyNUM - iNo))  # npy路径
        train_data1 = np.load(train_data_npyPath1).astype(np.float32)      # 加载数据1
        train_labels1 = np.load(train_labels_npyPath1).astype(np.float32)  # 加载标签1
        train_data2 = np.load(train_data_npyPath2).astype(np.float32)      # 加载数据2
        train_labels2 = np.load(train_labels_npyPath2).astype(np.float32)  # 加载标签2

        # 两者合并
        train_data1 = train_data1.tolist()                # 数据1转化成list
        train_labels1 = train_labels1.tolist()            # 标签1转化成list
        train_data2 = train_data2.tolist()                # 数据2转化成list
        train_labels2 = train_labels2.tolist()            # 标签2转化成list
        new_train_data = train_data1 + train_data2        # 数据1和数据2合并
        new_train_labels = train_labels1 + train_labels2  # 标签1和标签2合并

        # 打乱数组
        new_train_data, new_train_labels = ShuffledData(np.array(new_train_data), np.array(new_train_labels))

        # 拆分数据为训练集1，训练集2
        split_x = int(new_train_labels.shape[0] * 0.5)
        train_data1, train_data2 = np.split(new_train_data, [split_x], axis=0)        # 拆分特征数据集
        train_labels1, train_labels2 = np.split(new_train_labels, [split_x], axis=0)  # 拆分标签数据集

        # 保存
        SaveNO = SaveNO + 2
        save_train_data1 = (r".\npy2\Img21477_features_train_temp_%d.npy" % (SaveNO - 1))
        save_train_labels1 = (r".\npy2\Img21477_labels_train_temp_%d.npy" % (SaveNO - 1))
        np.save(save_train_data1, train_data1)
        np.save(save_train_labels1, train_labels1)
        save_train_data2 = (r".\npy2\Img21477_features_train_temp_%d.npy" % (SaveNO))
        save_train_labels2 = (r".\npy2\Img21477_labels_train_temp_%d.npy" % (SaveNO))
        np.save(save_train_data2, train_data2)
        np.save(save_train_labels2, train_labels2)
    ##-------------------------------------------------------------------------------------------------------
    SaveNO = 0
    for iNo in range(0, int(inpyNUM / 2), 1):  # npy文件数量
        print("2,第%d个npy..." % (iNo + 1))

        # 读取数据
        train_data_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_temp_%d.npy" % (iNo + 1))          # npy路径
        train_labels_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_labels_train_temp_%d.npy" % (iNo + 1))        # npy路径
        train_data_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_temp_%d.npy" % (inpyNUM - iNo))    # npy路径
        train_labels_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_labels_train_temp_%d.npy" % (inpyNUM - iNo))  # npy路径
        train_data1 = np.load(train_data_npyPath1).astype(np.float32)      # 加载数据1
        train_labels1 = np.load(train_labels_npyPath1).astype(np.float32)  # 加载标签1
        train_data2 = np.load(train_data_npyPath2).astype(np.float32)      # 加载数据2
        train_labels2 = np.load(train_labels_npyPath2).astype(np.float32)  # 加载标签2

        # 两者合并
        train_data1 = train_data1.tolist()                # 数据1转化成list
        train_labels1 = train_labels1.tolist()            # 标签1转化成list
        train_data2 = train_data2.tolist()                # 数据2转化成list
        train_labels2 = train_labels2.tolist()            # 标签2转化成list
        new_train_data = train_data1 + train_data2        # 数据1和数据2合并
        new_train_labels = train_labels1 + train_labels2  # 标签1和标签2合并

        # 打乱数组
        new_train_data, new_train_labels = ShuffledData(np.array(new_train_data), np.array(new_train_labels))

        # 拆分数据为训练集1，训练集2
        split_x = int(new_train_labels.shape[0] * 0.5)
        train_data1, train_data2 = np.split(new_train_data, [split_x], axis=0)        # 拆分特征数据集
        train_labels1, train_labels2 = np.split(new_train_labels, [split_x], axis=0)  # 拆分标签数据集

        # 保存
        SaveNO = SaveNO + 2
        save_train_data1 = (r".\npy2\Img21477_features_train_%d.npy" % (SaveNO - 1))
        save_train_labels1 = (r".\npy2\Img21477_labels_train_%d.npy" % (SaveNO - 1))
        np.save(save_train_data1, train_data1)
        np.save(save_train_labels1, train_labels1)
        save_train_data2 = (r".\npy2\Img21477_features_train_%d.npy" % (SaveNO))
        save_train_labels2 = (r".\npy2\Img21477_labels_train_%d.npy" % (SaveNO))
        np.save(save_train_data2, train_data2)
        np.save(save_train_labels2, train_labels2)
###############读取再打乱#####################################################################################

###############读取再打乱#####################################################################################
print ("最后一次打乱中...")
SaveNO=0
for iNo in range(0,int(inpyNUM/2),1):  # npy文件数量
    print("第%d个npy..." % (iNo + 1))
    # 读取数据
    # 读取数据
    train_data_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_%d.npy" % (iNo + 1))  # npy路径
    train_labels_npyPath1 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_labels_train_%d.npy" % (iNo + 1))  # npy路径
    train_data_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_%d.npy" % (inpyNUM - iNo))  # npy路径
    train_labels_npyPath2 = (r"E:\sxl_Programs\Python\ANN\npy2\Img21477_labels_train_%d.npy" % (inpyNUM - iNo))  # npy路径

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

    # 保存
    SaveNO = SaveNO + 2
    save_train_data1 = (r".\npy2\Img21477_features_train_%d.npy" % (SaveNO - 1))
    save_train_labels1 = (r".\npy2\Img21477_labels_train_%d.npy" % (SaveNO - 1))
    np.save(save_train_data1, train_data1)
    np.save(save_train_labels1, train_labels1)
    save_train_data2 = (r".\npy2\Img21477_features_train_%d.npy" % (SaveNO))
    save_train_labels2 = (r".\npy2\Img21477_labels_train_%d.npy" % (SaveNO))
    np.save(save_train_data2, train_data2)
    np.save(save_train_labels2, train_labels2)


    # # 拆分数据为训练集1，训练集2
    # split_x = int(new_train_data.shape[0] * 0.5)
    # train_data1, train_data2 = np.split(new_train_data, [split_x], axis=0)        # 拆分特征数据集
    # train_labels1, train_labels2 = np.split(new_train_labels, [split_x], axis=0)  # 拆分标签数据集
    #
    # # 拆分数据为训练集1，测试集1
    # split_x = int(train_data1.shape[0] * 0.8)
    # train_data, test_data = np.split(train_data1, [split_x], axis=0)        # 拆分特征数据集
    # train_labels, test_labels = np.split(train_labels1, [split_x], axis=0)  # 拆分标签数据集
    #
    # # 保存
    # SaveNO = SaveNO + 2
    # save_train_data = (r".\npy2\ImgFeatures3527_train_data%d.npy" % (SaveNO-1))
    # save_train_labels = (r".\npy2\ImgLabels3527_train_labels%d.npy" % (SaveNO-1))
    # save_test_data=(r".\npy2\ImgFeatures3527_test_data%d.npy" % (SaveNO-1))
    # save_test_labels= (r".\npy2\ImgLabels3527_test_labels%d.npy" % (SaveNO-1))
    #
    # np.save(save_train_data, train_data)
    # np.save(save_train_labels, train_labels)
    # np.save(save_test_data, test_data)
    # np.save(save_test_labels, test_labels)
    # # 清空列表
    # train_data = []
    # train_label = []
    # test_data=[]
    # test_labels=[]
    #
    # # 拆分数据为训练集2，测试集2
    # split_x = int(train_data2.shape[0] * 0.8)
    # train_data, test_data = np.split(train_data2, [split_x], axis=0)        # 拆分特征数据集
    # train_labels, test_labels = np.split(train_labels2, [split_x], axis=0)  # 拆分标签数据集
    #
    # # 保存
    # save_train_data = (r".\npy2\ImgFeatures3527_train_data%d.npy" % (SaveNO))
    # save_train_labels = (r".\npy2\ImgLabels3527_train_labels%d.npy" % (SaveNO))
    # save_test_data=(r".\npy2\ImgFeatures3527_test_data%d.npy" % (SaveNO))
    # save_test_labels= (r".\npy2\ImgLabels3527_test_labels%d.npy" % (SaveNO))
    #
    # np.save(save_train_data, train_data)
    # np.save(save_train_labels, train_labels)
    # np.save(save_test_data, test_data)
    # np.save(save_test_labels, test_labels)
    # # 清空列表
    # train_data = []
    # train_label = []
    # test_data = []
    # test_labels = []

###############读取再打乱#####################################################################################

#--------------清空临时文件-----------------------------------
for id in range(inpyNUM):
    path1=(r".\npy2\ImgFeatures3527_train_data_temp%d.npy" % (id+1))
    os.remove(path1)
    path2=(r".\npy2\ImgLabels3527_train_labels_temp%d.npy" % (id+1))
    os.remove(path2)
#--------------清空临时文件-----------------------------------
time_end=time.time()
time_h=(time_end-time_start)/3600
print('二次用时：%.6f 小时'% time_h)
print ("结束！")