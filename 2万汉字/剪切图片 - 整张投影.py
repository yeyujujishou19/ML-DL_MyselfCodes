# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import re  # 查找字符串   re.finditer(word, path)]
from matplotlib import pyplot as plt
from PIL import Image


# 可以读取带中文路径的图
def cv_imread(file_path, type=0):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    # print(file_path)
    # print(cv_img.shape)
    # print(len(cv_img.shape))
    if (type == 0):
        if (len(cv_img.shape) == 3):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img


#遍历文件夹
list = []
def TraverFolders(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        list.append(path)
        if os.path.isdir(path):
            TraverFolders(path)
    return list


# 切割存图
def MyCutImg(image, int_imgID, lines, baseSavePath, strtemp,imgID):

    # 返回高和宽
    (h, w) = image.shape
    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    a=0
    #---------------------确定分割位置-------------------------------
    #水平投影
    # ---------------------------
    hor = [0 for z in range(0, h)]
    for j in range(0, h):
        for i in range(0, w):
            if binary[j, i] == 0:
                hor[j] += 1

    #---------------------------
    horCutIndexStart = []     #水平开始位置
    horCutIndexEnd   = []     #水平结束位置
    iminWidth=50
    iminVal=60
    istart=0
    iWidthCount=0
    for j in range(0, h):
        if(hor[j] > iminVal):
            iWidthCount+=1
            if(iWidthCount==1):
               istart = j-2
        if(hor[j] < iminVal and iWidthCount>iminWidth):
            horCutIndexStart.append(istart)
            horCutIndexEnd.append(j+2)
            iWidthCount=0
        if(hor[j] < iminVal and iWidthCount<iminWidth):
            istart = 0
            iWidthCount=0

    #垂直投影
    ver = [0 for z in range(0, w)]
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if binary[i, j] == 0:
                ver[j] += 1
    # ---------------------------
    verCutIndexStart = []  # 垂直开始位置
    verCutIndexEnd = []    # 垂直结束位置
    istart=0
    iWidthCount=0
    for j in range(0, w):
        if(ver[j] > iminVal):
            iWidthCount+=1
            if(iWidthCount==1):
               istart = j-2
        if(ver[j] < iminVal and iWidthCount>iminWidth):
            verCutIndexStart.append(istart)
            verCutIndexEnd.append(j+2)
            iWidthCount=0
        if(ver[j] < iminVal and iWidthCount<iminWidth):
            istart = 0
            iWidthCount=0

    # print(len(horCutIndexStart))
    # print(len(horCutIndexEnd))
    # print(len(verCutIndexStart))
    # print(len(verCutIndexEnd))
    #---------------------确定分割位置-------------------------------

    # ---------------------划线-------------------------------
    # for i in range(0, len(horCutIndexStart)):   #逐行开始
    #     cv2.line(image, (0, horCutIndexStart[i]), (w, horCutIndexStart[i]), (0, 0, 0), 2)
    #     cv2.line(image, (0, horCutIndexEnd[i]), (w, horCutIndexEnd[i]), (0, 0, 0), 2)
    #
    # for i in range(0, len(verCutIndexStart)):
    #     cv2.line(image, (verCutIndexStart[i], 0), (verCutIndexStart[i], h), (0, 0, 0), 2)
    #     cv2.line(image, (verCutIndexEnd[i], 0), (verCutIndexEnd[i], h), (0, 0, 0), 2)
    #
    # cv2.imencode('.jpg', image)[1].tofile("D://66.jpg")
    # cv2.imshow("66",image)
    # cv2.waitKey(0)
    # ---------------------划线-------------------------------

    #---------------------开始分割存储-------------------------------
    imgNo=0    #存图序号
    for h in range(0, len(horCutIndexStart)):   #逐行开始
        for w in range(0, len(verCutIndexStart)):
            cutImage= image[int(horCutIndexStart[h]):int(horCutIndexEnd[h]), int(verCutIndexStart[w]):int(verCutIndexEnd[w])]
            imgNo+=1
            strImgName=''   #图像名称
            if(imgID>=1 and imgID<=46):
               strImgName=lines[(imgID-1)*24*19+imgNo-1]   #读取图像名称
            if(imgID>=47 and imgID<=115):
               strImgName=lines[(imgID-47)*16*19+imgNo-1]   #读取图像名称
            if(imgID>=116 and imgID<=161):
               strImgName=lines[(imgID-116)*24*19+imgNo-1]   #读取图像名称
            if(imgID>=162 and imgID<=207):
               strImgName=lines[(imgID-162)*24*19+imgNo-1]   #读取图像名称
            strImgName = strImgName[0:1]  # 图片序号，字符型，去掉\n
            imgname=(r"\%s\%s_%s_%d.jpg" % (strImgName,strtemp,strImgName,imgNo))

            savePath=baseSavePath+imgname
            # 图像大小归一化
            cutImage = cv2.resize(cutImage, (64, 64))
            # cv2.imencode('.jpg', cutImage)[1].tofile(savePath)
            print(savePath)
            # cv2.imwrite(savePath,cutImage)
            a=0
    #---------------------开始分割存储-------------------------------

    return 0


# --------------读取txt-------------------------------------------------------------
f = open("汉字.txt", "r")
lines = f.readlines()  # 读取全部内容
baseSavePath = r"D:\sxl\处理图片\汉字分类\2万汉字\汉字文件夹"
# --------------读取txt-------------------------------------------------------------

# --------------批量读图------------------------------------------------------------------------------------
path = r"D:\sxl\处理图片\汉字分类\不带花纹"
list = TraverFolders(path)
print("共%d个文件" % (len(list)))
countfile = 0
countImg=0
iDisPlay=100
for filename in list:
    countfile += 1
    # -----确定子文件夹名称------------------
    word = r'\\'
    a = [m.start() for m in re.finditer(word, filename)]
    if (len(a) == 5):  # 字文件夹
        strtemp = filename[a[-1] + 1:]  # 文件夹名称-字符名称
        countImg=0   #图片计数归0
    # -----确定子文件夹名称------------------

    # -----子文件夹图片特征提取--------------
    if (len(a) == 6):  # 子文件夹下图片
        if ('.tif' in filename):
            str_imgID = filename[-8:-4]  # 图片序号，字符型
            int_imgID = int(str_imgID)  # 图片序号，转换成整型
            countImg += 1
            if (countImg % 1 == 0):
                print("共%d个文件，正在处理第%d张图片..." % (len(list), countImg))
                print(int_imgID)
            image = cv_imread(filename, 0)

            # -----------处理图像--------------
            MyCutImg(image, int_imgID, lines, baseSavePath,strtemp,int_imgID)
            # -----------处理图像--------------
        else:
            continue

print("运行结束！")
