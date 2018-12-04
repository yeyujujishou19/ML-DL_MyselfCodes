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


# --------------批量读图------------------------------------------------------------------------------------
path = r"D:\sxl\处理图片\汉字分类\新建文件夹"
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
                print("共%d个文件，正在处理第%d张图片..." % (len(list), int_imgID))
                print("正在处理第图片序列号%d..." % (int_imgID))
            image = cv_imread(filename, 1)

            # -----------处理图像--------------
            filename = filename.replace("D:\sxl\处理图片\汉字分类\新建文件夹\桔黄", "D:\sxl\处理图片\汉字分类\桔黄")
            if(int_imgID>70):  #70以后的图像序号加1
                newint_imgID=int_imgID+1
                strnewint_imgID=str(newint_imgID)
                strnewint_imgID = strnewint_imgID.zfill(4)
                filename=filename.replace(str_imgID, strnewint_imgID)

            cv2.imencode('.tif', image)[1].tofile(filename)
            # -----------处理图像--------------
        else:
            continue

print("运行结束！")