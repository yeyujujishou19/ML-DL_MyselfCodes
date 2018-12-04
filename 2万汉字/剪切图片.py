# -*- coding: utf-8 -*-
import os
import cv2
import time
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

#倾斜校正
def copybook_correction(image):
    '''
    @description:字帖图像倾斜校正
    @image:字帖原图像
    @return:rotate_img 校正后的图片, angle 倾斜角度
    @author:RenHui
    '''
    edge_x = 50
    edge_y = 80
    # 设置ROI 截掉可能存在的边框
    image_ROI = image[edge_y:-edge_y, edge_x:-edge_x]

    # 灰度 二值 闭操作
    gary = cv2.cvtColor(image_ROI, cv2.COLOR_BGR2GRAY)
    retval, binary = cv2.threshold(gary, 0, 255, cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 将页码区域置白
    cv2.rectangle(closing, (1000, closing.shape[0] - 100), (1350, closing.shape[0]), 255, -1)

    # 取反
    closing = 255 - closing

    # 提取轮廓 合并 计算最小外接矩形
    _, contours, _ = cv2.findContours(closing,
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    max_contours = np.concatenate(contours)
    min_rect = cv2.minAreaRect(max_contours)

    # 计算在原图的中心点，角度
    center_point = (min_rect[0][0] + edge_x, min_rect[0][1] + edge_y)
    angle = min_rect[2]
    if angle < -45:
        angle += 90

    # 计算放射矩阵，旋转图像
    rotate_mat = cv2.getRotationMatrix2D(center_point, angle, 1)
    rotate_img = cv2.warpAffine(image, rotate_mat, (image.shape[1], image.shape[0]))

    return rotate_img, angle


# 切割存图
def MyCutImg(image,lines, baseSavePath, strtemp,imgID):
    # 灰度 二值 闭操作
    gary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 返回高和宽
    (img_h, img_w) = gary.shape

    #二值化
    retval,binary = cv2.threshold(gary,0,255,cv2.THRESH_OTSU)

    #---------------------确定分割位置-------------------------------
    #水平投影
    # ---------------------------
    hor = [0 for z in range(0, img_h)]
    for j in range(0, img_h):
        for i in range(0, img_w):
            if binary[j, i] == 0:
                hor[j] += 1

    #---------------------------
    horCutIndexStart = []     #水平开始位置
    horCutIndexEnd   = []     #水平结束位置
    iminWidth=50
    iminVal=12
    istart=0
    iWidthCount=0
    for j in range(0, img_h):
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

    print("文件夹%s第%d张图行数:%d"% (strtemp,imgID,len(horCutIndexStart)))

################行数不是4的整数倍##############################################################################
    if(len(horCutIndexStart)%4!=0):  #行数不是4的整数倍

        verCutIndexStart = []  # 垂直开始位置
        verCutIndexEnd = []  # 垂直结束位置
        # ------------切割成行---------------------------------
        for h in range(0, len(horCutIndexStart)-len(horCutIndexStart)%4, 4):  # 每4行一段
            cutLineImage = binary[int(horCutIndexStart[h]):int(horCutIndexEnd[h + 3]), int(0):int(img_w)]
            # cv2.imshow("666", cutLineImage)
            # cv2.waitKey(0)

            # 返回高和宽
            (lineImg_h, lineImg_w) = cutLineImage.shape
            # 垂直投影
            ver = [0 for z in range(0, lineImg_w)]
            for j in range(0, lineImg_w):  # 遍历一列
                for i in range(0, lineImg_h):  # 遍历一行
                    if cutLineImage[i, j] == 0:
                        ver[j] += 1

            # ------------垂直投影---------------
            verCutIndexStart.clear()  # 垂直开始位置 清空
            verCutIndexEnd.clear()    # 垂直结束位置  清空
            iminWidth = 60
            iminVal = 2
            istart = 0
            iWidthCount = 0
            for j in range(0, img_w):
                if (ver[j] > iminVal):
                    iWidthCount += 1
                    if (iWidthCount == 1):
                        istart = j - 2
                if (ver[j] < iminVal and iWidthCount > iminWidth):
                    verCutIndexStart.append(istart)
                    verCutIndexEnd.append(j + 2)
                    iWidthCount = 0
                if (ver[j] < iminVal and iWidthCount < iminWidth):
                    istart = 0
                    iWidthCount = 0
            # ------------垂直投影---------------

            print("文件夹%s第%d张图行数:%d,第%d行-第%d行有%d列" % (
            strtemp, imgID, len(horCutIndexStart), h + 1, h + 4, len(verCutIndexStart)))
            # ---------切割每行成单个字符---------------------
            for i_h in range(h, h + 4):
                for w in range(0, len(verCutIndexStart)):
                    cutCharImage = image[int(horCutIndexStart[i_h]):int(horCutIndexEnd[i_h]),
                                   int(verCutIndexStart[w]):int(verCutIndexEnd[w])]

                    # ----------保存--------------
                    strImgName = ''  # 图像名称
                    if (imgID >= 1 and imgID <= 46):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 1) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 47 and imgID <= 115):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 47) * 16 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 116 and imgID <= 161):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 116) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 162 and imgID <= 207):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 162) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    strImgName = strImgName[0:1]  # 图片序号，字符型，去掉\n
                    imgname = (r"\%s\%s%d_%s_%d.jpg" % (strImgName, strtemp, imgID, strImgName, imgNo))

                    savePath = baseSavePath + imgname
                    # 图像大小归一化
                    cutImage = cv2.resize(cutCharImage, (64, 64))
                    cv2.imencode('.jpg', cutImage)[1].tofile(savePath)
                    # print(savePath)
                    # ----------保存--------------

            # ---------切割每行成单个字符---------------------

        # ------------切割成行---------------------------------

        #==========================余数行数==========================================
        # ------------切割成行---------------------------------
        for h in range(len(horCutIndexStart)-len(horCutIndexStart)%4, len(horCutIndexStart)):

            # ---------切割每行成单个字符---------------------
            i_h=h    #为了下面代码不换i_h

            if(h!=len(horCutIndexStart)-1):    #非倒数第一行
                for w in range(0, len(verCutIndexStart)):
                    cutCharImage = image[int(horCutIndexStart[i_h]):int(horCutIndexEnd[i_h]),int(verCutIndexStart[w]):int(verCutIndexEnd[w])]

                    # ----------保存--------------
                    strImgName = ''  # 图像名称
                    if (imgID >= 1 and imgID <= 46):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 1) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 47 and imgID <= 115):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 47) * 16 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 116 and imgID <= 161):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 116) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 162 and imgID <= 207):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 162) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    strImgName = strImgName[0:1]  # 图片序号，字符型，去掉\n
                    imgname = (r"\%s\%s%d_%s_%d.jpg" % (strImgName, strtemp, imgID, strImgName, imgNo))

                    savePath = baseSavePath + imgname
                    # 图像大小归一化
                    cutImage = cv2.resize(cutCharImage, (64, 64))
                    cv2.imencode('.jpg', cutImage)[1].tofile(savePath)
                    # print(savePath)
                    # ----------保存--------------
            if (h == len(horCutIndexStart) - 1):  # 倒数第一行
                for w in range(0, 4):
                    cutCharImage = image[int(horCutIndexStart[i_h]):int(horCutIndexEnd[i_h]),
                                   int(verCutIndexStart[w]):int(verCutIndexEnd[w])]

                    # ----------保存--------------
                    strImgName = ''  # 图像名称
                    if (imgID >= 1 and imgID <= 46):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 1) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 47 and imgID <= 115):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 47) * 16 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 116 and imgID <= 161):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 116) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    if (imgID >= 162 and imgID <= 207):
                        imgNo = i_h * 19 + w + 1
                        strImgName = lines[(imgID - 162) * 24 * 19 + imgNo - 1]  # 读取图像名称
                    strImgName = strImgName[0:1]  # 图片序号，字符型，去掉\n
                    imgname = (r"\%s\%s%d_%s_%d.jpg" % (strImgName, strtemp, imgID, strImgName, imgNo))

                    savePath = baseSavePath + imgname
                    # 图像大小归一化
                    cutImage = cv2.resize(cutCharImage, (64, 64))
                    cv2.imencode('.jpg', cutImage)[1].tofile(savePath)
                    # print(savePath)
                    # ----------保存--------------

        # ---------切割每行成单个字符---------------------

    # ------------切割成行---------------------------------
        # ==========================余数行数==========================================
################行数不是4的整数倍##############################################################################

################行数是4的整数倍################################################################################
    if (len(horCutIndexStart) % 4 == 0):  # 行数是4的整数倍
        #------------切割成行---------------------------------
        for h in range(0, len(horCutIndexStart),4):  # 每4行一段
            cutLineImage = binary[int(horCutIndexStart[h]):int(horCutIndexEnd[h+3]),int(0):int(img_w)]
            # cv2.imshow("666", cutLineImage)
            # cv2.waitKey(0)

            # 返回高和宽
            (lineImg_h, lineImg_w) = cutLineImage.shape
            # 垂直投影
            ver = [0 for z in range(0, lineImg_w)]
            for j in range(0, lineImg_w):      # 遍历一列
                for i in range(0, lineImg_h):  # 遍历一行
                    if cutLineImage[i, j] == 0:
                        ver[j] += 1
            # ---------------------------
            verCutIndexStart = []  # 垂直开始位置
            verCutIndexEnd = []  # 垂直结束位置
            iminWidth = 60
            iminVal = 2
            istart = 0
            iWidthCount = 0
            for j in range(0, img_w):
                if (ver[j] > iminVal):
                    iWidthCount += 1
                    if (iWidthCount == 1):
                        istart = j - 2
                if (ver[j] < iminVal and iWidthCount > iminWidth):
                    verCutIndexStart.append(istart)
                    verCutIndexEnd.append(j + 2)
                    iWidthCount = 0
                if (ver[j] < iminVal and iWidthCount < iminWidth):
                    istart = 0
                    iWidthCount = 0
            print("文件夹%s第%d张图行数:%d,第%d行-第%d行有%d列" % (strtemp, imgID, len(horCutIndexStart),h+1,h+4,len(verCutIndexStart)))
            #---------切割每行成单个字符---------------------
            for i_h in range(h, h+4):
                for w in range(0, len(verCutIndexStart)):
                    cutCharImage = image[int(horCutIndexStart[i_h]):int(horCutIndexEnd[i_h]), int(verCutIndexStart[w]):int(verCutIndexEnd[w])]

                    #----------保存--------------
                    strImgName=''   #图像名称
                    if(imgID>=1 and imgID<=46):
                       imgNo = i_h * 19 + w + 1
                       strImgName=lines[(imgID-1)*24*19+imgNo-1]     #读取图像名称
                    if(imgID>=47 and imgID<=115):
                       imgNo = i_h * 19 + w + 1
                       strImgName=lines[(imgID-47)*16*19+imgNo-1]    #读取图像名称
                    if(imgID>=116 and imgID<=161):
                       imgNo = i_h * 19 + w + 1
                       strImgName=lines[(imgID-116)*24*19+imgNo-1]   #读取图像名称
                    if(imgID>=162 and imgID<=207):
                       imgNo = i_h * 19 + w + 1
                       strImgName=lines[(imgID-162)*24*19+imgNo-1]   #读取图像名称
                    strImgName = strImgName[0:1]  # 图片序号，字符型，去掉\n
                    imgname = (r"\%s\%s%d_%s_%d.jpg" % (strImgName, strtemp, imgID, strImgName, imgNo))

                    savePath=baseSavePath+imgname
                    # 图像大小归一化
                    cutImage = cv2.resize(cutCharImage, (64, 64))
                    cv2.imencode('.jpg', cutImage)[1].tofile(savePath)
                    # print(savePath)
                    # ----------保存--------------

            # ---------切割每行成单个字符---------------------

        # ------------切割成行---------------------------------
################行数是4的整数倍################################################################################

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
                print("共%d个文件，正在处理第%d张图片..." % (len(list), int_imgID))
                print("正在处理第图片序列号%d..." % (int_imgID))
            image = cv_imread(filename, 1)
            # -----------处理图像--------------
            #-----倾斜校正---- 传入彩色图像
            rotate_img, angle = copybook_correction(image)

            edge_x = 50
            edge_y = 80
            # 设置ROI 截掉可能存在的边框
            image_ROI = rotate_img[edge_y:-edge_y, edge_x:-edge_x]

            #-----剪切图像---- 传入彩色图像
            MyCutImg(image_ROI, lines, baseSavePath,strtemp,int_imgID)
            # -----------处理图像--------------
        else:
            continue

print("运行结束！")
