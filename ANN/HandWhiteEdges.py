# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pylab as pl  #画图
import random
import re   #查找字符串   re.finditer(word, path)]

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

# 处理白边
#找到上下左右的白边位置
#剪切掉白边
#二值化
#将图像放到64*64的白底图像中心
def HandWhiteEdges(img):
    ret, thresh1 = cv2.threshold(img, 249, 255, cv2.THRESH_BINARY)
    # pl.figure("img")
    # pl.imshow(img)
    # pl.show()
    # OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 膨胀图像
    thresh1 = cv2.dilate(thresh1, kernel)
    # pl.figure("thresh1")
    # pl.imshow(thresh1)
    # pl.show()
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
        val=thresh1.sum(axis=1)[56]
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

    #创建全白图片
    imageTemp = np.zeros((64, 64, 3), dtype=np.uint8)
    imageTemp = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2GRAY)
    imageTemp.fill(255)

    if(tempr1-tempr0==0 or tempc1-tempc0==0):   #空图
        return imageTemp    #返回全白图

    new_img = img[tempr0:tempr1, tempc0:tempc1]
    # pl.figure("new_img")
    # pl.imshow(new_img)
    # pl.show()
    #二值化
    retval,binary = cv2.threshold(new_img,0,255,cv2.THRESH_OTSU)
    # pl.figure("binary")
    # pl.imshow(binary)
    # pl.show()

    # #叠加两幅图像
    rstImg=ImageOverlay(imageTemp, binary)
    # #叠加两幅图像
    # rows= binary.shape[0]
    # cols = binary.shape[1]
    # beginrow=int((64-rows)/2)
    # begincol=int((64-cols)/2)
    # for row in range(rows):
    #     for col in range(cols):
    #         if(binary[row,col]==0):
    #             # imageTemp[row+beginrow, col+begincol]=binary[row, col]
    #             imageTemp[row+beginrow, col+begincol]=binary[row, col]
    pl.figure("rstImg")
    pl.imshow(rstImg)
    pl.show()

    return rstImg

#遍历文件夹
list = []
def TraverFolders(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        list.append(path)
        if os.path.isdir(path):
            TraverFolders(path)
    return list

# 读图
# path = r'D:/1.jpg'
# image = cv_imread(path,0)
# newimg=HandWhiteEdges(image)
# savePath = (r"D:/2.jpg")
# cv2.imencode('.jpg', newimg)[1].tofile(savePath)  # 保存图片
# print(image.shape)

# #-----------------------------------
path=r"D:\sxl\处理图片\汉字分类\train653_bad"
print("遍历图像中，可能需要花一些时间...")
list=TraverFolders(path)

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
        print(filename)
        # print(ImgLabelsTemp)
    # -----确定子文件夹名称------------------

    # -----子文件夹图片特征提取--------------
    if (len(a) == 6):   #子文件夹下图片
       if('.jpg'in filename):
          image = cv_imread(filename,0)
          # print("filename")
          # print(filename)
          newimg = HandWhiteEdges(image)
          strSavePath=str(filename)

          strSavePath=strSavePath.replace("train653_bad", "train653_badHandle")
          strSavePath=strSavePath.replace('.jpg', '_d.jpg')
          # print("strSavePath")
          # print(strSavePath)
          cv2.imencode('.jpg', newimg)[1].tofile(strSavePath)  # 保存图片
       else:
           continue
#
