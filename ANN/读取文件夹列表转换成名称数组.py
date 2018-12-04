# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
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

#遍历文件夹
list = []
def TraverFolders(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        list.append(path)   #只扫描到子文件夹
    return list

#--------------批量读图------------------------------------------------------------------------------------
# path=r"D:\sxl\处理图片\汉字分类\train3527"
path=r"D:\sxl\处理图片\汉字分类\train653"
# path=r"D:\sxl\处理图片\汉字分类\10类字符"
list=TraverFolders(path)
#print(list)
strcharName=[]
print("共%d个文件" % (len(list)))
count=0
for filename in list:
    count+=1
    #-----确定子文件夹名称------------------
    word = r'\\'
    a = [m.start() for m in re.finditer(word, filename)]
    if(len(a)==5):   #字文件夹
        strtemp=filename[a[-1]+1:]  #文件夹名称-字符名称
        strcharName.append(strtemp)
    # -----确定子文件夹名称------------------
    np.save("./npy2/ImgHanZiName653.npy",strcharName)
# np.save("./npy/ImgCharName_10.npy",strcharName)
#     fileObject = open('./npy2/ImgHanZiName3527.txt', 'w')
#     for ip in strcharName:
#         fileObject.write(ip)
#         fileObject.write('\n')
#     fileObject.close()

print (strcharName)
print ("运行结束！")

c = np.load( "./npy2/ImgHanZiName653.npy" )
print(c)
print(c.shape)