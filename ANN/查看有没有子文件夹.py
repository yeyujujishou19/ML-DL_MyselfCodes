import numpy as np
import os
import re   #查找字符串   re.finditer(word, path)]

#遍历文件夹
list = []
def TraverFolders(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        list.append(path)
        if os.path.isdir(path):
            TraverFolders(path)
    return list


path=r"D:\sxl\处理图片\汉字分类\train3000"     #文件夹路径
print("遍历图像中，可能需要花一些时间...")
list=TraverFolders(path)
print("总数%d"% (len(list)) )
count=0
iDisPlay=1
for filename in list:
    count+=1
    # if(count%iDisPlay==0):
       # print("共%d个文件，正在处理第%d个..." % (len(list),count))
    #-----确定子文件夹名称------------------
    word = r'\\'
    a = [m.start() for m in re.finditer(word, filename)]
    if(len(a)==5):   #字文件夹
        strtemp=filename[a[-1]+1:]  #文件夹名称-字符名称
        print(strtemp)
    # -----子文件夹图片特征提取--------------
    if (len(a) == 6):  # 子文件夹下图片
        if ('.jpg' in filename):
            continue
        else:
            strtemp = filename[a[-1] + 1:]  # 文件夹名称-字符名称
            print("       %s"%strtemp)
        #ImgLabelsTemp=InitImagesLabels(iMyindex)  #创建标签
        # print(iMyindex)
        # print(ImgLabelsTemp)
        # print(ImgLabelsTemp)
    # -----确定子文件夹名称------------------
    # if(len(a)==6):   #字文件夹
    #     strtemp=filename[a[-1]+1:]  #文件夹名称-字符名称
    #     print(""strtemp)

