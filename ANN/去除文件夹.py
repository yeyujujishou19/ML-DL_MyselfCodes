import numpy as np
import os
import shutil     #删除文件夹，包括文件夹内容

basePath=( r"D:/sxl/处理图片/汉字分类/train3587/" )
# deletePath=basePath+"66"
# shutil.rmtree(deletePath)

strcharName3000= np.load( r"./npy/ImgHanZiName3000.npy" )
strcharName3000=strcharName3000.tolist()

strcharName3587= np.load( r"./npy/ImgHanZiName3587.npy" )
strcharName3587=strcharName3587.tolist()
deleteCount=0
count=0
dstPath="D:\sxl\处理图片\汉字分类\新建文件夹"
for index, strcharName in enumerate(strcharName3587):
  count=index+1
  # if(count%1==0):
  #     print(count)
  if strcharName in strcharName3000:  #如果文件夹名称在3000类中，则跳过
      continue
  else:
      deletePath=basePath+strcharName #如果文件夹名称不在3000类中，则删除该文件夹
      deleteCount+=1
      print("%d: %s"%(deleteCount,deletePath))
      #shutil.rmtree(deletePath)
      # 移动文件夹
      shutil.move(deletePath, dstPath)


# path1=(r"D:/66/")
# shutil.rmtree(path1)