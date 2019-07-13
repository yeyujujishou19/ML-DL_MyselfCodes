import cv2
# pic = cv2.imread('./imgs/0A9A.jpg')
# pic = cv2.resize(pic, (80, 26), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('', pic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import os
import numpy as np

# root_path = "D:/"
# dir = root_path + "1" + "/"
dir="E:/sxl_Programs/Python/PDF2/"
count = 161
for root, dir0, files in os.walk(dir):
    for file in files:
        srcImg = cv2.imread(dir+ str(file))
        # str=dir + str(count)+".jpg"
        cv2.imwrite("E:/sxl_Programs/Python/PDF/test/"+ str(count)+".jpg", srcImg)
        count += 1
        # pic = cv2.resize(srcImg, (80, 26), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(dir+ str(file), pic)
        # count+=1
        # if(count%100==0):
        #     print("Iter:%d" % count)
        # cv2.imshow('', srcImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
