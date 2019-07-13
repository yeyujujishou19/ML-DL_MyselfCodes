import multiprocessing
import os, time, random
import numpy as np
import cv2
import os
import sys
from time import ctime

# 显示图像与标签
def ShowImageAndLabels(batch_xs, batch_ys):
    img_h = 64
    img_w = 64
    img = np.ones((img_h, img_w), dtype=np.uint8)
    icount=0
    for batch_image in batch_xs:  #转换成图像

        for h in range(img_h):
            for w in range(img_w):
                img[h, w]=batch_image[h * img_h + w] #图像复原

        sss="%d"%batch_ys[icount]
        # cv2.imshow(sss,img)
        # cv2.waitKey(0)
        cv2.imwrite(("D:/666/%d_%d.jpg"%(icount,batch_ys[icount])),img)
        icount+=1
        print (icount)



Img10_features_test_1 = np.load(r"E:/sxl_Programs/Python/CNN/npy/Img10_features_test_1.npy")
Img10_features_test_1 = Img10_features_test_1.tolist()

Img10_labels_test_1 = np.load(r"E:/sxl_Programs/Python/CNN/npy/Img10_labels_test_1.npy")
Img10_labels_test_1 = Img10_labels_test_1.tolist()


ShowImageAndLabels(Img10_features_test_1, Img10_labels_test_1)