# coding=utf-8
import cv2
import os
import numpy as np
import re   #查找字符串   re.finditer(word, path)]


# 度数转换
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res


# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    print(RotateMatrix)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate


# 通过霍夫变换计算角度
def CalcDegree(srcImage):
    #midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(srcImage, 50, 200, 3)
    lineimage = srcImage.copy()

    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 239)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    sum = 0
    icount=0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))

            #只取从左上角-》右下角（正确倾角）
            if(x0>0 and y0<0 and x1>0 and y1<0 and x2>0 and y2>0):
                sum += theta
                icount+=1
                #cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                #cv2.imshow("Imagelines", lineimage)
            # cv2.waitKey(0)
    # 对所有角度求平均，这样做旋转效果会更好
    if (icount==0):
        icount=1
    average = sum / icount
    angle = DegreeTrans(average) - 90
    return angle

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

if __name__ == '__main__':
    path=r"D:\sxl\处理图片\汉字分类\2万倾斜校正"
    list = TraverFolders(path)
    count=0
    countImg=0
    iDisPlay=1
    for filename in list:
        count += 1
        # -----确定子文件夹名称------------------
        word = r'\\'
        a = [m.start() for m in re.finditer(word, filename)]
        if (len(a) == 5):  # 字文件夹
            strtemp = filename[a[-1] + 1:]  # 文件夹名称-字符名称
            print(filename)
        # -----确定子文件夹名称------------------

        # -----子文件夹图片特征提取--------------
        if (len(a) == 6):  # 子文件夹下图片
            if ('.tif' in filename):
                str_imgID = filename[-8:-4]  # 图片序号，字符型
                int_imgID = int(str_imgID)  # 图片序号，转换成整型
                countImg += 1
                if (countImg % iDisPlay == 0):
                    print("共%d个文件，正在处理第%d张图片..." % (len(list), countImg))
                image = cv_imread(filename, 0)
            else:
                continue
            # image = cv2.imread("D://1.tif")

            img_h = image.shape[0]
            img_w = image.shape[1]
            imageCopy=image.copy()
            imageCopy = cv2.resize(imageCopy, (int(img_w/3), int(img_h/3)))

            # image =cv_imread("D://2.tif", type=0)
            #cv2.imshow("Image", image)
            # 倾斜角度矫正
            degree = CalcDegree(imageCopy)
            #print("调整角度：", degree)
            rotate = rotateImage(image, degree-90)
            #cv2.imshow("rotate", rotate)
            savePath=(r"D:\\2万汉字校正结果\\%s.tif" % str_imgID)
            cv2.imencode('.tif', rotate)[1].tofile(savePath)
           # cv2.waitKey(0)
            #cv2.destroyAllWindows()
