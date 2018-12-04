import multiprocessing
import os, time, random
import numpy as np
import cv2
import os
import sys
from time import ctime

# image_dir = r"D:/sxl/处理图片/汉字分类/train85/"       #图像文件夹路径
image_dir = r"E:/2万汉字分类/train21477/"       #图像文件夹路径
# image_dir = r"D:/sxl/处理图片/汉字分类/train85/"       #图像文件夹路径
data_type = 'test'
save_path = r'E:/sxl_Programs/Python/ANN/npy3/'       #存储路径
data_name = 'Img21477'                                #npy文件名

char_set = np.array(os.listdir(image_dir))            #文件夹名称列表
np.save(save_path+'ImgHanZiName21477.npy',char_set)   #文件夹名称列表
char_set_n = len(char_set)                            #文件夹列表长度

read_process_n = 4    #进程数
repate_n = 4          #随机移动次数
data_size = 1000000   #1个npy大小

shuffled = True      #是否打乱

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

#多个数组按同一规则打乱数据
def ShuffledData(features,labels):
    '''
    @description:随机打乱数据与标签，但保持数据与标签一一对应
    @author:RenHui
    '''
    permutation = np.random.permutation(features.shape[0])
    shuffled_features = features[permutation,:]  #多维
    shuffled_labels = labels[permutation]       #1维
    return shuffled_features,shuffled_labels

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

#函数功能：处理白边
#找到上下左右的白边位置
#剪切掉白边
#二值化
#将图像放到64*64的白底图像中心
def HandWhiteEdges(img):
    ret, thresh1 = cv2.threshold(img, 249, 255, cv2.THRESH_BINARY)
    # OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 膨胀图像
    thresh1 = cv2.dilate(thresh1, kernel)
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

    # 创建全白图片
    imageTemp = np.zeros((64, 64, 3), dtype=np.uint8)
    imageTemp = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2GRAY)
    imageTemp.fill(255)

    if(tempr1-tempr0==0 or tempc1-tempc0==0):   #空图
        return imageTemp    #返回全白图

    new_img = img[tempr0:tempr1, tempc0:tempc1]
    #二值化
    retval,binary = cv2.threshold(new_img,0,255,cv2.THRESH_OTSU)

    #叠加两幅图像
    rstImg=ImageOverlay(imageTemp, binary)
    return rstImg

#函数功能：简单网格
#函数要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
#返回数据：64*1维特征
def SimpleGridFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:RenHui
    '''
    new_img=HandWhiteEdges(image)  #白边处理
    #new_img=image
    #图像大小归一化
    image = cv2.resize(new_img,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)

    #定义特征向量
    grid_size1 = 16
    grid_size2 = 8
    grid_size3 = 4
    feature = np.zeros(grid_size1*grid_size1+grid_size2*grid_size2+grid_size3*grid_size3,dtype=np.int16)

    #计算网格大小1
    grid_h1 = binary.shape[0]/grid_size1
    grid_w1 = binary.shape[1]/grid_size1
    for j in range(grid_size1):
        for i in range(grid_size1):
            grid = binary[int(j*grid_h1):int((j+1)*grid_h1),int(i*grid_w1):int((i+1)*grid_w1)]
            feature[j*grid_size1+i] = grid[grid==0].size

    #计算网格大小2
    grid_h2 = binary.shape[0]/grid_size2
    grid_w2 = binary.shape[1]/grid_size2
    for j in range(grid_size2):
        for i in range(grid_size2):
            grid = binary[int(j*grid_h2):int((j+1)*grid_h2),int(i*grid_w2):int((i+1)*grid_w2)]
            feature[grid_size1*grid_size1+j*grid_size2+i] = grid[grid==0].size

    #计算网格大小3
    grid_h3 = binary.shape[0]/grid_size3
    grid_w3 = binary.shape[1]/grid_size3
    for j in range(grid_size3):
        for i in range(grid_size3):
            grid = binary[int(j*grid_h3):int((j+1)*grid_h3),int(i*grid_w3):int((i+1)*grid_w3)]
            feature[grid_size1*grid_size1+grid_size2*grid_size2+j*grid_size3+i] = grid[grid==0].size

    return feature

#随机移动图像，黑白图
def randomMoveImage(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    # 0 上，1 下，2 左，3 右
    idirection=random.randrange(0, 4) #随机产生0,1,2,3
    #随机移动距离
    iPixsNum=random.randrange(1, 3) #随机产生1,2

    if (idirection == 0): #上
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, 0], [0, 1, -iPixsNum]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for h in range(iPixsNum):              # 从上到下
            for w in range(img_w):             # 从左到右
                dst[img_h-h-1, w] = 255

    if (idirection == 1): #下
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, 0], [0, 1, iPixsNum]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for h in range(iPixsNum):              # 从上到下
            for w in range(img_w):             # 从左到右
                dst[h, w] = 255

    if (idirection == 2): #左
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, -iPixsNum], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for w in range(iPixsNum):  # 从左到右
            for h in range(img_h):  # 从上到下
                dst[h, img_w - w - 1] = 255

    if (idirection == 3): #右
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, iPixsNum], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for w in range(iPixsNum):  # 从左到右
            for h in range(img_h):  # 从上到下
                dst[h, w] = 255
    return dst


# 写数据进程执行的代码:
def read_image_to_queue(queue):
    print('Process to write: %s' % os.getpid())
    for j,dirname in enumerate(char_set):  # dirname 是文件夹名称
        label = np.where(char_set==dirname)[0][0]     #文件夹名称对应的下标序号
        print('序号：'+str(j),'读 '+dirname+' 文件夹...时间：',ctime() )
        for parent,_,filenames in os.walk(os.path.join(image_dir,dirname)):
            for filename in filenames:
                if(filename[-4:]!='.jpg'):
                    continue
                image = cv_imread(os.path.join(parent,filename),0)

                # cv2.imshow(dirname,image)
                # cv2.waitKey(0)
                queue.put((image,label))
    
    for i in range(read_process_n):
        queue.put((None,-1))

    print('读图结束!')
    return True
        
# 读数据进程执行的代码:
def extract_feature(queue,lock,count):
    '''
    @description:从队列中取出图片进行特征提取
    @queue:先进先出队列
     lock：锁，在计数时上锁，防止冲突
     count:计数
    @author:RenHui
    '''

    print('Process %s start reading...' % os.getpid())

    global data_n
    features = [] #存放提取到的特征
    labels = [] #存放标签
    flag = True #标志着进程是否结束
    while flag:
        image,label = queue.get()  #从队列中获取图像和标签

        if len(features) >= data_size or label == -1:   #特征数组的长度大于指定长度，则开始存储

            array_features = np.array(features)  #转换成数组
            array_labels = np.array(labels)

            array_features,array_labels = ShuffledData(array_features,array_labels) #打乱数据
            
            lock.acquire()   # 锁开始

            # 拆分数据为训练集，测试集
            split_x = int(array_features.shape[0] * 0.8)
            train_data, test_data = np.split(array_features, [split_x], axis=0)     # 拆分特征数据集
            train_labels, test_labels = np.split(array_labels, [split_x], axis=0)  # 拆分标签数据集

            count.value += 1    #下标计数加1
            str_features_name_train = data_name+'_features_train_'+str(count.value)+'.npy'
            str_labels_name_train = data_name+'_labels_train_'+str(count.value)+'.npy'
            str_features_name_test = data_name+'_features_test_'+str(count.value)+'.npy'
            str_labels_name_test = data_name+'_labels_test_'+str(count.value)+'.npy'

            lock.release()   # 锁释放

            np.save(save_path+str_features_name_train,train_data)
            np.save(save_path+str_labels_name_train,train_labels)
            np.save(save_path+str_features_name_test,test_data)
            np.save(save_path+str_labels_name_test,test_labels)
            print(os.getpid(),'save:',str_features_name_train)
            print(os.getpid(),'save:',str_labels_name_train)
            print(os.getpid(),'save:',str_features_name_test)
            print(os.getpid(),'save:',str_labels_name_test)
            features.clear()
            labels.clear()

        if label == -1:
            break

        # 获取特征向量，传入灰度图
        feature = SimpleGridFeature(image)  # 简单网格
        features.append(feature)
        labels.append(label)

        # # 随机移动4次
        # for itime in range(repate_n):
        #     rMovedImage = randomMoveImage(image)
        #     feature = SimpleGridFeature(rMovedImage)  # 简单网格
        #     features.append(feature)
        #     labels.append(label)
    
    print('Process %s is done!' % os.getpid())

if __name__=='__main__':
    time_start = time.time()  # 开始计时

    # 父进程创建Queue，并传给各个子进程：
    image_queue = multiprocessing.Queue(maxsize=1000)  #队列
    lock = multiprocessing.Lock()                      #锁
    count = multiprocessing.Value('i',0)               #计数

    #将图写入队列进程
    write_sub_process = multiprocessing.Process(target=read_image_to_queue, args=(image_queue,))

    read_sub_processes = []                            #读图子线程
    for i in range(read_process_n):
        read_sub_processes.append(
            multiprocessing.Process(target=extract_feature, args=(image_queue,lock,count))
        )

    # 启动子进程pw，写入:
    write_sub_process.start()

    # 启动子进程pr，读取:
    for p in read_sub_processes:
        p.start()

    # 等待进程结束:
    write_sub_process.join()
    for p in read_sub_processes:
        p.join()

    time_end=time.time()
    time_h=(time_end-time_start)/3600
    print('用时：%.6f 小时'% time_h)
    print ("读图提取特征存npy,运行结束！")