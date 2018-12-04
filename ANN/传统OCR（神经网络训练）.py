#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from math import ceil        #向上取整
import time
import os
import random
import gc                    #释放内存
import tensorflow as tf
from tensorflow.python.framework import graph_util

logdir='./model/'

#全局变量
inpyNUM = 8              # npy文件数量
n_fc1 = 672               #第一层隐藏层神经元个数
n_fc2 = 336               #第二层隐藏层神经元个数
n_feature = 336           #特征维度
n_label = 21477           #标签维度
# learning_rate = 1e-4  #学习率
batch_size = 5000         #最小数据集大小
training_iters = 200000    #训练次数
test_step = 1000          #测试数据步长
display_step = 100        #显示数据步长  100
MODEL_SAVE_PATH=r"./model/"
MODEL_NAME="ANN_model_21477"
#指数衰减型学习率
LEARNING_RATE_BASE=1e-4   #最初学习率
LEARNING_RATE_DECAY=0.5   #学习率衰减率
LEARNING_RATE_STEP=10000  #喂入多少轮BATCH_SIZE后，更新一次学习率，一般为总样本数/batch_size

############################################################################

#多个数组按同一规则打乱数据
def ShuffledData(features,labels):
    '''
    @description:随机打乱数据与标签，但保持数据与标签一一对应
    @author:RenHui
    '''
    permutation = np.random.permutation(features.shape[0])
    shuffled_features = features[permutation,:]
    shuffled_labels = labels[permutation]
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

# 处理白边
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

#字符图像的特征提取方法
#要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
def SimpleGridFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:RenHui
    '''

    new_img = HandWhiteEdges(image)  # 白边处理
    #图像大小归一化
    image = cv2.resize(new_img,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    #计算网格大小
    grid_size=16
    grid_h = binary.shape[0]/grid_size
    grid_w = binary.shape[1]/grid_size
    #定义特征向量
    feature = np.zeros(grid_size*grid_size)
    for j in range(grid_size):
        for i in range(grid_size):
            grid = binary[int(j*grid_h):int((j+1)*grid_h),int(i*grid_w):int((i+1)*grid_w)]
            feature[j*grid_size+i] = grid[grid==0].size
    return feature

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

#预测函数
def prediction(logits):
    with tf.name_scope('prediction'):
        pred = tf.argmax(logits,1)
    return pred
############################################################################

#将数据集和标签分成小块
def gen_batch(data,labels,batch_size):
    global flag   #全局标记位
    if(flag+batch_size>data.shape[0]):
        flag = 0
        #最后一个batch_size从末尾向前取数据
        data_batch = data[data.shape[0]-batch_size:data.shape[0]]
        labels_batch = labels[data.shape[0]-batch_size:data.shape[0]]
        return data_batch, labels_batch
    data_batch = data[flag:flag+batch_size]
    labels_batch = labels[flag:flag+batch_size]
    flag = flag+batch_size
    return data_batch,labels_batch

# w,b
def weight_variable(layer,shape,stddev,name):
    with tf.name_scope(layer):  #该函数返回用于定义 Python 操作系统的上下文管理器，生成名称范围。
        return tf.Variable(tf.truncated_normal(shape,stddev=stddev,name=name))
def bias_variable(layer, value, dtype, shape, name):
    with tf.name_scope(layer):
        return tf.Variable(tf.constant(value, dtype=dtype, shape=shape, name=name))

#前向计算
def inference(features):
    # 计算第一层隐藏层 a=x*w+b
    with tf.name_scope('fc1'):
        fc1 = tf.add(tf.matmul(features,weight_variable('fc1',[n_feature,n_fc1],0.04,'w_fc1')),bias_variable('fc1',0.1,tf.float32,[n_fc1],'b_fc1'))
        fc1 = tf.nn.relu(fc1)  #激励函数
    # 计算第二层隐藏层 a=x*w+b
    with tf.name_scope('fc2'):
        fc2 = tf.add(tf.matmul(fc1,weight_variable('fc2',[n_fc1, n_fc2],0.04,'w_fc2')),bias_variable('fc2',0.1,tf.float32,[n_fc2],'b_fc2'))
        fc2 = tf.nn.relu(fc2)  #激励函数
    with tf.name_scope('output'):
        fc3 = tf.add(tf.matmul(fc2,weight_variable('fc3',[n_fc2,n_label],0.04,'w_fc3')),bias_variable('fc3',0.1,tf.float32,[n_label],'b_fc3'), name="output")   ####这个名称很重要！！)
    return fc3

#损失函数
def loss(logits,labels):
    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
        tf.summary.scalar('cost',cost)
    return cost

#训练，减小损失函数
def train(cost):
    #此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
    #相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)
    return optimizer

#准确率
def accuracy(labels,preds):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(labels,1),tf.argmax(preds,1))
        acc = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    return acc

# 从数字标签转换为数组标签 [0,0,0,...1,0,0]
def InitImagesLabels(labels_batch):
    labels_batch_new=[]
    for id in labels_batch:
        aa = np.zeros(n_label, np.int16)
        aa[id] = 1
        labels_batch_new.append(aa)
    return labels_batch_new

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
#此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
#
# 定义输入和参数
# 用placeholder实现输入定义（sess.run中喂一组数据）
features_tensor = tf.placeholder(tf.float32,shape=[None,n_feature], name="input") ####这个名称很重要！！！
labels_tensor = tf.placeholder(tf.float32,shape=[None,n_label], name='labels')

#------------指数衰减型学习率----------------------------
#运行了几轮batch_size的计数器，初值给0，设为不被训练
global_step=tf.Variable(0,trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)
#-----------指数衰减型学习率-----------------------------

#前向计算
# logits = inference(features_tensor)

keep_prob=tf.placeholder(tf.float32)  #drop_out比率

# 计算第一层隐藏层 a=x*w+b
with tf.name_scope('fc1'):
    fc1 = tf.add(tf.matmul(features_tensor, weight_variable('fc1', [n_feature, n_fc1], 0.04, 'w_fc1')),
                 bias_variable('fc1', 0.1, tf.float32, [n_fc1], 'b_fc1'))
    fc1 = tf.nn.relu(fc1)  # 激励函数
    L1_drop = tf.nn.dropout(fc1, keep_prob)
# 计算第二层隐藏层 a=x*w+b
with tf.name_scope('fc2'):
    fc2 = tf.add(tf.matmul(L1_drop, weight_variable('fc2', [n_fc1, n_fc2], 0.04, 'w_fc2')),
                 bias_variable('fc2', 0.1, tf.float32, [n_fc2], 'b_fc2'))
    fc2 = tf.nn.relu(fc2)  # 激励函数
    L2_drop = tf.nn.dropout(fc2, keep_prob)

logits = tf.add(tf.matmul(L2_drop, weight_variable('fc3', [n_fc2, n_label], 0.04, 'w_fc3')),
                 bias_variable('fc3', 0.1, tf.float32, [n_label], 'b_fc3'))

y = tf.nn.softmax(logits, name="output") # 预测值 这个名称很重要！！ 输出概率值

#损失函数
cost = loss(logits,labels_tensor)
#训练减小损失函数
optimizer = train(cost)

merged_summary_op = tf.summary.merge_all()  #合并默认图形中的所有汇总

#返回准确率
acc = accuracy(labels_tensor,logits)

# 我们经常在训练完一个模型之后希望保存训练的结果，
# 这些结果指的是模型的参数，以便下次迭代的训练或者用作测试。
# Tensorflow针对这一需求提供了Saver类。
saver = tf.train.Saver(max_to_keep=1)  #保存网络模型
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
#会话graph=tf.Graph()
with tf.Session() as sess:
    print("启动执行...")
    time_start = time.time()  #计时

    sess.run(tf.global_variables_initializer()) #用于初始化所有的变量
    sess.run(tf.local_variables_initializer())

    #----------断点续训--------------------------
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # ----------断点续训--------------------------

    #logs是事件文件所在的目录，这里是工程目录下的logs目录。第二个参数是事件文件要记录的图，也就是tensorflow默认的图。
    summary_writer = tf.summary.FileWriter('logs',graph=tf.get_default_graph())

    #==============循环读取-训练-数据进行训练==================================================================
    flag = 0
    icounter=0
    while icounter<training_iters:  # training_iters 训练次数

        for iNo in range(inpyNUM):   # npy文件数量
            # 读取数据
            train_data_npyPath = (r".\npy3\Img21477_features_train_%d.npy" % (iNo + 1))    #npy路径
            train_labels_npyPath = (r".\npy3\Img21477_labels_train_%d.npy" % (iNo + 1))    #npy路径

            #print("共%d个训练数据集，加载第%d训练数据集中..." % (inpyNUM, iNo + 1))
            train_data = np.load(train_data_npyPath).astype(np.int16)      #加载数据
            train_labels = np.load(train_labels_npyPath).astype(np.int16)  #加载数据


            # print(train_data.shape, train_labels.shape)  # 打印数据参数

            # len_train_data=int(len(train_data)/batch_size)  #训练集被分成了n个batch_size
            len_train_data =int(ceil(len(train_data)/batch_size))  #ceil(3.01)=4  向上取整
            for i_batchSize in range (len_train_data):             #对当前训练集逐个获取batch_size块的数据
                data_batch,labels_batch = gen_batch(train_data,train_labels,batch_size) #获取一块最小数据集
                # data_batch, labels_batch = ShuffledData(data_batch, labels_batch)  #每次读取都打乱数据
                labels_batch_new = InitImagesLabels(labels_batch)  #从数字标签转换为数组标签 [0,0,0,...1,0,0]
                #_,c,summary = sess.run([optimizer,cost,merged_summary_op],feed_dict={features_tensor:data_batch,labels_tensor:labels_batch})
                _, c, icounter = sess.run([optimizer, cost, global_step],feed_dict={features_tensor: data_batch,
                                                                                    labels_tensor: labels_batch_new,keep_prob:1})

                #----------指数衰减型学习率--------------------
                learning_rate_val=sess.run(learning_rate)
                global_step_val=sess.run(global_step)
                # print("After %s steps,global_step is %f,learing rate is %f"%(icounter,global_step_val, learning_rate_val))
                #----------指数衰减型学习率--------------------

                icounter+=1  #执行轮数
                if (icounter) % test_step == 0:  # test_step = 1000
                    # ==============循环读取-测试-数据进行测试==================================================================
                    sumAccuracy = 0
                    countNum=0    #测试数据总共多少轮
                    for iNo in range(inpyNUM):
                        # 读取测试数据
                        test_data_npyPath = (r".\npy3\Img21477_features_test_%d.npy" % (iNo + 1))
                        test_labels_npyPath = (r".\npy3\Img21477_labels_test_%d.npy" % (iNo + 1))
                        test_data = np.load(test_data_npyPath).astype(np.int16)
                        test_labels = np.load(test_labels_npyPath).astype(np.int16)

                        len_test_data = int(ceil(len(test_data) / batch_size))  # ceil(3.01)=4  向上取整
                        for i_batchSize in range(len_test_data):  # 对当前训练集逐个获取batch_size块的数据
                            test_data_batch, test_labels_batch = gen_batch(test_data, test_labels, batch_size)  # 获取一块最小数据集
                            # test_data_batch, test_labels_batch = ShuffledData(test_data_batch, test_labels_batch)  # 每次读取都打乱数据
                            test_labels_batch_new = InitImagesLabels(test_labels_batch)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
                            a = sess.run(acc, feed_dict={features_tensor: test_data_batch,
                                                         labels_tensor: test_labels_batch_new, keep_prob: 1})  # 测试数据
                            countNum += 1
                            sumAccuracy = sumAccuracy + a  # 计算正确率和

                    avgAccuracy = sumAccuracy / countNum
                    print(("测试数据集正确率为%f:" % (avgAccuracy)))
                # ==============循环读取-测试-数据进行测试==================================================================

                if (icounter) % display_step == 0:  # display_step = 100
                    # ----------断点续训--------------------------
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                    # ----------断点续训--------------------------

                    # ----------显示消息--------------------------
                    print("Iter" + str(icounter) +",learing rate="+ "{:.8f}".format(learning_rate_val)+",Training Loss=" + "{:.6f}".format(c))
                    # ----------显示消息--------------------------

                    # ----------保存pb----------------------------
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])  #["inputs", "outputs"] 此处字符串和前面对应上
                    with tf.gfile.FastGFile("./model/OCRsoftmax21477.pb", mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
                    # ----------保存pb----------------------------
        #for iNo in range(inpyNUM):  # npy文件数量


    summary_writer.close()  # log记录文件关闭
    time_end = time.time()
    time_h = (time_end - time_start) / 3600
    print('训练用时：%.2f 小时' % time_h)
    print("Optimization Finished!")
    #==============循环读取-训练-数据进行训练==================================================================

    # saver.save(sess, r"./save_ANN.ckpt")  #保存模型


# ###############预测##############################################################
#     # char_set = ['嫒', '扳', '板', '苯', '笨', '币', '厕', '侧', '厂', '憧', '脆', '打', '大', '钉', '丢', '懂',
#     #                '儿', '非', '干', '古', '诂', '诡', '乎', '话', '几', '己', '减', '減', '韭', '钜', '决', '決',
#     #                '扛', '钌', '昧', '孟', '末', '沐', '钼', '蓬', '篷', '平', '亓', '千', '去', '犬', '壬', '晌',
#     #                '舌', '十', '士', '市', '拭', '栻', '適', '沭', '耍', '巳', '趟', '天', '土', '王', '枉', '未',
#     #                '味', '文', '兀', '淅', '晰', '响', '写', '要', '已', '盂', '与', '元', '媛', '丈', '趙', '谪',
#     #                '锺', '鍾', '柱', '状', '狀']
#     char_set = np.load(r"./npy/ImgHanZiName653.npy")
#     char_set = char_set.tolist()
#     data_dir=r"D:\sxl\处理图片\汉字分类\train653"
#     pred = prediction(logits)
#     error_n = 0
#     print ("预测图像中...")
#     precount=0
#     for dirname in os.listdir(data_dir):   #返回指定路径下所有文件和文件夹的名字，并存放于一个列表中。
#         #os.walk 返回的是一个3个元素的元组 (root, dirs, files) ，分别表示遍历的路径名，该路径下的目录列表和该路径下文件列表
#         for parent, _, filenames in os.walk(data_dir + '/' + dirname):
#             for filename in filenames:
#                 if (filename[-4:] != '.jpg'):
#                     continue
#                 precount+=1
#                 if (precount % 100 == 0):
#                     print("正在执行第%d个..." % (precount))
#                 image = cv_imread(parent + '/' + filename, 0)
#                 feature =SimpleGridFeature(image).reshape(-1, 256)   #提取特征 reshape(-1, 64) 行数未知，列数等于64
#                 p = sess.run(pred, feed_dict={features_tensor: feature})  #当前图片和之前训练好的模型比较，返回预测结果
#                 if (char_set[p[0]] != dirname):
#                     error_n += 1
#                     cv2.imencode('.jpg', image)[1].tofile(
#                         'error_images/' + dirname+ '_'+char_set[p[0]] + '_' + str(error_n) + '.jpg')
# ###############预测##############################################################
time_end=time.time()
time_h=(time_end-time_start)/3600
print('总用时：%.6f 小时'% time_h)
print ("运行结束！")
#####################################################
