# !/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow.python.framework import graph_util

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "data4_model"

inpyNUM = 8                 # npy文件数量
n_label = 4                 # 标签维度
batch_size = 1              # 每个批次的大小
learning_rate_base = 0.001  # 初始学习速率时0.1
decay_rate = 0.96           # 衰减率
global_steps = 50001        # 总的迭代次数
decay_steps = 500           # 衰减次数


# 参数概要,tf.summary.scalar的作用主要是存储变量，并赋予变量名，tf.name_scope主要是给表达式命名
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    # x input tensor of shape [batch,in_height,in_width,in_channels]
    # W filter/ kernel tensor of shape [filter_height,filter_width,in_channels,out_channels]
    # strides[0]=strides[3]=1恒等于1,
    # strides[1]代表x方向的步长，strides[2]代表y方向的步长，
    # padding:"SAME","VALID"
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    # x input tensor of shape [batch,in_height,in_width,in_channels]
    # ksize [1,x,y,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 从数字标签转换为数组标签 [0,0,0,...1,0,0]
def InitImagesLabels(labels_batch):
    labels_batch_new = []
    for id in labels_batch:
        aa = np.zeros(n_label, np.float32)
        aa[int(id)] = 1
        labels_batch_new.append(aa)
    return labels_batch_new


# 显示图像与标签
def ShowImageAndLabels(batch_xs, batch_ys):
    img_h = 128
    img_w = 256
    img = np.ones((img_h, img_w), dtype=np.uint8)
    icount = 0
    for batch_image in batch_xs:  # 转换成图像

        for h in range(img_h):
            for w in range(img_w):
                img[h, w] = batch_image[h * img_h + w]  # 图像复原

        sss = "%d" % batch_ys[icount]
        cv2.imshow(sss, img)
        cv2.waitKey(0)

        icount += 1


# keep_prob用来表示神经元的输出概率
with tf.name_scope('keep_prob'):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 32768], name='x-input')
    y = tf.placeholder(tf.float32, [None, 4], name='y-input')
    with tf.name_scope('x_image'):
        # 改变x的格式转为4D的向量[batch,in_height,in_width,in_channels]
        x_image = tf.reshape(x, [-1, 128, 256, 1], name='x_image')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu1'):
        # relu1 = tf.nn.relu(conv2d_1)
        relu1 = tf.nn.leaky_relu(conv2d_1)
        # relu1 = tf.nn.elu(conv2d_1)
        # relu3 = tf.nn.sigmoid(conv2d_3)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(relu1)  # 进行max_pooling
    # with tf.name_scope('h_pool1_drop'):
    #     h_pool1 = tf.nn.dropout(h_pool1, keep_prob, name='h_pool1_drop')

with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu2'):
        # relu2 = tf.nn.relu(conv2d_2)
        relu2 = tf.nn.leaky_relu(conv2d_2)
        # relu2 = tf.nn.elu(conv2d_2)
        # relu3 = tf.nn.sigmoid(conv2d_3)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(relu2)  # 进行max_pooling
    # with tf.name_scope('h_pool2_drop'):
    #     h_pool2 = tf.nn.dropout(h_pool2, keep_prob, name='h_pool2_drop')

with tf.name_scope('Conv3'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv3'):
        W_conv3 = weight_variable([3, 3, 64, 64], name='W_conv3')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv3'):
        b_conv3 = bias_variable([64], name='b_conv3')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，
    with tf.name_scope('conv2d_3'):
        conv2d_3 = conv2d(h_pool2, W_conv3) + b_conv3
    with tf.name_scope('relu3'):
        # relu2 = tf.nn.relu(conv2d_2)
        relu3 = tf.nn.leaky_relu(conv2d_3)
        # relu2 = tf.nn.elu(conv2d_2)
        # relu3 = tf.nn.sigmoid(conv2d_3)
    with tf.name_scope('h_pool3'):
        h_pool3 = max_pool_2x2(relu3)  # 进行max_pooling
    # with tf.name_scope('h_pool2_drop'):
    #     h_pool2 = tf.nn.dropout(h_pool2, keep_prob, name='h_pool2_drop')

with tf.name_scope('Conv4'):
    # 初始化第三个卷积层的权值和偏置
    with tf.name_scope('W_conv4'):
        W_conv4 = weight_variable([3, 3, 64, 64], name='W_conv4')  # 5*5的采样窗口，64个卷积核从64个平面抽取特征
    with tf.name_scope('b_conv4'):
        b_conv4 = bias_variable([64], name='b_conv4')  # 每一个卷积核一个偏置值

    # 把h_pool2和权值向量进行卷积，再加上偏置值，
    with tf.name_scope('conv2d_4'):
        conv2d_4 = conv2d(h_pool3, W_conv4) + b_conv4  #batch_size*8*8*64

    # 加入BN层，批标准化20190707
    with tf.name_scope('BN'):
        axis = list(range(len(conv2d_4.get_shape()) - 1))
        wb_mean, wb_var = tf.nn.moments(conv2d_4, axis)
        scale = tf.Variable(tf.ones([64]))
        offset = tf.Variable(tf.zeros([64]))
        variance_epsilon = 0.001
        conv2d_4_BN = tf.nn.batch_normalization(conv2d_4, wb_mean, wb_var, offset, scale, variance_epsilon)

    with tf.name_scope('relu4'):
        # relu3 = tf.nn.relu(conv2d_3)
        relu4 = tf.nn.leaky_relu(conv2d_4_BN)
        # relu3 = tf.nn.elu(conv2d_3)
        # relu3 = tf.nn.sigmoid(conv2d_3)
    with tf.name_scope('h_pool4'):
        h_pool4 = max_pool_2x2(relu4)  # 进行max_pooling
    # with tf.name_scope('h_pool3_drop'):
    #     h_pool3 = tf.nn.dropout(h_pool3, keep_prob, name='h_pool3_drop')

# 64*64的图片第一次卷积后还是256*128，第一次池化后变为128*64
# 第二次卷积后为128*64，第二次池化后变成了64*32
# 第三次卷积后为64*32，第三次池化后变成了32*16
# 第三次卷积后为32*16，第三次池化后变成了16*8
# 经过上面操作后得到64张16*8的平面

with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([16 * 8 * 64, 2048], name='W_fc1')  # 上一层有8*8*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([2048], name='b_fc1')  # 1024个节点

    # 把池化层3的输出扁平化为1维
    with tf.name_scope('h_pool1_flat'):
        h_pool1_flat = tf.reshape(h_pool4, [-1, 16 * 8 * 64], name='h_pool1_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

    with tf.name_scope('relu_fc1'):
    #     # h_fc1 = tf.nn.relu(wx_plus_b1)
        h_fc1 = tf.nn.leaky_relu(wx_plus_b1)
    #     # h_fc1 = tf.nn.elu(wx_plus_b1)
    #     # h_fc1 = tf.nn.sigmoid(wx_plus_b1)
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([2048, 512], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([512], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('relu_fc2'):
    #     # h_fc2 = tf.nn.relu(wx_plus_b2)
        h_fc2 = tf.nn.leaky_relu(wx_plus_b2)
    #     # h_fc2 = tf.nn.elu(wx_plus_b2)
    #     # h_fc2 = tf.nn.sigmoid(wx_plus_b2)
    with tf.name_scope('h_fc2_drop'):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob, name='h_fc2_drop')

with tf.name_scope('fc3'):
    # 初始化第三个全连接层
    with tf.name_scope('W_fc3'):
        W_fc3 = weight_variable([512, 4], name='W_fc3')
    with tf.name_scope('b_fc3'):
        b_fc3 = bias_variable([4], name='b_fc3')
    with tf.name_scope('wx_plus_b3'):
        wx_plus_b3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    with tf.name_scope('relu_fc3'):
    #     # h_fc2 = tf.nn.relu(wx_plus_b2)
        h_fc3 = tf.nn.leaky_relu(wx_plus_b3)
    #     # h_fc2 = tf.nn.elu(wx_plus_b2)
    #     # h_fc2 = tf.nn.sigmoid(wx_plus_b2)
    with tf.name_scope('h_fc3_drop'):
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob, name='h_fc3_drop')
    with tf.name_scope('softmax'):
        # 计算输出
        # prediction = tf.nn.softmax(wx_plus_b2)  # tf.nn.softmax_cross_entropy_with_logits 有softmax操作
        prediction = h_fc3_drop

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 运行了几轮batch_size的计数器，初值给0，设为不被训练
global_step = tf.Variable(0, trainable=False)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, decay_steps, decay_rate, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate, ).minimize(cross_entropy,global_step=global_step)  # 断点续训这里不加global_step=global_step会出错

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    print("启动执行...")
    sess.run(tf.global_variables_initializer())

    # 加入断点续训功能
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_writer = tf.summary.FileWriter('E:/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('E:/logs/test', sess.graph)

    # 读取数据
    train_data_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img4_features_train_1.npy")  # npy路径
    train_labels_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img4_labels_train_1.npy")  # npy路径

    test_data_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img4_features_test_1.npy")  # npy路径
    test_labels_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img4_labels_test_1.npy")  # npy路径

    train_data = np.load(train_data_npyPath).astype(np.float32)  # 加载数据
    train_labels = np.load(train_labels_npyPath).astype(np.float32)  # 加载数据

    test_data = np.load(test_data_npyPath).astype(np.float32)  # 加载数据
    test_labels = np.load(test_labels_npyPath).astype(np.float32)  # 加载数据

    step = 0
    while step < global_steps:
        # 训练模型
        # 记录训练集计算的参数
        flag = step
        if ((flag + 1) * batch_size > len(train_data)):
            flag = 0
        batch_xs, batch_ys = train_data[flag * batch_size:(flag + 1) * batch_size], train_labels[flag * batch_size:(flag + 1) * batch_size]  # 获取一块最小数据集

        batch_ys_new = InitImagesLabels(batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]

        _, loss, step = sess.run([train_step, cross_entropy, global_step],
                                 feed_dict={x: batch_xs, y: batch_ys_new, keep_prob: 0.5})
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys_new, keep_prob: 1.0})
        train_writer.add_summary(summary, step)

        T_lr = sess.run(learning_rate, feed_dict={global_step: step})

        # 计算测试集计算的参数
        flag2 = step
        if ((flag2 + 1) * batch_size > len(test_data)):
            flag2 = 0
        batch_xs, batch_ys = test_data[flag2 * batch_size:(flag2 + 1) * batch_size], test_labels[flag2 * batch_size:(flag2 + 1) * batch_size]  # 获取一块最小数据集
        batch_ys_new = InitImagesLabels(batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys_new, keep_prob: 1.0})
        test_writer.add_summary(summary, step)

        if step % 100 == 0:
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            print("step:",  step, ", loss:", loss, ", lr:", T_lr)

        if step % 500 == 0:
            # 训练集正确率
            icount_train_batchsize = int(len(train_data) / batch_size)
            sum_train_acc = 0
            for iNo in range(0, icount_train_batchsize):
                train_batch_xs, train_batch_ys = train_data[iNo * batch_size:(iNo + 1) * batch_size], train_labels[iNo * batch_size:(iNo + 1) * batch_size]  # 获取一块最小数据集
                train_labels_new = InitImagesLabels(train_batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
                train_acc = sess.run(accuracy, feed_dict={x: train_batch_xs, y: train_labels_new, keep_prob: 1.0})
                sum_train_acc = sum_train_acc + train_acc
            sum_train_acc = sum_train_acc / icount_train_batchsize
            print("Iter " + str(step) + ",Training Accuracy=" + str(sum_train_acc))

            # 测试集正确率
            icount_test_batchsize = int(len(test_data) / batch_size)
            sum_test_acc = 0
            for iNo in range(0, icount_test_batchsize):
                test_batch_xs, test_batch_ys = test_data[iNo * batch_size:(iNo + 1) * batch_size], test_labels[iNo * batch_size:(iNo + 1) * batch_size]  # 获取一块最小数据集
                test_labels_new = InitImagesLabels(test_batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
                test_acc = sess.run(accuracy, feed_dict={x: test_batch_xs, y: test_labels_new, keep_prob: 1.0})
                sum_test_acc = sum_test_acc + test_acc
            sum_test_acc = sum_test_acc / icount_test_batchsize
            print("Iter " + str(step) + ",Testing Accuracy=" + str(sum_test_acc))
