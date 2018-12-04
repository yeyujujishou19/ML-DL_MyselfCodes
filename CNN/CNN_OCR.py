# !/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

n_label = 10  # 标签维度
# 每个批次的大小
batch_size = 5000


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
        aa = np.zeros(n_label, np.int16)
        aa[id] = 1
        labels_batch_new.append(aa)
    return labels_batch_new


# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        # 改变x的格式转为4D的向量[batch,in_height,in_width,in_channels]
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # 进行max_pooling

with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # 进行max_pooling

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
# 第二次卷积后为14*14，第二次池化后变成了7*7
# 经过上面操作后得到64张7*7的平面

with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')  # 上一层有7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点

    # 把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # keep_prob用来表示神经元的输出概率
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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

with tf.Session() as sess:
    print("启动执行...")
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('E:/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('E:/logs/test', sess.graph)

    for i in range(1001):

        # 读取数据
        train_data_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img10_features_train_1.npy")  # npy路径
        train_labels_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img10_labels_train_1.npy")  # npy路径

        test_data_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img10_features_test_1.npy")  # npy路径
        test_labels_npyPath = (r"E:/sxl_Programs/Python/CNN/npy/Img10_labels_test_1.npy")  # npy路径

        train_data = np.load(train_data_npyPath).astype(np.int16)  # 加载数据
        train_labels = np.load(train_labels_npyPath).astype(np.int16)  # 加载数据

        test_data = np.load(test_data_npyPath).astype(np.int16)  # 加载数据
        test_labels = np.load(test_labels_npyPath).astype(np.int16)  # 加载数据

        # 训练模型
        # 记录训练集计算的参数
        flag = i
        if ((flag + 1) * batch_size > len(train_data)):
            flag = 0
        batch_xs, batch_ys = train_data[flag * batch_size:(flag + 1) * batch_size], train_labels[flag * batch_size:(
                                                                                                                               flag + 1) * batch_size]  # 获取一块最小数据集
        batch_ys_new = InitImagesLabels(batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys_new, keep_prob: 0.5})
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys_new, keep_prob: 1.0})
        train_writer.add_summary(summary, i)

        # 计算测试集计算的参数
        flag2 = i
        if ((flag2 + 1) * batch_size > len(test_data)):
            flag2 = 0
        batch_xs, batch_ys = test_data[flag2 * batch_size:(flag2 + 1) * batch_size], test_labels[flag2 * batch_size:(
                                                                                                                                flag2 + 1) * batch_size]  # 获取一块最小数据集
        batch_ys_new = InitImagesLabels(batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys_new, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 10 == 0:
            print(i)

        if i % 100 == 0:
            # 训练集正确率
            icount_train_batchsize = int(len(train_data) / batch_size)
            sum_train_acc = 0
            for iNo in range(0, icount_train_batchsize):
                train_batch_xs, train_batch_ys = train_data[iNo * batch_size:(iNo + 1) * batch_size], train_labels[
                                                                                                      iNo * batch_size:(
                                                                                                                                   iNo + 1) * batch_size]  # 获取一块最小数据集
                train_labels_new = InitImagesLabels(train_batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
                train_acc = sess.run(accuracy, feed_dict={x: train_batch_xs, y: train_labels_new, keep_prob: 1.0})
                sum_train_acc = sum_train_acc + train_acc
            sum_train_acc = sum_train_acc / icount_train_batchsize
            print("Iter " + str(i) + ",Training Accuracy=" + str(sum_train_acc))

            # 测试集正确率
            icount_test_batchsize = int(len(test_data) / batch_size)
            sum_test_acc = 0
            for iNo in range(0, icount_test_batchsize):
                test_batch_xs, test_batch_ys = test_data[iNo * batch_size:(iNo + 1) * batch_size], test_labels[
                                                                                                   iNo * batch_size:(
                                                                                                                                iNo + 1) * batch_size]  # 获取一块最小数据集
                test_labels_new = InitImagesLabels(test_batch_ys)  # 从数字标签转换为数组标签 [0,0,0,...1,0,0]
                test_acc = sess.run(accuracy, feed_dict={x: test_batch_xs, y: test_labels_new, keep_prob: 1.0})
                sum_test_acc = sum_test_acc + test_acc
            sum_test_acc = sum_test_acc / icount_test_batchsize
            print("Iter " + str(i) + ",Testing Accuracy=" + str(sum_test_acc))
