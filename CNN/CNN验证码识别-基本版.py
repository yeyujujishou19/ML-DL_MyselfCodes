import captcha
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = r"./model/" # 模型保存路径
MODEL_NAME = "CNN_Captcha"    # 保存模型名称
IMAGE_HEIGHT = 60             # 图像高
IMAGE_WIDTH = 160             # 图像宽
training_iters = 3000         # 训练次数
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

# 随机生成4个数字的数组
#def random_captcha_text(char_set=number+ alphabet + ALPHABET, captcha_size=4): #数字+大小写字母
def random_captcha_text(char_set=number, captcha_size=4):  #数字
    captcha_text = []  # 初始化一个空列表
    for i in range(captcha_size):  # 产生字符的个数
        c = random.choice(char_set)  # 随机产生数字
        captcha_text.append(c)  # 加入列表
    return captcha_text  # 返回生成的字符

# 随机生成4个数字的图片
def gen_captcha_text_and_image():
    image = ImageCaptcha()  # 生成验证码图片,使用之前先实例化
    captcha_text = random_captcha_text()  # 随机生成4个数字的数组
    captcha_text = ''.join(captcha_text)  # 将数组转成字符串
    captcha = image.generate(captcha_text)  # 根据随机产生的字符串生成验证码图片
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)  # 打开图像
    captcha_image = np.array(captcha_image)  # 转化成array数组
    return captcha_text, captcha_image

text, image = gen_captcha_text_and_image()  # 随机生成4个数字的图片
print("验证码图像channel:", image.shape)  # (60, 160, 3)
MAX_CAPTCHA = len(text)  # 验证码长度
#char_set = number+ alphabet + ALPHABET + ['_'] #数字+大小写字母
char_set = number+ ['_'] #数字
CHAR_SET_LEN = len(char_set)

# 转换成灰度图
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

# 字符串转换成0000100的数组
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

# 0000100的数组转换成字符串
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

# 产生用于训练的bacth_size大小的数据集
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 内部定义一个用于产生图片和标签的函数
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):  # 按batch_size大小循环产生图片
        text, image = wrap_gen_captcha_text_and_image()  # 产生图片
        image = convert2gray(image)  # 转化成灰度图

        batch_x[i, :] = image.flatten() / 255  # image.flatten()是转化为一行，除以255是归一化
        batch_y[i, :] = text2vec(text)  # 转换为标签

    return batch_x, batch_y

# 定义卷积神经网络
def crack_captcha_cnn(X, Y, keep_prob, w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # 恢复成图像的宽高

    # 3层卷积网络
    # 第一层卷积 转化为30*80*32维数据
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))  # 权重 #3*3的采样窗口，32个卷积核从1个平面抽取特征
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))  # 偏置
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # 第二层卷积 转化为15*40*64维数据
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 第三层卷积 转化为8*20*64维数据
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 全连接层
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

# 训练
def train_crack_captcha_cnn():

    # 变量占位
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)  # dropout防止过拟合

    # 运行了几轮batch_size的计数器，初值给0，设为不被训练
    global_step = tf.Variable(0, trainable=False)

    # 定义卷积神经网络
    output = crack_captcha_cnn(X, Y, keep_prob)

    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,global_step=global_step)  # 断点续训这里不加global_step=global_step会出错

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])            #预测结果
    max_idx_p = tf.argmax(predict, 2)                                        #求预测的最大值下标
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2) #求标签的下标
    correct_pred = tf.equal(max_idx_p, max_idx_l)                            #求正确率
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))             #求准确率均值

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        print("启动执行...")
        time_start = time.time()  # 计时
        sess.run(tf.global_variables_initializer())  # 初始化全局变量
        sess.run(tf.local_variables_initializer())   # 初始化局部变量

        # ----------断点续训--------------------------
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path: #如果已有模型，则加载
            saver.restore(sess, ckpt.model_checkpoint_path)
        # ----------断点续训--------------------------
        step = 0
        while step < training_iters:  # training_iters 训练次数

            batch_x, batch_y = get_next_batch(64)  # 产生一组数据
            _, loss_, step= sess.run([optimizer, loss, global_step],feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            step += 1  # 执行轮数
            if step % 100 == 0:
                print("当前轮数%s,loss:%s"%(step, loss_))  # 打印损失函数值
                # ----------断点续训--------------------------
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                # ----------断点续训--------------------------

                batch_x_test, batch_y_test = get_next_batch(100)  # 产生一组测试数据
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("准确率=", acc)
                if acc > 0.95:
                    saver.save(sess, "model/AccAbove9_capcha.model", global_step=step)
        time_end = time.time()
        time_h = (time_end - time_start) / 3600
        print('训练用时：%.2f 小时' % time_h)

        #==========测试------------------------------
        count = 0
        all_count = 100
        for i in range(all_count):
            text, image = gen_captcha_text_and_image()  # 随机生成4个数字的图片
            gray_image = convert2gray(image)  # 转化成灰度图
            captcha_image = gray_image.flatten() / 255  # image.flatten()是转化为一行，除以255是归一化

            text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})  # 预测
            prediction_labels = np.argmax(text_list, axis=2)  # 转化成数字标签
            predict_text = prediction_labels[0].tolist()  # 转化成列表
            predict_text = str(predict_text)  # 转化成字符串
            predict_text = predict_text.replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
            if text == predict_text:  # 如果预测结果和标签相等，表明预测正确
                count += 1
                check_result = "，预测结果正确"
            else:  # 不正确
                #                 f = plt.figure() #产生一张图
                #                 ax = f.add_subplot(111)
                #                 ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
                #                 plt.imshow(image) #显示图像
                #                 plt.show()
                check_result = "，预测结果错误"
                print("正确: {}  预测: {}".format(text, predict_text) + check_result)
        print("正确率:", count, "/", all_count)


# 训练
train_crack_captcha_cnn()



