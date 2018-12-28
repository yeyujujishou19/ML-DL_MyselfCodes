#-*- coding:utf-8 -*
import os
import random
import captcha
import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image,ImageDraw,ImageFont,ImageFilter

#########全局变量###########################################
path = os.getcwd()  #项目所在路径
output_path = path + '/result/result.txt'   #测试结果存放路径
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "LSTM_Captcha"    # 保存模型名称

#要识别的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

batch_size = 64     # size of batch
time_steps = 26     # 每个time_step是图像的一行像素 height
n_input = 80        # rows of 28 pixels  #width
image_channels = 1  # 图像的通道数
captcha_num = 4     # 验证码中字符个数
n_classes = len(number) + len(ALPHABET)    #类别分类

learning_rate = 0.001   #learning rate for adam
decaystep = 5000  # 实现衰减的频率
decay_rate = 0.5  # 衰减率
num_units = 64   #hidden LSTM units
layer_num = 2   #网络层数
iteration = 20000   #训练迭代次数

#自动生成图像
IMAGE_HEIGHT = 26     # 图像高
IMAGE_WIDTH = 80      # 图像宽
#生成验证码图片的宽度和高度
size = (IMAGE_WIDTH,IMAGE_HEIGHT)
#背景颜色，默认为白色
bgcolor = (255,255,255)
#字体颜色，默认为黑色
fontcolor = (0,0,0)
#字体的位置，不同版本的系统会有不同BuxtonSketch.ttf
font_path = 'C:/Windows/Fonts/Georgia.ttf'
#########全局变量###########################################

# 随机生成4个数字+大小写字母的数组
def random_captcha_text(char_set=number+ALPHABET, captcha_size=4):  #数字
    captcha_text = []  # 初始化一个空列表
    for i in range(captcha_size):  # 产生字符的个数
        c = random.choice(char_set)  # 随机产生数字
        captcha_text.append(c)  # 加入列表
    return ''.join(captcha_text)  # 返回生成的字符

# 随机生成4个数字的图片
def gen_captcha_text_and_image():
    width,height = size #宽和高
    image = Image.new('RGBA',(width,height),bgcolor) #创建图片
    font = ImageFont.truetype(font_path,25) #验证码的字体
    draw = ImageDraw.Draw(image)  #创建画笔
    captcha_text = random_captcha_text()  # 随机生成4个数字的数组
    font_width, font_height = font.getsize(captcha_text) #字体大小
    draw.text(((width - font_width) / captcha_num, (height - font_height) / captcha_num),\
              captcha_text,font= font,fill=fontcolor) #填充字符串
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强
    # aa = str(".png")
    # path = "./" + captcha_text + aa
    # image.save(path)
    captcha_image = np.array(image)  # 转化成array数组
    return captcha_text, captcha_image

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
    if text_len > captcha_num:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(captcha_num*n_classes)

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
        idx = i * n_classes + char2pos(c)
        vector[idx] = 1
    return vector

# 0000100的数组转换成字符串
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % n_classes
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

# [22,32,1,5]类型转换成字符
def index2char(vec):
    text=[]
    chr=''
    for i in range(len(vec[0])):
        subVec=vec[0][i]
        listChr=[]
        for id in range(captcha_num):
            if subVec[id]<10:
                chr=number[subVec[id]]
                listChr.append(chr)
            elif subVec[id]<36:
                chr=ALPHABET[subVec[id]-10]
                listChr.append(chr)
            elif subVec[id] < 62:
                chr = ALPHABET[subVec[id] - 36]
                listChr.append(chr)
            elif subVec[id] == 62:
                listChr.append('_')
            else:
                raise ValueError('error')
        str=''.join(listChr)
        text.append(str)
    return text

# 产生用于训练的bacth_size0大小的数据集
def get_next_batch(batch_size0=64):
    batch_x = np.zeros([batch_size0, time_steps, n_input])
    batch_y = np.zeros([batch_size0, captcha_num, n_classes])

    # 内部定义一个用于产生图片和标签的函数
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 4):
                return text, image

    for i in range(batch_size0):  # 按batch_size0大小循环产生图片
        text, image = wrap_gen_captcha_text_and_image()  # 产生图片
        image = convert2gray(image)  # 转化成灰度图
        image = np.array(image)
        image=image/255
        # image = image.flatten() / 255  # image.flatten()是转化为一行，除以255是归一化
        # image = np.reshape(np.array(image), [IMAGE_HEIGHT, IMAGE_WIDTH])  # 转换格式：(2080,) => (26,80)
        batch_x[i] =image
        ss=text2vec(text)
        batch_y[i] = np.reshape(text2vec(text), [captcha_num,n_classes])# 转换为标签
    return batch_x, batch_y

#构建lstm网络
def computational_graph_lstm(x, y, global_step):
    #weights and biases of appropriate shape to accomplish above task
    out_weights = tf.Variable(tf.random_normal([num_units,n_classes]), name = 'out_weight')
    out_bias = tf.Variable(tf.random_normal([n_classes]),name = 'out_bias')

    #构建网络
    lstm_layer = [tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True) for _ in range(layer_num)]    #创建两层的lstm
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layer, state_is_tuple = True)   #将lstm连接在一起
    init_state = mlstm_cell.zero_state(batch_size, tf.float32)  #cell的初始状态

    outputs = list()    #每个cell的输出
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(time_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(x[:, timestep, :], state) # 这里的state保存了每一层 LSTM 的状态
            outputs.append(cell_output)
    # h_state = outputs[-1] #取最后一个cell输出

    #计算输出层的第一个元素
    prediction_1 = tf.nn.softmax(tf.matmul(outputs[-4],out_weights)+out_bias)   #获取最后time-step的输出，使用全连接, 得到第一个验证码输出结果
    #计算输出层的第二个元素
    prediction_2 = tf.nn.softmax(tf.matmul(outputs[-3],out_weights)+out_bias)   #输出第二个验证码预测结果
    #计算输出层的第三个元素
    prediction_3 = tf.nn.softmax(tf.matmul(outputs[-2],out_weights)+out_bias)   #输出第三个验证码预测结果
    #计算输出层的第四个元素
    prediction_4 = tf.nn.softmax(tf.matmul(outputs[-1],out_weights)+out_bias)   #输出第四个验证码预测结果,size:[batch,num_class]
    #输出连接
    prediction_all = tf.concat([prediction_1, prediction_2, prediction_3, prediction_4],1)  # 4 * [batch, num_class] => [batch, 4 * num_class]
    prediction_all = tf.reshape(prediction_all,[batch_size, captcha_num, n_classes],name ='prediction_merge') # [4, batch, num_class] => [batch, 4, num_class]

    #loss_function
    # 损失
    with tf.name_scope('loss'):  # 损失
        loss = -tf.reduce_mean(y * tf.log(prediction_all),name = 'loss')
        tf.summary.scalar('loss', loss)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_all,labels=y))
    #optimization
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name = 'opt').minimize(loss,global_step=global_step)  # 断点续训这里不加global_step=global_step会出错
    #model evaluation
    pre_arg = tf.argmax(prediction_all,2,name = 'predict')
    y_arg = tf.argmax(y,2)
    correct_prediction = tf.equal(pre_arg, y_arg)

    with tf.name_scope('accuracy'):  # 损失
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name = 'accuracy')
        tf.summary.scalar('accuracy', accuracy)

    return opt, loss, accuracy, pre_arg, y_arg

#训练
def train():
    # defining placeholders
    x = tf.placeholder("float",[None,time_steps,n_input], name = "x") #input image placeholder
    y = tf.placeholder("float",[None,captcha_num,n_classes], name = "y")  #input label placeholder

    # 运行了几轮batch_size的计数器，初值给0，设为不被训练
    global_step = tf.Variable(0, trainable=False)

    # 学习率自然指数衰减
    learing_rate_decay = tf.train.natural_exp_decay(learning_rate, global_step, decaystep, decay_rate, staircase=True)

    # computational graph
    opt, loss, accuracy, pre_arg, y_arg = computational_graph_lstm(x, y, global_step)

    # 创建训练模型保存类
    saver = tf.train.Saver(max_to_keep=1)

    # 初始化变量值
    init = tf.global_variables_initializer()

    # 将图形、训练过程等数据合并在一起
    merged = tf.summary.merge_all()

    with tf.Session() as sess:  # 创建tensorflow session
        sess.run(init)

        writer = tf.summary.FileWriter('logs', sess.graph)  # 将训练日志写入到logs文件夹下

        # ----------断点续训--------------------------
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # ----------断点续训--------------------------

        iter = 1 #迭代次数计数器
        while iter < iteration:
            batch_x, batch_y = get_next_batch(batch_size)
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})   #只运行优化迭代计算图


            if iter %100==0:
                result = sess.run(merged, feed_dict={x: batch_x, y: batch_y})  # 只运行优化迭代计算图
                writer.add_summary(result, iter)  # 将日志数据写入文件

                los, acc, parg, yarg, iter = sess.run([loss, accuracy, pre_arg, y_arg, global_step],feed_dict={x:batch_x,y:batch_y})
                print("iter:%d,Accuracy:%f,Loss:%f " % (iter, acc, los))

            if iter % 1000 == 0:   #保存模型
                # ----------指数衰减型学习率-------------------
                learning_rate_val = sess.run(learing_rate_decay)
                print("After %s steps,learing rate is %f" % (iter, learning_rate_val))
                # ----------指数衰减型学习率-------------------

                # ----------断点续训--------------------------
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                # ----------断点续训--------------------------

            iter += 1
        # 计算验证集准确率
        valid_x, valid_y = get_next_batch(batch_size)
        print("Validation Accuracy:", sess.run(accuracy, feed_dict={x: valid_x, y: valid_y}))

#预测
def predict():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(path + "/model/" + "LSTM_Captcha-19000.meta")
        saver.restore(sess, tf.train.latest_checkpoint(path + "/model/")) #读取已训练模型

        graph = tf.get_default_graph()  #获取原始计算图，并读取其中的tensor
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        pre_arg = graph.get_tensor_by_name("predict:0")

        # test_x, file_list = get_test_set()  #获取测试集
        test_x, test_y =get_next_batch(batch_size)
        batch_test_y = np.zeros([batch_size, captcha_num, n_classes])  # 创建空的y输入
        test_predict = sess.run([pre_arg], feed_dict={x: test_x, y: batch_test_y})
        predict_result=index2char(np.array(test_predict)) #转成字符串
        predict_result = predict_result[:len(test_y)]     #预测结果
        write_to_file(predict_result, test_y)             #保存到文件

#预测结果写入文档
def write_to_file(predict_list, test_y):
    label_y = np.reshape(test_y, [batch_size, captcha_num * n_classes])
    with open(output_path, 'w') as f:
        for i, res in enumerate(predict_list):
            y_ = vec2text(label_y[i]) #转成字符串
            if i == 0:
                f.write("id\tfile\tresult\n")
            f.write(str(i) + "\t" + y_ + "\t" + res + "\n")
            f.write("\n")
    print("预测结果保存在：",output_path)

#训练
train()

#预测
# predict()