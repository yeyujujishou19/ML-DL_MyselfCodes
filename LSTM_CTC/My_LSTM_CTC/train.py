# -*- coding: utf-8 -*-
# LSTM+CTC_loss训练识别不定长数字字符图片
from CaptchaGenerateTextImage import GenerateCaptchaListImage
from freeTypeGenerateTextImage import GenerateCharListImage
import tensorflow as tf
import numpy as np
import time
import os

tf.reset_default_graph() #重置default graph计算图以及nodes节点

#========超参数====================
#要生成的图片的像素点大小
image_shape=(40,120)
#隐藏层神经元数量
num_hidden=64
#初始化学习率和学习率衰减因子
lr_start=1e-3
lr_decay_factor=0.9
lr_decay_step=2000   #每两千步衰减一次
#一批训练样本和测试样本的样本数量
batch_size=64
iteration=10000     #迭代次数
report_step=100    #打印信息间隔
#用来恢复标签用的候选字符集
char_set=['0','1','2','3','4','5','6','7','8','9']
#设定准确率达到多少后停止训练
acc_reach_to_stop=0.96
#模型保存路径
MODEL_SAVE_PATH = "./free_type_image_lstm_model/"
#模型名称
MODEL_NAME="LSTM_CTC"

obj_number=GenerateCharListImage()
#类别为10位数字+blank+ctc blank
num_classes=obj_number.len+1+1

# 使用freetype库生成一批样本
def free_type_get_next_batch(bt_size,img_shape):
    obj_batch=GenerateCharListImage()
    bt_x_inputs=np.zeros([bt_size,image_shape[1],image_shape[0]])
    bt_y_inputs=[]
    for i in range(bt_size):
        #生成不定长度的字符串及其对应的彩色图片
        color_image,text,text_vector=obj_batch.generate_color_image(img_shape,noise="gaussian")
        #图片降噪，然后由彩色图片生成灰度图片的一位数组形式
        color_image=obj_batch.image_reduce_noise(color_image)
        #转成灰度图
        gray_image_array=obj_batch.color_image_to_gray_image(color_image)
        #然后将这个图片的数据写入bt_x_inputs中第0个维度上的第i个元素(每个元素就是一张图片的所有数据)
        bt_x_inputs[i,:]=np.transpose(gray_image_array.reshape((image_shape[0],image_shape[1])))
        #存入标签
        bt_y_inputs.append(list(text))
    #将bt_y_inputs中的每个元素都转化为np数组
    targets=[np.asarray(i) for i in bt_y_inputs]
    #将targets列表转化为稀疏矩阵
    sparse_matrix_targets=sparse_tuple_from(targets)
    cha_list=decode_sparse_tensor(sparse_matrix_targets)
    #seq_length就是每个样本中有多少个时间序列
    seq_length=np.ones(bt_x_inputs.shape[0])*image_shape[1]

    return bt_x_inputs, sparse_matrix_targets,seq_length

# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    param:sequences:列表，里面的元素也是列表
    param:dtype:列表元素的数据类型
    return:返回一个元祖(indices,values,shape)
    """
    indices=[]
    values=[]

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq),range(len(seq))))
        values.extend(seq)

    # indices:二维int64的矩阵，代表元素在batch样本矩阵中的位置
    # values:二维tensor，代表indice位置的数据值
    # dense_shape:一维，代表稀疏矩阵的大小
    # 假设sequences有2个，值分别为[1 3 4 9 2]、[ 8 5 7 2]。(即batch_size=2）
    # 则其indices=[[0 0][0 1][0 2][0 3][0 4][1 0][1 1][1 2][1 3]]
    # values=[1 3 4 9 2 8 5 7 2]
    # shape=[2 5]
    indices=np.asarray(indices,dtype=np.int64)
    values=np.asarray(values,dtype=dtype)
    shape=np.asarray([len(sequences),np.asarray(indices).max(0)[1]+1],dtype=np.int64)

    return indices,values,shape

#由稀疏矩阵转换为字符串
def decode_sparse_tensor(sparse_tensor):
    decoded_indexes=list()
    current_i=0
    current_seq=[]
    # i_and_index即sparse_tensor[0]也就是indices中的每个元素，i_and_index[0]即sparse_tensor[0]中每个元素属于第几号样本
    for offset,i_and_index in enumerate(sparse_tensor[0]):
        # i记录现在遍历到的sparse_tensor[0]元素属于第几号样本
        i=i_and_index[0]
        if i!=current_i:
            # 每次属于同一个样本的sparse_tensor[0]元素遍历完以后，decoded_indexes添加这个样本的完整current_seq
            decoded_indexes.append(current_seq)
            # 更新i
            current_i=i
            # 对这样新编号的样本建立一个新的current_seq
            current_seq=list()
        # current_seq记录我们现在遍历到的sparse_tensor[0]元素在这批样本中的位置(下标)
        current_seq.append(offset)
    # for循环遍历完以后，添加最后一个样本的current_seq到decoded_indexes，这样decoded_indexes就记录了这批样本中所有样本的current_seq
    decoded_indexes.append(current_seq)
    result=[]
    # 遍历decoded_indexes，依次解码每个样本的字符串内容
    # 实际上decoded_indexes就是记录了一批样本中每个样本中的所有字符在这批样本中的位置(下标)
    for index in decoded_indexes:
        result.append(decode_a_seq(index,sparse_tensor))
    # result记录了这批样本中每个样本的字符串内容，result的每个元素就是一个样本的字符串的内容
    # 这个元素是一个列表，列表每个元素是一个单字符
    return result

#根据下标获取spars_tensor[1]中的字符
def decode_a_seq(indexes,spars_tensor):
    decoded=[]
    for m in indexes:
        ch=char_set[spars_tensor[1][m]]
        decoded.append(ch)
    return decoded

#定义训练模型
def get_train_model():
    x_inputs=tf.placeholder(tf.float32,[None,None,image_shape[0]])
    # 定义ctc_loss需要的标签向量(稀疏矩阵形式)
    targets=tf.sparse_placeholder(tf.int32)
    # 每个样本中有多少个时间序列
    seq_length=tf.placeholder(tf.int32,[None])
    # 定义LSTM网络的cell层，这里定义有num_hidden个单元
    cell=tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
    outputs,_=tf.nn.dynamic_rnn(cell,x_inputs,seq_length,time_major=False,dtype=tf.float32)
    # x_inputs的shape=[batch_size,image_shape[1],image_shape[0]]
    shape=tf.shape(x_inputs)
    batch_s,max_time_steps=shape[0],shape[1]
    # reshape后的shape=[batch_size*max_time_step,num_hidden]
    outputs=tf.reshape(outputs,[-1,num_hidden])
    # 相当于一个全连接层，做一次线性变换
    w=tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1),name="w")
    b=tf.Variable(tf.constant(0.,shape=[num_classes]),name="b")
    logits=tf.matmul(outputs,w)+b
    logits=tf.reshape(logits,[batch_s,-1,num_classes])
    # logits的维度交换，第1个维度和第0个维度互相交换
    logits=tf.transpose(logits,(1,0,2))
    # 注意返回的logits预测标签此时还未压缩，而targets真实标签是被压缩过的
    return  logits,x_inputs,targets,seq_length,w,b

#计算准确率
def report_accuracy(decoded_list,test_targets):
    original_list=decode_sparse_tensor(test_targets)
    detected_list=decode_sparse_tensor(decoded_list)
    if len(original_list) != len(detected_list):
        return 0  #正确率为0
    #本批样本中预测正确的次数
    correct_prediction=0
    print("真实标签（长度）  <--------> 预测标签（长度）")
    for idx,true_number in enumerate(original_list):
        detect_number=detected_list[idx]
        signal=(true_number==detect_number)
        print(signal,true_number,"(",len(true_number),")  <-------->  ",detect_number,"(",len(detect_number),")")
        #如果相等，统计正确的预测次数加1
        if signal is True:
            correct_prediction+=1
    #计算本批样本预测的正确率
    acc=correct_prediction*1.0/len(original_list)
    print("本批样本预测准确率：{}".format(acc))
    return acc

def train():
    global_step=tf.Variable(0,trainable=False)
    #学习率
    learning_rate=tf.train.exponential_decay(lr_start,global_step,lr_decay_step,lr_decay_factor)
    #获得模型相关参数
    logits,inputs,targets,seq_len,w,b=get_train_model()
    #损失函数
    cost=tf.reduce_mean(tf.nn.ctc_loss(labels=targets,inputs=logits,sequence_length=seq_len))
    #优化器
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)
    #对输入中给出的logits执行波束搜索解码。解码成非压缩状态
    decoded,log_prob=tf.nn.ctc_beam_search_decoder(logits,seq_len,merge_repeated=False)
    #与标签对比，得出正确率
    accuracy1=tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0],tf.int32),targets))
    print('accuracy:',accuracy1)

    #产生一个数据集，测试正确率
    def do_report():
        #产生测试集
        test_inputs,test_targets,test_seq_len=free_type_get_next_batch(batch_size,image_shape)

        test_feed={inputs:test_inputs,
                   targets:test_targets,
                   seq_len:test_seq_len}
        dd,log_probs,accuracy=sess.run([decoded[0],log_prob,accuracy1],test_feed)
        report_acc=report_accuracy(dd,test_targets)
        #返回准确率
        return report_acc

    #产生一个数据集，用于训练
    def do_batch():
        train_inputs,train_targets,train_seq_len=free_type_get_next_batch(batch_size,image_shape)

        train_feed={inputs:train_inputs,targets:train_targets,seq_len:train_seq_len}
        b_cost,b_lr,b_acc,steps,_=sess.run([cost,learning_rate,accuracy1,global_step,optimizer],feed_dict=train_feed)

        return b_cost,steps,b_acc,b_lr

    #创建模型文件保存路径
    if not os.path.exists( MODEL_SAVE_PATH):
        os.mkdir( MODEL_SAVE_PATH)
    saver=tf.train.Saver(max_to_keep=1)
    #创建会话，开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # ----------断点续训--------------------------
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # ----------断点续训--------------------------

        train_steps=0
        while train_steps<iteration:
            start=time.time() #计时开始
            #每一轮将一个batch的样本喂进去训练
            batch_cost,train_steps,acc,batch_lr=do_batch()


            if train_steps%report_step==0:

                # ----------断点续训--------------------------
                # saver.save(sess, "./free_type_image_lstm_model/train_model", global_step=train_steps)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=train_steps)
                # ----------断点续训--------------------------

                acc=do_report()
                # if(acc>acc_reach_to_stop):
                #     print("准确率已达到临界值{}，目前准确率{}，停止训练".format(acc_reach_to_stop,acc))
                #     # break
                batch_seconds = time.time() - start #计时结束
                log = "iteration:{},acc:{:.6f},batch_cost:{:.6f},batch_learning_rate:{:.6f},batch seconds:{:.6f}"
                print(log.format(train_steps,acc, batch_cost, batch_lr, batch_seconds))

if __name__ == '__main__':
	train()