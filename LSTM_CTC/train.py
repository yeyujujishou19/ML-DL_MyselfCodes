# -*- coding: utf-8 -*-
# LSTM+CTC_loss训练识别不定长数字字符图片
from CaptchaGenerateTextImage import GenerateCaptchaListImage
from freeTypeGenerateTextImage import GenerateCharListImage
import tensorflow as tf
import numpy as np
import time
import os

tf.reset_default_graph()  #重置default graph计算图以及nodes节点
 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
 
# 超参数
# 要生成的图片的像素点大小
char_list_image_shape = (40, 120)
# 隐藏层神经元数量
num_hidden = 64
# 初始学习率和学习率衰减因子
lr_start = 1e-3
lr_decay_factor = 0.9
# 一批训练样本和测试样本的样本数量，训练迭代次数，每经过test_report_step_interval测试一次模型预测的准确率
train_batch_size = 64
test_batch_size = 64
train_iteration = 5000
test_report_step_interval = 100
# 用来恢复标签用的候选字符集
char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#char_set='0123456789'
# 如果用freetype生成的验证码，则为True；如果是用captcha生成的验证码，则为False
use_freeType_or_captcha = True
# 设定准确率达到多少后停止训练
acc_reach_to_stop = 0.95
 
obj_number = GenerateCharListImage()
# 类别为10位数字+blank+ctc blank
num_classes = obj_number.len + 1 + 1
 
 
# 生成batch_size个样本，样本的shape变为[batch_size,image_shape[1],image_shape[0]]
# 输入的图片是把每一行的数据看成一个时间间隔t内输入的数据，然后有多少行就是有多少个时间间隔
# 使用freetype库生成一批样本
def free_type_get_next_batch(bt_size, img_shape):
	obj_batch = GenerateCharListImage()
	bt_x_inputs = np.zeros([bt_size, char_list_image_shape[1], char_list_image_shape[0]])# 此处不同
	bt_y_inputs = []
	for i in range(bt_size):
		# 生成不定长度的字符串及其对应的彩色图片
		color_image, text, text_vector = obj_batch.generate_color_image(img_shape, noise="gaussian")
		# 图片降噪，然后由彩色图片生成灰度图片的一维数组形式
		color_image = obj_batch.image_reduce_noise(color_image)
		gray_image_array = obj_batch.color_image_to_gray_image(color_image)
		# np.transpose函数将得到的图片矩阵转置成(image_shape[1]，image_shape[0])形状的矩阵，且由行有序变成列有序
		# 然后将这个图片的数据写入bt_x_inputs中第0个维度上的第i个元素(每个元素就是一张图片的所有数据)
		bt_x_inputs[i, :] = np.transpose(gray_image_array.reshape((char_list_image_shape[0], char_list_image_shape[1])))
		# 把每个图片的标签添加到bt_y_inputs列表，注意这里直接添加了图片对应的字符串
		bt_y_inputs.append(list(text))
	# 将bt_y_inputs中的每个元素都转化成np数组
	targets = [np.asarray(i) for i in bt_y_inputs]
	# 将targets列表转化为稀疏矩阵
	sparse_matrix_targets = sparse_tuple_from(targets)
	# bt_size个1乘以char_list_image_shape[1]，也就是batch_size个样本中每个样本（每个样本即图片）的长度上的像素点个数（或者说列数）
	# seq_length就是每个样本中有多少个时间序列
	seq_length = np.ones(bt_x_inputs.shape[0]) * char_list_image_shape[1]
	# 得到的bt_x_inputs的shape=[bt_size, char_list_image_shape[1], char_list_image_shape[0]]
	return bt_x_inputs, sparse_matrix_targets, seq_length
 
 
# 使用captcha库生成一批样本
def captcha_get_next_batch(bt_size, img_shape):
	obj_batch = GenerateCaptchaListImage()
	bt_x_inputs = np.zeros([bt_size, char_list_image_shape[1], char_list_image_shape[0]])
	bt_y_inputs = []
	for i in range(bt_size):
		# 生成不定长度的字符串及其对应的彩色图片
		color_image, text, text_vector = obj_batch.generate_color_image(img_shape)
		# 图片降噪，然后由彩色图片生成灰度图片的一维数组形式
		color_image = obj_batch.image_reduce_noise(color_image)
		gray_image_array = obj_batch.color_image_to_gray_image(color_image)
		# np.transpose函数将得到的图片矩阵转置成(image_shape[1]，image_shape[0])形状的矩阵，且由行有序变成列有序
		# 然后将这个图片的数据写入bt_x_inputs中第0个维度上的第i个元素(每个元素就是一张图片的所有数据)
		bt_x_inputs[i, :] = np.transpose(gray_image_array.reshape((char_list_image_shape[0], char_list_image_shape[1])))
		# 把每个图片的标签添加到bt_y_inputs列表，注意这里直接添加了图片对应的字符串
		bt_y_inputs.append(list(text))
	# 将bt_y_inputs中的每个元素都转化成np数组
	targets = [np.asarray(i) for i in bt_y_inputs]
	# 将targets列表转化为稀疏矩阵
	sparse_matrix_targets = sparse_tuple_from(targets)
	# bt_size个1乘以char_list_image_shape[1]，也就是batch_size个样本中每个样本（每个样本即图片）的长度上的像素点个数（或者说列数）
	seq_length = np.ones(bt_x_inputs.shape[0]) * char_list_image_shape[1]
	# 得到的bt_x_inputs的shape=[bt_size, char_list_image_shape[1], char_list_image_shape[0]]
	return bt_x_inputs, sparse_matrix_targets, seq_length
 
 
# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
	"""
	:param sequences: 一个元素是列表的列表
	:param dtype: 列表元素的数据类型
	:return: 返回一个元组(indices, values, shape)
	"""
	indices = []
	values = []
 
	for n, seq in enumerate(sequences):
		# sequences存储了你的样本对应的字符串(由数字组成)的所有数字
		# 每次取list中的一个元素，即一个数字，代表的是一个样本(即一个字符串)中的一个数字值，注意这个单独的数字是也是一个列表
		# extend()函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
		# zip()函数将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的一个对象。
		# zip(a,b)函数分别从a和b中取一个元素组成元组，再次将组成的元组组合成一个新的迭代器。a与b的维数相同时，正常组合对应位置的元素。
		# 每个seq是一个字符串，index即这是第几个字符串(第几个样本)
		indices.extend(zip([n] * len(seq), range(len(seq))))
		# [index]的值为[0]、[1]、[2]。。。，len(seq)为每个字符串的长度
		# 如[1]*4的结果是[1, 1, 1, 1]
		# * 操作符在实现上是复制了值的引用，而不是创建了新的对象。所以上述的list里面，是4个指向同一个对象的引用，所以4个值都是1
		values.extend(seq)
 
	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
	# indices:二维int64的矩阵，代表元素在batch样本矩阵中的位置
	# values:二维tensor，代表indice位置的数据值
	# dense_shape:一维，代表稀疏矩阵的大小
	# 假设sequences有2个，值分别为[1 3 4 9 2]、[ 8 5 7 2]。(即batch_size=2）
	# 则其indices=[[0 0][0 1][0 2][0 3][0 4][1 0][1 1][1 2][1 3]]
	# values=[1 3 4 9 2 8 5 7 2]
	# shape=[2 5]
	return indices, values, shape
 
 
# 解压缩压缩过的所有样本的字符串的列表的集合，return为不压缩的所有样本的字符串的列表的集合
def decode_sparse_tensor(sparse_tensor):
	decoded_indexes = list()
	current_i = 0
	current_seq = []
	# sparse_tensor[0]即sparse_tuple_from函数的返回值中的indices
	# 这里是一批样本的字符串的列表集合经过sparse_tuple_from函数处理后的返回值中的indices
	# offset即indices中元素的下标，即indices中的第几个元素(每个元素是一个单字符，代表这个单字符在这批样本中的位置)
	# i_and_index即sparse_tensor[0]也就是indices中的每个元素，i_and_index[0]即sparse_tensor[0]中每个元素属于第几号样本
	for offset, i_and_index in enumerate(sparse_tensor[0]):
		# i记录现在遍历到的sparse_tensor[0]元素属于第几号样本
		i = i_and_index[0]
		# 如果新遍历到的sparse_tensor[0]元素和前一个元素不属于同一个样本
		if i != current_i:
			# 每次属于同一个样本的sparse_tensor[0]元素遍历完以后，decoded_indexes添加这个样本的完整current_seq
			decoded_indexes.append(current_seq)
			# 更新i
			current_i = i
			# 对这样新编号的样本建立一个新的current_seq
			current_seq = list()
		# current_seq记录我们现在遍历到的sparse_tensor[0]元素在这批样本中的位置(下标)
		current_seq.append(offset)
	# for循环遍历完以后，添加最后一个样本的current_seq到decoded_indexes，这样decoded_indexes就记录了这批样本中所有样本的current_seq
	decoded_indexes.append(current_seq)
	result = []
	# 遍历decoded_indexes，依次解码每个样本的字符串内容
	# 实际上decoded_indexes就是记录了一批样本中每个样本中的所有字符在这批样本中的位置(下标)
	for index in decoded_indexes:
		result.append(decode_a_seq(index, sparse_tensor))
	# result记录了这批样本中每个样本的字符串内容，result的每个元素就是一个样本的字符串的内容
	# 这个元素是一个列表，列表每个元素是一个单字符
	return result
 
 
# 将压缩过的所有样本的字符串的列表的集合spars_tensor中取出第indexes号样本中的所有字符在这个样本中的位置(下标)，解压缩成字符串
def decode_a_seq(indexes, spars_tensor):
	decoded = []
	# indexes是decoded_indexes中第indexes号样本中的所有字符在这批样本中的位置(下标)
	# for循环取出的m就是这个样本中每个字符在这批样本中的位置(下标)
	for m in indexes:
		# spars_tensor[1][m]即spars_tensor中的values列表的第m个值
		# ch即取出了m对应的spars_tensor中的values列表的第m个值，是一个字符
		ch = char_set[spars_tensor[1][m]]
		# 把这个字符加到decoded列表中
		decoded.append(ch)
	# decoded列表即存储一个样本中的所有字符
	return decoded
 
 
# 定义训练模型
def get_train_model():
	x_inputs = tf.placeholder(tf.float32, [None, None, char_list_image_shape[0]])
	# inputs的维度是[batch_size,num_steps,input_dim]
	# 定义ctc_loss需要的标签向量(稀疏矩阵形式)
	targets = tf.sparse_placeholder(tf.int32)
	# 每个样本中有多少个时间序列
	seq_length = tf.placeholder(tf.int32, [None])
	# 定义LSTM网络的cell层，这里定义有num_hidden个单元
	# cell_multilayer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=size) for size in [64, 128]],state_is_tuple=True)
	cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
	# state_is_tuple:如果为True，接受和返回的states是n-tuples，其中n=len(cells)。
	# 如果cell选择了state_is_tuple=True，那final_state是个tuple，分别代表Ct和ht，其中ht与outputs中的对应的最后一个时刻的输出ht相等；
	# 如果time_major == False(default)，输出张量形如[batch_size, max_time, cell.output_size]。
	# 如果time_major == True, 输出张量形如：[max_time, batch_size, cell.output_size]。
	# cell.output_size其实就是我们的num_hidden，即cell层的神经元的个数。
	outputs, _ = tf.nn.dynamic_rnn(cell, x_inputs, seq_length, time_major=False, dtype=tf.float32)
	# ->[batch_size,max_time_step,num_features]->lstm
	# ->[batch_size,max_time_step,cell.output_size]->reshape
	# ->[batch_size*max_time_step,num_hidden]->affine projection AW+b
	# ->[batch_size*max_time_step,num_classes]->reshape
	# ->[batch_size,max_time_step,num_classes]->transpose
	# ->[max_time_step,batch_size,num_classes]
	# 上面最后的shape就是标签向量的shape,此时标签向量还未压缩
 
	shape = tf.shape(x_inputs)
	# x_inputs的shape=[batch_size,image_shape[1],image_shape[0]]
	# 所以输入的数据是按列来排的，一列的像素为一个时间序列里输入的数据，一共120个时间序列
	batch_s, max_time_steps = shape[0], shape[1]
	# 输出的outputs为num_hidden个隐藏层单元的所有时刻的输出
	# reshape后的shape=[batch_size*max_time_step,num_hidden]
	outputs = tf.reshape(outputs, [-1, num_hidden])
	# 相当于一个全连接层，做一次线性变换
	w = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="w")
	b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
 
	logits = tf.matmul(outputs, w) + b
	# 变换成和标签向量一致的shape
	logits = tf.reshape(logits, [batch_s, -1, num_classes])
	# logits的维度交换，第1个维度和第0个维度互相交换
	logits = tf.transpose(logits, (1, 0, 2))
	# 注意返回的logits预测标签此时还未压缩，而targets真实标签是被压缩过的
	return logits, x_inputs, targets, seq_length, w, b
 
 
# test_targets即用sparse_tuple_from压缩过的所有样本的字符串的一个列表的集合，decoded_list也是一样
def report_accuracy(decoded_list, test_targets):
	# 将压缩的真实标签和预测标签解压缩，解压缩后都是一个列表，列表中存储了这批样本中的所有字符串。
	# 列表中的每个元素都是一个列表，这个列表中包含一个样本中的所有字符。
    print('decoded_list',decoded_list)
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
	# 本批样本中预测正确的次数
    correct_prediction = 0
	# 注意这里的标签不是指独热编码，而是这批样本的每个样本代表的字符串的集合
	# 如果解压缩后的真实标签和预测标签的样本个数不一样
    if len(original_list) != len(detected_list):
        print("真实标签样本个数:{},预测标签样本个数:{},真实标签与预测标签样本个数不匹配".format(len(original_list), len(detected_list)))
        return -1

    print("真实标签(长度) <-------> 预测标签(长度)")
	# 注意这里的标签不是指独热编码，而是这批样本的每个样本代表的字符串的集合
	# 如果真实标签和预测标签的样本个数吻合，则分别比对每一个样本的预测结果
	# for循环从original_list中取出一个一个的字符串(注意字符串存在一个列表中,列表中每个元素是单个字符)
    for idx, true_number in enumerate(original_list):
		# detected_list[idx]即detected_list中第idx号字符串(注意字符串存在一个列表中,列表中每个元素是单个字符)
        detect_number = detected_list[idx]
		# signal即真实标签是否与预测标签相等的结果，相等则为true
        signal = (true_number == detect_number)
		# 打印true_number和detect_number直观对比
        print(signal, true_number, "(", len(true_number), ") <-------> ", detect_number, "(", len(detect_number), ")")
		# 如果相等，统计正确的预测次数加1
        if signal is True:
            correct_prediction += 1
	# 计算本批样本预测的准确率
    acc = correct_prediction * 1.0 / len(original_list)
    print("本批样本预测准确率:{}".format(acc))
    return acc


# 定义训练过程
def train():
	global_step = tf.Variable(0, trainable=False)
	# tf.train.exponential_decay函数实现指数衰减学习率
	learning_rate = tf.train.exponential_decay(lr_start, global_step, train_iteration, lr_decay_factor, staircase=True)
	logits, inputs, targets, seq_len, w, b = get_train_model()
	# 注意得到的logits此时是未压缩的标签向量
	# 设置loss函数是ctc_loss函数
	# CTC ：Connectionist Temporal Classifier 一般译为联结主义时间分类器 ，适合于输入特征和输出标签之间对齐关系不确定的时间序列问题
	# CTC可以自动端到端地同时优化模型参数和对齐切分的边界。
	# 本例40X120大小的图片，切片成120列，输出标签最大设定为4(即不定长验证码最大长度为4),这样就可以用CTC模型进行优化。
	# 假设40x120的图片，数字串标签是"123"，把图片按列切分（CTC会优化切分模型），然后分出来的每块再去识别数字
	# 找出这块是每个数字或者特殊字符的概率（无法识别的则标记为特殊字符"-"）
	# 这样就得到了基于输入特征序列（图片）的每一个相互独立建模单元个体（划分出来的块）（包括“-”节点在内）的类属概率分布。
	# 基于概率分布，算出标签序列是"123"的概率P（123），当然这里设定"123"的概率为所有子序列之和，这里子序列包括'-'和'1'、'2'、'3'的连续重复

	# tf.nn.ctc_loss(labels, inputs, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)
	# labels: label实际上是一个稀疏矩阵SparseTensor，即真实标签(被压缩过的)
	# inputs:是RNN的输出logits，shape=[max_time_step,batch_size,num_classes]
	# sequence_length: bt_size个1乘以char_list_image_shape[1]，即bt_size个样本每个样本有多少个time_steps
	# preprocess_collapse_repeated: 设置为True的话, tensorflow会对输入的labels进行预处理, 连续重复的会被合成一个。
	# ctc_merge_repeated: 连续重复的是否被合成一个。
	cost = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len))
	# 这里用Adam算法来优化
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
	# tf.nn.ctc_beam_search_decoder对输入中给出的logits执行波束搜索解码。
	# ctc_greedy_decoder是ctc_beam_search_decoder中参数top_paths=1和beam_width=1（但解码器在这种特殊情况下更快）的特殊情况。
	# 如果merge_repeated是True，则合并输出序列中的重复类。这意味着如果梁中的连续条目相同，则仅发出第一个条目。
	# 如，当顶部路径为时A B B B B，返回值为：A B如果merge_repeated = True；A B B B B如果merge_repeated = False。
	# inputs：3-D float Tensor，尺寸 [max_time x batch_size x num_classes]。输入是预测的标签向量。
	# sequence_length：bt_size个1乘以char_list_image_shape[1]，即bt_size个样本每个样本有多少个time_steps
	# beam_width：int标量> = 0（波束搜索波束宽度）。
	# top_paths：int标量> = 0，<= beam_width（控制输出大小）。
	# merge_repeated：布尔值。默认值：True。如果merge_repeated是True，则合并输出序列中的重复类。
	# 返回值：
	# 元组(decoded, log_probabilities)
	# decoded：decoded是一组SparseTensor。由于我们每一次训练只输入一组训练数据，所以decoded里只有一个SparseTensor。
	# 即decoded[0]就是我们这组训练样本预测得到的SparseTensor，decoded[0].indices就是其位置矩阵。
	# log_probability：包含序列对数概率的float矩阵(batch_size)。
	decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

	# tf.edit_distance(hypothesis, truth, normalize=True, name="edit_distance"),计算序列之间的(Levenshtein)莱文斯坦距离
	# 莱文斯坦距离(LD)用于衡量两个字符串之间的相似度。莱文斯坦距离被定义为将字符串a变换为字符串b所需的删除、插入、替换操作的次数。
	# hypothesis: SparseTensor,包含序列的假设.truth: SparseTensor, 包含真实序列.
	# normalize: 布尔值,如果值True的话,求出来的Levenshtein距离除以真实序列的长度. 默认为True
	# name: operation 的名字,可选。
	accuracy1 = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
	print('accuracy:', accuracy1)

	def do_report():
		# 生成一批样本数据，进行测试，根据使用的是freetype还是captcha生成的验证码，使用不同的批样本
		# 为true时使用freetype生成验证码
		if use_freeType_or_captcha is True:
			test_inputs, test_targets, test_seq_len = free_type_get_next_batch(test_batch_size, char_list_image_shape)
		else:
			test_inputs, test_targets, test_seq_len = captcha_get_next_batch(test_batch_size, char_list_image_shape)
		# test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
		test_feed = {inputs: test_inputs,
					 targets: test_targets,
					 seq_len: test_seq_len}
		dd, log_probs, accuracy = sess.run([decoded[0], log_prob, accuracy1], test_feed)
		# dd = sess.run(decoded[0], feed_dict=test_feed)
		report_acc = report_accuracy(dd, test_targets)
		# 返回准确率
		return report_acc

	def do_batch():
		# 生成一批样本数据，进行训练，根据使用的是freetype还是captcha生成的验证码，使用不同的批样本
		# 为true时使用freetype生成验证码
		if use_freeType_or_captcha is True:
			train_inputs, train_targets, train_seq_len = free_type_get_next_batch(train_batch_size,
																				  char_list_image_shape)
		else:
			train_inputs, train_targets, train_seq_len = captcha_get_next_batch(train_batch_size, char_list_image_shape)
		train_feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
		b_cost, b_lr, b_acc, steps, _ = sess.run([cost, learning_rate, accuracy1, global_step, optimizer],
												 feed_dict=train_feed)
		return b_cost, steps, b_acc, b_lr

	# 创建模型文件保存路径
	if not os.path.exists("./free_type_image_lstm_model/"):
		os.mkdir("./free_type_image_lstm_model/")
	if not os.path.exists("./captcha_image_lstm_model/"):
		os.mkdir("./captcha_image_lstm_model/")
	saver = tf.train.Saver()
	# 创建会话，开始训练
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#        if use_freeType_or_captcha is True and os.path.exists("./free_type_image_lstm_model/checkpoint"):
		#			# 判断模型是否存在，如果存在则从模型中恢复变量
		#            saver.restore(sess, tf.train.latest_checkpoint('./free_type_image_lstm_model/'))
		#        if use_freeType_or_captcha is False and os.path.exists("./captcha_image_lstm_model/checkpoint"):
		#			# 判断模型是否存在，如果存在则从模型中恢复变量
		#            saver.restore(sess, tf.train.latest_checkpoint('./captcha_image_lstm_model/'))
		# 训练循环
		while True:
			start = time.time()
			# 每轮将一个batch的样本喂进去训练
			batch_cost, train_steps, acc, batch_lr = do_batch()
			batch_seconds = time.time() - start
			log = "iteration:{},batch_cost:{:.6f},batch_learning_rate:{:.12f},batch seconds:{:.6f}"
			print(log.format(train_steps, batch_cost, batch_lr, batch_seconds))
			if train_steps % test_report_step_interval == 0:
				# 如果使用freetype生成的验证码，则生成的模型存在free_type_image_lstm_model文件夹
				# 为True时使用freetype库生成验证码
				if use_freeType_or_captcha is True:
					saver.save(sess, "./free_type_image_lstm_model/train_model", global_step=train_steps)
				# 如果使用captcha生成的验证码，则生成的模型存在captcha_image_lstm_model文件夹
				else:
					saver.save(sess, "./captcha_image_lstm_model/train_model", global_step=train_steps)

				acc = do_report()
				if acc > acc_reach_to_stop:
					print("准确率已达到临界值{}，目前准确率{}，停止训练".format(acc_reach_to_stop, acc))
					break

 
 
if __name__ == '__main__':
	train()

