# -*- coding: utf-8 -*-
import numpy as np
import freetype
import copy
import random
import cv2
import os
 
 
# 使用FreeType库生成验证码图片时，我们输入的文本的属性pos位置和text_size大小是以像素点为单位的，但是将文本转化成字形时
# 需要把这些数据转换成1/64像素点单位计数的值，然后在画位图时，还要把相关的数据重新转化成像素点单位计数的值
# 这就是本class中几个方法主要做的工作
class PutChineseText(object):
	def __init__(self, ttf):
		# 创建一个face对象，装载一种字体文件.ttf
		self._face = freetype.Face(ttf)
 
	# 在一个图片（用三维数组表示）上绘制文本字符
	def draw_text(self, image, pos, text, text_size, text_color):
		"""
		draw chinese(or not) text with ttf
		:param image:     一个图片平面，用三维数组表示
		:param pos:       在图片上开始绘制文本字符的位置，以像素点为单位
		:param text:      文本的内容
		:param text_size: 文本字符的字体大小，以像素点为单位
		:param text_color:文本字符的字体颜色
		:return:          返回一个绘制了文本的图片
		"""
		# self._face.set_char_size以物理点的单位长度指定了字符尺寸，这里只设置了宽度大小，则高度大小默认和宽度大小相等
		# 我们将text_size乘以64倍得到字体的以point单位计数的大小，也就是说，我们认为输入的text_size是以像素点为单位来计量字体大小
		self._face.set_char_size(text_size * 64)
		# metrics用来存储字形布局的一些参数，如ascender，descender等
		metrics = self._face.size
		# 从基线到放置轮廓点最高(上)的距离，除以64是重新化成像素点单位的计数
		# metrics中的度量26.6象素格式表示，即数值是64倍的像素数
		# 这里我们取的ascender重新化成像素点单位的计数
		ascender = metrics.ascender / 64.0
		# 令ypos为从基线到放置轮廓点最高(上)的距离
		ypos = int(ascender)
		# 如果文本不是unicode格式，则用utf-8编码来解码，返回解码后的字符串
		if isinstance(text, str) is False:
			text = text.decode('utf-8')
		# 调用draw_string方法来在图片上绘制文本，也就是说draw_text方法其实主要是在定位字形位置和设定字形的大小，然后调用draw_string方法来在图片上绘制文本
		img = self.draw_string(image, pos[0], pos[1] + ypos, text, text_color)
		return img
 
	# 绘制字符串方法
	def draw_string(self, img, x_pos, y_pos, text, color):
		"""
		draw string
		:param x_pos: 文本在图片上开始的x轴位置，以1/64像素点为单位
		:param y_pos: 文本在图片上开始的y轴位置，以1/64像素点为单位
		:param text:  unicode形式编码的文本内容
		:param color: 文本的颜色
		:return:      返回一个绘制了文本字形的图片（三维数组形式）
		"""
		prev_char = 0
		# pen是笔位置或叫原点，用来定位字形
		pen = freetype.Vector()
		# 设定pen的x轴位置和y轴位置，注意pen.x和pen.y都是以1/64像素点单位计数的，而x_pos和y_pos都是以像素点为单位计数的
		# 因此x_pos和y_pos都左移6位即乘以64倍化成1/64像素点单位计数
		pen.x = x_pos << 6
		pen.y = y_pos << 6
 
		hscale = 1.0
		# 设置一个仿射矩阵
		matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000), int(0.0 * 0x10000), int(1.1 * 0x10000))
		cur_pen = freetype.Vector()
		pen_translate = freetype.Vector()
 
		# 将输入的img图片三维数组copy过来
		image = copy.deepcopy(img)
		# 一个字符一个字符地将其字形画成位图
		for cur_char in text:
			# 当字形图像被装载时，对该字形图像进行仿射变换,这只适用于可伸缩（矢量）字体格式。set_transform()函数就是做这个工作
			self._face.set_transform(matrix, pen_translate)
			# 装载文本中的每一个字符
			self._face.load_char(cur_char)
			# 获取两个字形的字距调整信息，注意获得的值是1/64像素点单位计数的。因此可以用来直接更新pen.x的值
			kerning = self._face.get_kerning(prev_char, cur_char)
			# 更新pen.x的位置
			pen.x += kerning.x
			# 创建一个字形槽，用来容纳一个字形
			slot = self._face.glyph
			# 字形图像转换成位图
			bitmap = slot.bitmap
			# cur_pen记录当前光标的笔位置
			cur_pen.x = pen.x
			# pen.x的位置上面已经更新过
			# bitmap_top是字形原点(0,0)到字形位图最高像素之间的垂直距离，由于是像素点计数的，我们用其来更新cur_pen.y时要转换成1/64像素点单位计数
			cur_pen.y = pen.y - slot.bitmap_top * 64
			# 调用draw_ft_bitmap方法来画出字形对应的位图，注意这里是循环，也就是一个字符一个字符地画
			self.draw_ft_bitmap(image, bitmap, cur_pen, color)
			# 每画完一个字符，将pen.x更新成下一个字符的笔位置（原点位置）,advanceX即相邻两个原点的水平距离(字间距)
			pen.x += slot.advance.x
			# prev_char更新成当前新画好的字符的字形的位置
			prev_char = cur_char
		# 返回包含所有字形的位图的图片（三维数组）
		return image
 
	# 将字形转化成位图
	def draw_ft_bitmap(self, img, bitmap, pen, color):
		"""
		draw each char
		:param bitmap: 要转换成位图的字形
		:param pen:    开始画字形的位置，以1/64像素点为单位
		:param color:  RGB三个通道值表示，每个值0-255范围
		:return:       返回一个三维数组形式的图片
		"""
		# 获得笔位置的x轴坐标和y轴坐标，这里右移6位是重新化为像素点单位计数的值
		x_pos = pen.x >> 6
		y_pos = pen.y >> 6
		# rows即位图中的水平线数
		# width即位图的水平象素数
		cols = bitmap.width
		rows = bitmap.rows
		# buffer数一个指向位图象素缓冲的指针，里面存储了我们的字形在某个位置上的信息，即字形轮廓中的所有的点上哪些应该画成黑色，或者是白色
		glyph_pixels = bitmap.buffer
 
		# 循环画位图
		for row in range(rows):
			for col in range(cols):
				# 如果当前位置属于字形的一部分而不是空白
				if glyph_pixels[row * cols + col] != 0:
					# 写入每个像素点的三通道的值
					img[y_pos + row][x_pos + col][0] = color[0]
					img[y_pos + row][x_pos + col][1] = color[1]
					img[y_pos + row][x_pos + col][2] = color[2]
 
 
# 快速设置带字符串的图片的属性
class GenerateCharListImage(object):
	# 初始化图片属性
	def __init__(self):
		# 候选字符集为数字0-9
		self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
		# 令char_set为候选字符集
		self.char_set = self.number
		# 计算候选字符集的字符个数
		self.len = len(self.char_set)
		# 生成的不定长验证码最大长度
		self.max_size = 4
		self.ft = PutChineseText('fonts/OCR-B.ttf')
 
	# 生成随机长度0-max_size之间的字符串，并返回字符串及对应的标签向量
	def random_text(self):
		# 空字符串
		text = ''
		# 空标签向量
		text_vector = np.zeros((self.max_size * self.len))
		# 设置字符串的长度是随机的
		size = random.randint(1, self.max_size)
		# size = self.max_size
		# 逐个生成字符串和对应的标签向量
		for index in range(size):
			c = random.choice(self.char_set)
			one_element_vector = self.one_char_to_one_element_vector(c)
			# 更新字符串和标签向量
			text = text + c
			text_vector[index * self.len:(index + 1) * self.len] = np.copy(one_element_vector)
		# 返回字符串及对应的标签向量
		return text, text_vector
 
	# 根据生成的字符串，生成验证码图片，返回图片数据和其标签,默认给图片添加高斯噪声
	def generate_color_image(self, img_shape, noise):
		text, text_vector = self.random_text()
		# 创建一个图片背景，图片背景为黑色
		img_background = np.zeros([img_shape[0], img_shape[1], 3])
		# 设置图片背景为白色
		img_background[:, :, 0], img_background[:, :, 1], img_background[:, :, 2] = 255, 255, 255
		# (0, 0, 0)黑色，(255, 255, 255)白色，(255, 0, 0)深蓝色，(0, 255, 0)绿色，(0, 0, 255)红色
		# 设置字体颜色为黑色
		text_color = (0, 0, 0)
		# 设置文本在图片上起始位置和文本大小，单位都是像素点
		pos = (20, 10)
		text_size = 20
		# 画出验证码图片，返回的image是一个三维数组
		image = self.ft.draw_text(img_background, pos, text, text_size, text_color)
		# 如果想添加噪声
		if noise == "gaussian":
			# 添加20%的高斯噪声
			image = self.image_add_gaussian_noise(image, 0.2)
		elif noise == "salt":
			# 添加20%的椒盐噪声
			image = self.image_add_salt_noise(image, 0.1)
		elif noise == "None":
			pass
		# 返回三维数组形式的彩色图片   此处也不同 另外一个程序返回的是单通道值
		return image, text, text_vector
 
	# 给一张生成的图片加入随机椒盐噪声
	def image_add_salt_noise(self, image, percent):
		rows, cols, dims = image.shape
		# 要添加椒盐噪声的像素点的数量，用全图像素点个数乘以一个百分比计算出来
		salt_noise_num = int(percent * image.shape[0] * image.shape[1])
		for i in range(salt_noise_num):
			# 获得随机的一个x值和y值，代表一个像素点
			x = np.random.randint(0, rows)
			y = np.random.randint(0, cols)
			# 所谓的椒盐噪声就是随机地将图像中的一定数量(这个数量就是椒盐的数量num)的像素值取极大或者极小
			# 即让维度0第x个，维度1第y个确定的一个像素点的数组(这个数组有三个元素)的三个值都为0,即噪点是黑色，因为我们的图片背景是白色
			image[x, y, :] = 0
		return image
 
	# 给一张生成的图片加入高斯噪声
	def image_add_gaussian_noise(self, image, percent):
		rows, cols, dims = image.shape
		# 要添加的高斯噪点的像素点的数量，用全图像素点个数乘以一个百分比计算出来
		gaussian_noise_num = int(percent * image.shape[0] * image.shape[1])
		# 逐个给像素点添加噪声
		for index in range(gaussian_noise_num):
			# 随机挑一个像素点
			x_temp, y_temp = np.random.randint(0, rows), np.random.randint(0, cols)
			# 随机3个值，加到这个像素点的3个通道值上，为了不超过255，后面再用clamp函数限定其范围不超过255
			value_temp = np.random.normal(0, 255, 3)
			for subscript in range(3):
				image[x_temp, y_temp, subscript] = image[x_temp, y_temp, subscript] - value_temp[subscript]
				if image[x_temp, y_temp, subscript] > 255:
					image[x_temp, y_temp, subscript] = 255
				elif image[x_temp, y_temp, subscript] < 0:
					image[x_temp, y_temp, subscript] = 0
		return image
 
	# 图片降噪函数
	def image_reduce_noise(self, image):
		# 使用方框滤波，normalize如果等于true就相当于均值滤波了，-1表示输出图像深度和输入图像一样，(2,2)是方框大小
		image = cv2.boxFilter(image, -1, (2, 2), normalize=False)
		return image
 
	# 将彩色图像转换成灰度图片的一维数组形式的数据形式
	def color_image_to_gray_image(self, image):
		# 将图片转成灰度数据，并进行标准化(0-1之间)
		r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
		gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
		# 生成灰度图片对应的一维数组数据，即输入模型的x数据形式
		gray_image_array = np.array(gray).flatten()
		# 返回度图片的一维数组的数据形式
		return gray_image_array
 
	# 单个字符转为向量
	def one_char_to_one_element_vector(self, c):
		one_element_vector = np.zeros([self.len, ])
		# 找每个字符是字符集中的第几个字符，是第几个就把标签向量中第几个元素值置1
		for index in range(self.len):
			if self.char_set[index] == c:
				one_element_vector[index] = 1
		return one_element_vector
 
	# 整个标签向量转为字符串
	def text_vector_to_text(self, text_vector):
		text = ''
		text_vector_len = len(text_vector)
		# 找标签向量中为1的元素值，找到后index即其下标，我们就知道那是候选字符集中的哪个字符
		for index in range(text_vector_len):
			if text_vector[index] == 1:
				text = text + self.char_set[index % self.len]
		# 返回字符串
		return text
 
 
if __name__ == '__main__':
	# 创建文件保存路径
	if not os.path.exists("./image/"):
		os.mkdir("./image/")
	# 图片尺寸
	image_shape = (40, 120)
	test_object = GenerateCharListImage()
	# 生成一个不加噪声的图片
	test_color_image_no_noise, test_text_no_noise, test_text_vector_no_noise = test_object.generate_color_image(
		image_shape, noise="None")
	test_gray_image_array_no_noise = test_object.color_image_to_gray_image(test_color_image_no_noise)
	print(test_gray_image_array_no_noise)
	print(test_text_no_noise, test_text_vector_no_noise)
	cv2.imwrite("./image/test_color_image_no_noise.jpg", test_color_image_no_noise)
	# 显示这张不加噪声的图片
	cv2.imshow("test_color_image_no_noise", test_color_image_no_noise)
	# 2000毫秒后刷新图像
	cv2.waitKey(2000)
	# 生成一个加了高斯噪声的图片
	test_color_image_gaussian_noise, test_text_gaussian_noise, test_text_vector_gaussian_noise = \
		test_object.generate_color_image(image_shape, noise="gaussian")
	cv2.imwrite("./image/test_color_image_gaussian_noise.jpg", test_color_image_gaussian_noise)
#	cv2.imshow("test_color_image_gaussian_noise", test_color_image_gaussian_noise)
#	# 2000毫秒后刷新图像
#	cv2.waitKey(2000)
	# 高斯噪声图片降噪后的图片
	test_color_image_reduce_gaussian_noise = test_object.image_reduce_noise(test_color_image_gaussian_noise)
	cv2.imwrite("./image/test_color_image_reduce_gaussian_noise.jpg", test_color_image_reduce_gaussian_noise)
#	cv2.imshow("test_color_image_reduce_gaussian_noise", test_color_image_reduce_gaussian_noise)
#	# 2000毫秒后刷新图像
#	cv2.waitKey(2000)
	# 生成一个加了椒盐噪声的图片
	test_color_image_salt_noise, test_text_salt_noise, test_text_vector_salt_noise = test_object.generate_color_image(
		image_shape, noise="salt")
	cv2.imwrite("./image/test_color_image_salt_noise.jpg", test_color_image_salt_noise)
#	cv2.imshow("test_color_image_salt_noise", test_color_image_salt_noise)
#	# 2000毫秒后刷新图像
#	cv2.waitKey(2000)
	# 椒盐噪声图片降噪后的图片
	test_color_image_reduce_salt_noise = test_object.image_reduce_noise(test_color_image_salt_noise)
	cv2.imwrite("./image/test_color_image_reduce_salt_noise.jpg", test_color_image_reduce_salt_noise)
#	cv2.imshow("test_color_image_reduce_salt_noise", test_color_image_reduce_salt_noise)
#	# 2000毫秒后刷新图像
#	cv2.waitKey(2000)
	cv2.destroyAllWindows()

