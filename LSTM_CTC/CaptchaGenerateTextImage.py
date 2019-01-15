# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import cv2
import os
 
class GenerateCaptchaListImage(object):
	# 生成验证码的函数，生成的验证码序列长度为4
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
 
	def random_captcha_text(self):
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
 
	# 获取和生成的验证码对应的字符图片
	def generate_color_image(self, img_shape):
		# 生成指定大小的图片
		img = ImageCaptcha(height=img_shape[0], width=img_shape[1])
		# 生成一个随机的验证码序列
		text, text_vector = self.random_captcha_text()
		# 根据验证码序列生成对应的字符图片
		image = img.generate(text)
		image = Image.open(image)
		# 因为图片是用RGB模式表示的，将其转换成数组即图片的分辨率160X60的矩阵，矩阵每个元素是一个像素点上的RGB三个通道的值
		image = np.array(image)
		return image, text, text_vector
 
	# 图片降噪函数,对于captcha生成的验证码，使用中值滤波降噪效果较好
	def image_reduce_noise(self, image):
		# 使用中值滤波，7表示中值滤波器使用7×7的范围来计算。
		# 即对像素的中心值及其7×7邻域组成了一个数值集，对其进行处理计算，当前像素被其中位值替换掉。
		# 这样，如果在某个像素周围有白色或黑色的像素，这些白色或黑色的像素不会选择作为中值（最大或最小值不用），而是被替换为邻域值。
		image = cv2.medianBlur(image, 7)
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
 
 
# 测试
if __name__ == '__main__':
	# 创建文件保存路径
	if not os.path.exists("./captcha_image/"):
		os.mkdir("./captcha_image/")
	# 图片尺寸
	image_shape = (60, 120)
	test_object = GenerateCaptchaListImage()
	# 生成一张有噪声的验证码图片(captcha生成的验证码图片默认就是带有噪声的)
	test_color_image_noise, test_text_noise, test_text_vector_noise = test_object.generate_color_image(image_shape)
	test_gray_image_array_noise = test_object.color_image_to_gray_image(test_color_image_noise)
	print(test_gray_image_array_noise)
	print(test_text_noise, test_text_vector_noise)
	cv2.imwrite("./captcha_image/test_color_image_noise.jpg", test_color_image_noise)
	cv2.imshow("test_color_image_noise", test_color_image_noise)
	# 2000毫秒后刷新图像
	cv2.waitKey(2000)
	# 降噪后的图片
	test_color_image_reduce_noise = test_object.image_reduce_noise(test_color_image_noise)
	cv2.imwrite("./captcha_image/test_color_image_reduce_noise.jpg", test_color_image_reduce_noise)
	cv2.imshow("test_color_image_reduce_noise", test_color_image_reduce_noise)
	# 2000毫秒后刷新图像
	cv2.waitKey(2000)
	cv2.destroyAllWindows()

