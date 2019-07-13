from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number+ ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)

    captcha_image = np.array(captcha_image)  #转成np.array
    captcha_image=Image.fromarray(np.uint8(captcha_image)) #转成PIL Image
    captcha_image = captcha_image.resize((80, 26), Image.ANTIALIAS) #缩放
    captcha_image.save("./imgs/"+captcha_text + '.jpg') #存图
    # captcha_image.write(captcha_text, "./imgs/"+captcha_text + '.jpg')  # 写到文件
    return captcha_text, captcha_image


if __name__ == '__main__':
    # 测试
    for i in range(10000):
        text, image = gen_captcha_text_and_image()
        if(i%100==0):
            print("生成第%s张图" % i)


    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    # plt.imshow(image)
    #
    # plt.show()