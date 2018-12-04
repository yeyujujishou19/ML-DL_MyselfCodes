# 引入模块
import os

#创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


###################################################
f = open("汉字.txt", "r")
lines = f.readlines()  # 读取全部内容
for line in lines:
    print(line)
    # 定义要创建的目录
    mkpath=(r"//192.168.107.145/宋晓利任辉/data2/汉字文件夹./%s" %(line))
    # 调用函数
    mkdir(mkpath)
print(len(lines))