import cv2
import pickle
import numpy as np
import gzip

"""OpenCV ANN Handwritten digit recognition example

Wraps OpenCV's own ANN by automating the loading of data and supplying default paramters,
such as 20 hidden layers, 10000 samples and 1 training epoch.

The load data code is taken from http://neuralnetworksanddeeplearning.com/chap1.html
by Michael Nielsen
"""
"""
这是一个ann类库，为了尽可能自动执行，我们进行封装
"""


# 读取mnist.pkl.gz数据
def load_data():
    mnist = gzip.open('E:/sxl_Programs/Python/MNIST_data/MNIST_data/mnist.pkl.gz', 'rb')
    # 训练集，校验集和测试集 注意 pickle.load(mnist,encoding="bytes")
    training_data, classification_data, test_data = pickle.load(mnist, encoding="bytes")
    mnist.close()
    return (training_data, classification_data, test_data)


def wrap_data():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    # 训练集，校验集和测试集，只有训练集是有results的，也就是有监督的
    return (training_data, validation_data, test_data)


# 创建包含10个元素的零元组，在期望结果的位置设置1.这样可以用作输出层的类标签
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# 创建一个用于手写数字识别的ann
def create_ANN(hidden=20):
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([784, hidden, 10]))
    ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    # 截至条件
    ann.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1))
    return ann


def train(ann, samples=10000, epochs=1):
    tr, val, test = wrap_data()

    # 给定一定数量的样本和训练周期，加载数据，然后迭代某个设定的次数
    for x in range(epochs):
        counter = 0
        for img in tr:

            if (counter > samples):
                break
            if (counter % 1000 == 0):
                print("Epoch %d: Trained %d/%d" % (x, counter, samples))
            counter += 1
            data, digit = img
            ann.train(np.array([data.ravel()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
                      np.array([digit.ravel()], dtype=np.float32))
        print("Epoch %d complete" % x)
    return ann, test


# 封装ann的test
def test(ann, test_data):
    sample = np.array(test_data[0][0].ravel(), dtype=np.float32).reshape(28, 28)
    cv2.imshow("sample", sample)
    cv2.waitKey()
    print(ann.predict(np.array([test_data[0][0].ravel()], dtype=np.float32)))


# 封装ann的predict
def predict(ann, sample):
    resized = sample.copy()
    rows, cols = resized.shape
    if (rows != 28 or cols != 28) and rows * cols > 0:
        resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_LINEAR)
    return ann.predict(np.array([resized.ravel()], dtype=np.float32))

"""
# 使用方法：:
ann, test_data = train(create_ANN())
test(ann, test_data)
"""
