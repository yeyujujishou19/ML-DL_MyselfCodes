
import cv2
import numpy as np
import random
from random import randint

np.random.seed(0)   #保证下次运行程序时产生的点和本次一样
# ANN网络初始化
animals_net = cv2.ml.ANN_MLP_create()
animals_net.setLayerSizes(np.array([3, 10,10, 2]))
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
animals_net.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
animals_net.setBackpropWeightScale(0.1)
animals_net.setBackpropMomentumScale(0.1)

#狗，数据
def dog_sample():
  return [random.uniform(-1,0), random.uniform(-1,0), random.uniform(-1,0)]
  # return [randint(10, 10), randint(20,20)]

#狗，标签
def dog_class():
  return [1, 0]

#猫，数据
def cat_sample():
  return [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
  # return [randint(50,50), randint(70,70)]

#猫，标签
def cat_class():
  return [0, 1]


#记录
def record(sample, classification):
  return (np.array([sample], dtype=np.float32), np.array([classification], dtype=np.float32))

records = []


#生成训练集
RECORDS = 500
for x in range(0, RECORDS):
  # records.append(record(cat_sample(), cat_class()))
  records.append(record(dog_sample(), dog_class()))
for x in range(0, RECORDS):
  records.append(record(cat_sample(), cat_class()))

random.shuffle(records)  # 打乱数

#迭代200轮，训练
EPOCHS = 20
for e in range(0, EPOCHS):
  print("Epoch %d:" % e)
  for t, c in records:
    animals_net.train(t, cv2.ml.ROW_SAMPLE, c)

#测试
TESTS = 100
dog_results = 0
cat_results = 0

for x in range(0, TESTS):
  clas = int(animals_net.predict(np.array([dog_sample()], dtype=np.float32))[0])
  res = animals_net.predict(np.array([dog_sample()], dtype=np.float32))
  print("class0: {}".format(res))
  if (clas) == 0:
    dog_results += 1

for x in range(0, TESTS):
  clas = int(animals_net.predict(np.array([cat_sample()], dtype=np.float32))[0])
  res = animals_net.predict(np.array([cat_sample()], dtype=np.float32))
  print("class1: {}".format(res))
  if (clas) == 1:
    cat_results += 1


print ("Dog accuracy: %f%%" % (dog_results))
print ("Cat accuracy: %f%%" % (cat_results))
