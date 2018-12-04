import csv        #加载csv文件
import random     #产生随机数
import math       #数学库
import operator


def loadDataset(filename,split,trainingSet=[],testSet=[]):   #加载数据，将数据分为训练集和测试集
    with open(filename,'rt') as csvfile:   #打开csv文件
        lines=csv.reader(csvfile)          #读取csv文件数据
        dataset=list(lines)                #转化数据类型
        for x in range(len(dataset)-1):    #x行
            for y in range(4):             #4列
                dataset[x][y]=float(dataset[x][y])
            if random.random() < split:       #产生0-1的随机数，小于split则放入训练集
                trainingSet.append(dataset[x])   #数据放入训练集
            else:
                testSet.append(dataset[x])    #数据放入测试集

def euclideanDistance(instance1,instance2,length):   #计算多维度距离
    distance=0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet,testInstance,k):     #获取前k个数据
    distances=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):                 #返回分类结果
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet,predictions):          #测试集返回正确率
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return (correct/float(len(testSet)))*100.0

def main():    #主函数
     trainingSet=[]     #训练集
     testSet=[]         #测试机
     split=0.67         #对原始数据分割，67%分为训练集
     loadDataset(r'E:\sxl_Programs\Python\Iris.txt',split,trainingSet,testSet)   #读取数据集
     print('Train set:'+ repr(len(trainingSet)))    #打印训练集
     print('Test set:' + repr(len(testSet)))  #打印测试集

     predictions=[]
     k=3     #确定k为3
     for x in range(len(testSet)):
         neighbors=getNeighbors(trainingSet,testSet[x],3)  #对当前测试集，计算它与所有训练集的距离，并返回值最小的前三个数据
         result=getResponse(neighbors)     #返回分类结果
         predictions.append(result)   #将分类结果存入列表
         print('>predicted'+repr(result)+',actual='+repr(testSet[x][-1]))   #打印结果
     accuracy=getAccuracy(testSet,predictions)  #测试集返回正确率
     print('Accuracy:'+repr(accuracy)+'%')

main()   #执行main()函数