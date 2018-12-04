import numpy as np


def kmeans(X,k,maxIt):

    #返回X的行列数
    numPoints,numDim=X.shape

    #在X列的基础上多加入一列，为了放类的标签
    dataSet =np.zeros((numPoints,numDim+1))
    #不算最后一列，其他行列赋值为X的值
    dataSet[:,:-1]=X

    #随机产生中心点
    centroids=dataSet[np.random.randint(numPoints,size=k),:]
    # centroids=dataSet[0:2,:]

    centroids[:,-1]=range(1,k+1)

    iterations=0
    oldCentroids=None

    #旧中心点和新中心点不相等，循环次数没达到指定次数
    while not shouldStop(oldCentroids,centroids,iterations,maxIt):
        print ("iteration: \n",iterations)   #第几轮
        print("dataSet: \n",dataSet)         #数据集
        print ("centroids \n",centroids)     #中心点


        oldCentroids=np.copy(centroids)      #将中心点拷出，不能直接用=号
        iterations+=1                        #轮数加1

        updateLabels(dataSet,centroids)      #更新标签

        centroids=getCentroids(dataSet,k)    #获取中心点

    return dataSet                           #返回数据集

#判断要不要停止
def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids,centroids)

#更新每个点的类标签
def updateLabels(dataSet,centroids):
    numPoints,numDim=dataSet.shape
    for i in range(0,numPoints):
        #当前行最后一列
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1],centroids)

#比较当前行和每一个中心点距离，选择最近的距离，将其归为该中心点的类
def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]; #中心点的类标签
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])  #先让它等于第一个中心点的距离
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1]) #从第二个中心点开始判断
        if dist<minDist:   #更新最小值
            minDist=dist
            label=centroids[i,-1]   #保存标签
    print ("minDist",minDist)
    return label

#更新中心点坐标
def getCentroids(dataSet,k):
    result=np.zeros((k,dataSet.shape[1]))
    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1,-1]=i
    return result

x1=np.array([1,1])
x2=np.array([2,1])
x3=np.array([4,3])
x4=np.array([5,4])
testX=np.vstack((x1,x2,x3,x4))

result=kmeans(testX,2,10)
print("final result:")
print(result)