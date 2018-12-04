from sklearn import neighbors  #包含邻近算法模块
from sklearn import datasets   #数据集

knn=neighbors.KNeighborsClassifier()     #调用算法分类器

iris=datasets.load_iris()    #加载数据集数据

print (iris)                 #打印数据集数据

knn.fit(iris.data,iris.target)   #建立KNN模型

predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])   #新的待分类数据

print (predictedLabel)     #打印分类结果