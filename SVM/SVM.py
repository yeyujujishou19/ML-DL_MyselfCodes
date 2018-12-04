from sklearn import svm

#线性可分的情况下应用SVM分类

x=[[2,0],[1,1],[2,3]]     #训练点
y=[0,0,1]                 #类型标签
clf=svm.SVC(kernel='linear')    #建立分类器   kernel='linear' 线性核函数
clf.fit(x,y)   #创建模型

print(clf)

#打印支持向量点
print(clf.support_vectors_)

#打印支持向量点下标
print(clf.support_)

#打印每个分类支持向量点的个数
print(clf.n_support_)

#预测一个新点类型
x1=[[2,0]]
print(clf.predict([[2,4]]))