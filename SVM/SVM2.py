

import numpy as np  #numpy是支持矩阵运算的数据包
import pylab as pl  #画图的功能
from sklearn import svm    #导入svm库

#创建40个独立点，并且是线性可分的
np.random.seed(0)   #保证下次运行程序时产生的点和本次一样
#  np.random.randn(20,2)  产生20个2维的点
#  - [2,2]   均值是2，方差是2，“-”表示在下方  “+”表示上方
X=np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2) + [2,2]]
Y=[0]*20+[1]*20

#创建模型
clf=svm.SVC(kernel='linear')
clf.fit(X,Y)

#获得独立的超平面，中间那条线     w0*x+w1*y+w3=0  -> y=-w0/w1-w3/w1
w=clf.coef_[0]    #获取w值 w0 w1
a=-w[0]/w[1]      #斜率
xx=np.linspace(-5,5)     #产生-5到5的连续值，x=-5,-4,-3,-2,-1,0,1,2,3,4,5
yy=a*xx-(clf.intercept_[0])/w[1]   #(clf.intercept_[0])/w[1] 截距

#画出与超平面平行的且经过支持向量的平行线
b=clf.support_vectors_[0]      #获取支持向量的第一个点
yy_down=a*xx+(b[1]-a*b[0])     #(b[1]-y)/b[0]=a  -> y=b[1]-a*b[0]
b=clf.support_vectors_[-1]     #获取支持向量的最后一个点
yy_up=a*xx+(b[1]-a*b[0])

print("w:",w)
print("a:",a)
print("support_vectors:",clf.support_vectors_)
print("clf.coef_:",clf.coef_)

#画出线，点，和支持向量
pl.plot(xx,yy,'k--')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

#支持向量单独标记出来
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
           s=80,facecolors='red')
pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()