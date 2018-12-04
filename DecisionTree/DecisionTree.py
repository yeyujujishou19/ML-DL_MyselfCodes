from sklearn.feature_extraction import DictVectorizer    #转换数据用
import csv    #存储原始数据
from sklearn import preprocessing   #处理
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData=open(r'E:\sxl_Programs\DecisionTree.csv','rt')  #
reader=csv.reader(allElectronicsData)
headers=next(reader)   #读取文件头
print (headers)

featureList=[]   #特征值
labelList=[]     #分类结果标签，是否买电脑，yes no


for row in reader:
    labelList.append(row[len(row)-1])        #读取分类标签值
    rowDict={}
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)              #读取特征值

print(featureList)

vec=DictVectorizer()
dumpyX=vec.fit_transform(featureList).toarray()  #将特征值转化成需要的格式

print("dumpyX:"+str(dumpyX))
print(vec.get_feature_names())

print("LabelList:"+str(labelList))

#Vectorize class labels
lb=preprocessing.LabelBinarizer()
dumpyY=lb.fit_transform(labelList)   #分类结果标签转换成需要的格式
print("dumpY:"+str(dumpyY))

#using decison tree for classification
# clf =tree.DecisionTreeClassifier()
clf=tree.DecisionTreeClassifier(criterion='entropy')  #产生决策树
clf=clf.fit(dumpyX,dumpyY)
print("clf:"+str(clf))

with open("allElectronicInformationGainOri.dot","w") as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

oneRowX=dumpyX[0,:]
print("oneRowX:"+str(oneRowX))

newRowX=oneRowX      #测试决策树

newRowX[0]=1
newRowX[2]=0
print("newRowX:"+str(newRowX))