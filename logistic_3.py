# -*- coding: utf-8 -*-
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
iter_time=1000


def sigmoid(x):
    y=1.0/(1+np.power(math.e,-(x)))
    return y
def innner(x,y):
    output=0.0
    if len(x)!=len(y):
        print('矩阵长度不匹配')
    else:
        for t in range(len(x)):
            output=output+x[t]*y[t]
    return output


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([ float(lineArr[0]), float(lineArr[1]),1.0,])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def logistics(X,y,weights,l_wb):#N为测试集数据集长度
    n=len(X)
    n1=len(X[1])
    alpha = 0.1
    m,n = np.shape(X)
    for k in range(n):
        x=X[k]
        hh=innner(weights,x)#求内积
         #随机梯度上升法
        h = sigmoid(sum(x* weights))
        error = y[k] - h
        weights = weights + alpha * error * x
        p1=(math.e**hh)/(1.0+(math.e**hh))
        p0=1.0/(1.0+(math.e**hh))
        l_wb=l_wb+(-(y[k])*hh+math.log(1+math.e**(hh)))
    return {'l_wb':l_wb,
            'weights':weights}

def stocGradAscent0(dataMatrix, classLabels):  #随机梯度上升算法
    dataMatrix = np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    alpha = 0.1
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

[X,y]=loadDataSet()
print(X)
result={}
result_w={}
result['l_wb']=[]
result_w['weights']=[]
alpha = 0.1
m,n = np.shape(X)
X=np.array(X)
l_wb=0.0
weights =np.ones(n)
print(weights)
b=np.random.random()

for l in range(iter_time):
    logistics_1=logistics(X,y,weights,l_wb)
    l_wb=logistics_1.get('l_wb')
    weights=logistics_1.get('weights')
    result['l_wb'].append(l_wb)
    result_w['weights'].append(weights)
    if l==iter_time-1:
        wei=weights
        print(wei)

plt.plot(result.get('l_wb'))
plt.show()

    #画图
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('original data')
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []
for i in range(m):
        if int(y[i]) == 1.0:
            xcord1.append(X[i,0]); ycord1.append(X[i,1])
        else: xcord2.append(X[i,0]); ycord2.append(X[i,1])

ax = fig.add_subplot(111)
ax.scatter(xcord1, ycord1,  c = 'red', marker='o')
ax.scatter(xcord2, ycord2, c = 'green',marker='o')
x = np.arange(-3.0,3.0,0.1)

y =(-wei[0]*wei[2]- wei[1]*x)/wei[2]# -wei[0]*x/wei[1]-wei[2]
ax.plot(x, y)
plt.xlabel('X1');
plt.ylabel('X2');
plt.show()
