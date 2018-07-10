# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 20:59:15 2018

@author: Zhang Xiang
"""
import numpy as np

def loadData(filename):
    """数据读取，使用的是原文中的数据"""
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    yMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        yMat.append(float(curLine[-1]))
    return dataMat, yMat

def Regress(xArr,yArr):
    # 使用正规方程来求解，需要判断行列式是否为零，奇异矩阵不能使用正规方程求解
    # 返回函数的系数
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

    
def lwlr(xArr, yArr, testPoint, k = 1.0):
    # 局部加权回归, 返回预测之后的数值
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for i in range(m):
        diff = testPoint - xMat[i, :]
        weights[i, i] = np.exp(diff*diff.T/(-2*k**2))
    xTx = xMat.T*weights*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*weights*yMat)
    return testPoint*ws
    
def lwlrTest(testArr, xArr, yArr, k = 1.0):
    # 预测局部加权回归的结果
    m = len(xArr)
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(xArr, yArr, testArr[i], k)
    return yHat


if __name__ == "__main__":
#    filename = r"E:\machinelearninginaction\Ch08\ex0.txt"
#    dataMat, yMat = loadData(filename)
#    weights = Regress(dataMat, yMat)
#    print(weights)
#    xData = np.mat(dataMat)
#    yData = np.mat(yMat)
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.scatter(xData[:, 1].flatten().A[0], yData.T.flatten().A[0])
#    xcopy = xData.copy()
#    xcopy.sort(0)
#    yHat = xcopy*weights
#    ax.plot(xcopy[:, 1], yHat)
#    plt.show()
#    r = np.corrcoef(yHat, yData.T)

    filename = r"E:\machinelearninginaction\Ch08\ex0.txt"
    xArr, yArr = loadData(filename)
    yHat = lwlrTest(xArr, xArr, yArr, k = 0.01)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0])
    sortInd = xMat[:, 1].argsort(0)
    Xsort = xMat[sortInd][:, 0, :]
    ax.plot(Xsort[:, 1], yHat[sortInd])
    plt.show()
    r = np.corrcoef(yHat, yMat)
    print(r)
    
    