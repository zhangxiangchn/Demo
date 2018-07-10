# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:30:13 2018

@author: Zhang Xiang
"""
import numpy as np

def loadData(filename):
    """导入数据"""
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.split('\t')
        curline = list(map(float, curline))
        dataMat.append(curline)
    return dataMat

def binsplitDataSet(dataMat, feature, value):
    """将数据分为左右两部分"""
    mat0 = dataMat[np.nonzero(dataMat[:, feature]>value)[0], :]
    mat1 = dataMat[np.nonzero(dataMat[:, feature]<=value)[0], :]
    return mat0, mat1

def solverLinear(dataSet):
    # 构造线性方程
    m, n = np.shape(dataSet)
    X = np.matrix(np.ones((m, n)))
    y = np.matrix(np.ones((m, 1)))
    X[:, 1:] = dataSet[:, :n-1]
    y = dataSet[:, -1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0:
        print('This matrix is singular, try increasing the value of ops[1]')
    ws = xTx.I*(X.T*y)
    return ws, X, y

def modelleaf(dataSet):
    # 对模型树叶节点的处理, 模型树的叶节点的返回值为线性回归方程的系数
    ws, X, y = solverLinear(dataSet)
    return ws

def modelErr(dataSet):
    # 模型树误差的计算
    ws, X, y = solverLinear(dataSet)
    yHat = X*ws
    return sum(np.power(yHat - y, 2))
    

def ChooseBestSplit(dataSet, leafType = modelleaf, errType = modelErr, ops = (1, 4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf;bestIndex = 0;bestValue = 0
    for featureIndex in range(n-1):
        for splitValue in set(dataSet[:,featureIndex].T.tolist()[0]):
            mat0, mat1 = binsplitDataSet(dataSet, featureIndex, splitValue)
            if (len(mat0)<tolN or len(mat1)<tolN):
                continue
            errS = errType(mat0) + errType(mat1)
            if errS < bestS:
                bestIndex = featureIndex
                bestS = errS
                bestValue = splitValue
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binsplitDataSet(dataSet, bestIndex, bestValue)
    if (len(mat0) < tolN or len(mat1) < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue
    

def CreatTree(dataSet, leafType = modelleaf, errType = modelErr, ops = (10, 100)):
    """构造树"""
    feat, val =ChooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lmat, rmat = binsplitDataSet(dataSet, feat, val)
    retTree['left'] = CreatTree(lmat, leafType, errType, ops)
    retTree['right'] = CreatTree(rmat, leafType, errType, ops)
    return retTree

if __name__ == "__main__":
    # 建树过程， 条件ops控制树的大小
    filename = r'E:\machinelearninginaction\Ch09\ex2.txt'
    dataMat = loadData(filename)
    dataMat = np.mat(dataMat)
    myTree = CreatTree(dataMat)
    print(myTree)
