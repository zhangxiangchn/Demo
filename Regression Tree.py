# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 21:30:27 2018

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

def regleaf(dataSet):
    return np.mean(dataSet[:, -1])

def regError(dataSet):
    return np.var(dataSet[:,-1])*(np.shape(dataSet)[0])

def ChooseBestSplit(dataSet, leafType = regleaf, errType = regError, ops = (1, 4)):
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
    

def CreatTree(dataSet, leafType = regleaf, errType = regError, ops = (1, 4)):
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

# 下面的三个函数时后剪枝用到的，预防过拟合需要进行后剪枝，一般情况需要后剪枝和预剪枝一起实现
def isTree(obj):
    # 判断是不是tree
    return type(obj).__name__ == 'dict'
    
def genMean(tree):
    # 递归函数,tree的叶节点是一个均值
    if isTree(tree['left']):
        tree['left'] = genMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = genMean(tree['right'])
    return (tree['left']+tree['right'])/2

def prune(testData, tree):
    # 后剪枝，根据是否可以减少误差，确定是不是需要后剪枝
    if (len(testData)==0):
        return genMean(tree)    # 没有测试集
    if isTree(tree['left']) or isTree(tree['right']):
        ltest, rtest = binsplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(ltest, tree['left'])
    if isTree(tree['right']):
        tree['right'] = prune(rtest, tree['right'])
    if not isTree(tree['left']) and not isTree(tree['right']):
        ltest, rtest = binsplitDataSet(testData, tree['spInd'], tree['spVal'])
        ErrorNoMerge = sum(np.power(ltest[:, -1] - tree['left'], 2)) + sum(np.power(rtest[:, -1] - tree['right'], 2))
        TreeMeanValue = (tree['left'] + tree['right'])/2
        ErrorMerge = sum(np.power(testData[:, -1] - TreeMeanValue, 2))
        if ErrorMerge<ErrorNoMerge:    # 当剪枝之后误差变小，进行剪枝，否则不进行剪枝，直接返回树
            print('merging')
            return TreeMeanValue
        else:
            return tree
    else:
        return tree


if __name__ == "__main__":
    # 建树过程， 条件ops控制树的大小
    filename = r'E:\machinelearninginaction\Ch09\ex2.txt'
    dataMat = loadData(filename)
    dataMat = np.mat(dataMat)
    myTree = CreatTree(dataMat)
    print(myTree)
    
    # 剪枝过程
    filename2 = r'E:\machinelearninginaction\Ch09\ex2test.txt'
    testMat = loadData(filename2)
    testMat = np.mat(testMat)
    myTree = prune(testMat, myTree)
    print(myTree)