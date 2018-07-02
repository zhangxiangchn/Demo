#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import random

"""这里的所有的停止条件只有迭代次数结束，没有设置error满足要求退出，需要可自行对error进行判断"""

def loadData(filename):
    # 导入需要的数据，这里以包含label的文本数据为例,假设包含两个特征，一个label
    data = []; labels = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        data.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labels.append(int(lineArr[2]))
    return data, labels

def sigmoid(x):
    # 非线性化处理，将数据结果归一化到[0， 1]之间，便于进行二分类处理
    return 1.0/(1+np.exp(-x))

def gradAscent(dataMat,classLabel,alpha,maxCynum):
    """输入变量依次是：特征矩阵，目标列，学习率，最大循环次数"""
    # 基于梯度上升算法实现，每循环一次，所有数据都需要计算一遍
    dataMatrix = np.mat(dataMat)     # 这里需要转化为矩阵格式，下面进行的是矩阵计算
    labelMat   = np.mat(classLabel).transpose()
    m,n = dataMatrix.shape
    weights = np.ones((n,1))
    for i in range(maxCynum):
        h = sigmoid(dataMatrix*weights)
        error = labelMat - h     # error 即代价函数，这里直接以真实值和计算值的差作为代价函数
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

def stoGradAscent0(dataMat, classlabel, alpha):
    # 随机梯度上升实现，每次处理一个数据，对weights进行一次更新。
    m, n = np.shape(dataMat)  # 如果dataMat是list， 则不能使用dataMat.shape 方法， 只能用np.shape(dataMat)
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMat[i]*weights))
        error = classlabel[i] - h   # error这里是一个数值
        weights = weights + alpha*error*dataMat[i]
    return weights

def stoGradAscent1(dataMat, classlabel, num_iter = 150):
    # 优化的随机梯度上升方法，实时更新alpha，循环多次，并且打乱顺序处理
    m, n = np.shape(dataMat)
    weights = np.ones(n)
    for j in range(num_iter):
        dataIndex = list(range(m))
        for i in range(n):
            alpha = 1.0/(1.0+i+j)+0.01
            Index = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[Index]*weights))
            error = classlabel[Index] - h
            weights = weights + alpha*error*dataMat[Index]
    return weights

def classify(testVec, weights):
    # 根据学习到的weights系数对测试数据进行分类
    res = sigmoid(sum(testVec*weights))
    if res>0.5:
        return 1
    else:
        return 0

if __name__ == "__main__":
    filename = r'E:\machinelearninginaction\Ch05\testSet.txt'
    alpha = 0.01
    maxnum = 500
    dataMat, labels = loadData(filename)
    print('Training...')
    weights = gradAscent(dataMat,labels,alpha,maxnum)
    print('Testing...')
    testVec = [1.0, 3.5, 5.2]  # 1
    result = classify(testVec, weights)
    print(result)