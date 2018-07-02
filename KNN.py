# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:42:37 2018

@author: Zhang Xiang
"""

import operator
import numpy as np

def creatData():
    """随机构造数据"""
    dataMat = np.array([[1, 100], [2, 150], [5, 40], [2, 40], [3, 120], [4, 50]])
    labels = ['YES', 'YES', 'No', 'No', 'YES', 'No']
    return dataMat, labels

def normData(dataSet):
    """避免不同特征本身取值大小的影响，对数据进行归一化处理 （data - min）/ (max - min)"""
    m = len(dataSet)
    Min = dataSet.min(axis=0)   # list() 没有min() 方法， dataSet --> np.array
    Max = dataSet.max(axis=0)
    normData = dataSet - np.tile(Min, (m, 1))       # tile(A, reps), Construct an array by repeating A the number of times given by reps
    normData = normData/np.tile((Max - Min), (m,1))
    return normData

def calcDistance(testVec, dataSet):
    """以欧几里得距离为例，计算测试向量与其他数据的距离, 矩阵运算"""
    m = len(dataSet)
    difference = dataSet - np.tile(testVec, (m, 1))
    difference = difference**2
    dist_Square = difference.sum(axis=1)
    distance = dist_Square**0.5
    return distance

def KNNclassify(testVec, dataSet, labels, k):
    """KNN算法分类"""
    distance = calcDistance(testVec, dataSet)
    dist_index = distance.argsort()    # 返回排序后的index， 方便确定对应的label
    valueCount = {}
    for i in range(k):
        label = labels[dist_index[i]]
        valueCount[label] = valueCount.get(label, 0)+1
#    valueCount.sort(key=operator.itemgetter(1), reverse=True)   # 'dict' object has no attribute 'sort'
    Count_sort = sorted(valueCount.items(), key=operator.itemgetter(1), reverse=True)
    return Count_sort[0][0]

if __name__ == "__main__":
    k = 2
    dataMat, labels = creatData()
    testVec = np.array([2, 200])   # 'YES'
    dataMat = normData(dataMat)
    result = KNNclassify(testVec, dataMat, labels, k)
    print(result)