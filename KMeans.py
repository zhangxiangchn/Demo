#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

def loadData(filename):
    fr = open(filename)
    dataMat = []
    for line in fr.readlines():
        curline = line.split('\t')
        curline = map(float, curline)
        dataMat.append(curline)
    return dataMat

def distEclud(vecA, vecB):
    # 计算两个向量的欧几里得距离
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def creatCent(dataSet, k):
    """随机选择聚类中心"""
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros(k, n))
    for i in range(n):
        minI = min(dataSet[:, i])
        rangeI = max(dataSet[:, i]) - minI
        centroids[:, i] = minI + rangeI*np.random.rand(k, 1)
    return centroids

def KMeans(dataSet,k):
    m,n = np.shape(dataSet)
    clusterAssment = np.mat(np.zeros((m,2)))   # 第一类的值为属于哪一个cluster，第二列放置的数据是到中心点的距离
    centroids = creatCent(dataSet, k)   # 随机确定质心的位置
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):    # 遍历所有数据点
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                vecA = np.array(centroids)[j,:]   # 某一个质心的坐标
                vecB = np.array(dataSet)[i,:]
                distJI = np.sqrt(sum(pow(vecA-vecB,2)))

                if distJI<minDist:            # 判断数据点与哪一cluster最近
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i,0]!=minIndex:  # 标记某一个数据点属于哪一个 cluster,当所有的数据点都不在变换cluster时，聚类结束
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(np.array(clusterAssment[:,0])==cent)]  # 找到属于某一类别的数据
            centroids[cent,:] = np.mean(ptsInClust,axis = 0)    # 计算这一类别的数据的中心点，axis= 0表示沿列方向求均值

    return centroids,clusterAssment

def biKmeans(dataSet, k):
    # 二分K均值聚类（K==2）
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0]     # create a list with one centroid
    
    for j in range(m):     #  calc initial Error
        clusterAssment[j,1] = distEclud(np.mat(centroid0), dataSet[j,:])**2
                      
    while (len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]  # get the data points currently in cluster i
            centroidMat, splitClustAss = KMeans(ptsInCurrCluster, 2)
            sseSplit = sum(splitClustAss[:,1])      # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
                
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)   # change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
                     
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]   # replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss   # reassign new clusters, and SSE
        
    return np.mat(centList), clusterAssment
    
    
