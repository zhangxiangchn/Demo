""" 
coding: utf-8
@author: zhangxiang
"""
from math import log
import operator
import pickle

def loadData():
    """方便后面使用数据集"""
    dataMat = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [1, 0, 'no']]
    label = ['feature1', 'feature2']
    return dataMat, label

def calcEnt(dataMat):
    """计算数据集的信息熵"""
    numlength = len(dataMat)
    Labelcount = {}
    for feat in dataMat:
        labelValue = feat[-1]
        if labelValue not in Labelcount.keys():
            Labelcount[labelValue] = 0
        Labelcount[labelValue] += 1
    shanentroy = 0.0
    for key in Labelcount:
        prob = float(Labelcount[key])/numlength
        shanentroy -= prob*log(prob, 2)
    return shanentroy

def splitdata(dataMat, index, value):
    """返回index列等于value的行的数据集（去掉index列）"""
    returndata = []
    for featdata in dataMat:
        if featdata[index]==value:
             newlist = featdata[:index]
             newlist.extend(featdata[index+1:])
             returndata.append(newlist)
    return returndata

def choosebestfeature(dataMat):
    """根据信息增益，选择数据集中最好的特征（信息增益最大的特征）, 并返回该特征的位置"""
    numberfeature = len(dataMat[0])-1
    baseentropy = calcEnt(dataMat)
    bestinfoGain = 0.0; bestfeature = -1
    for i in range(numberfeature):
        featurelist = [example[i] for example in dataMat]
        featureset = set(featurelist)
        new_entropy = 0.0
        for value in featureset:
            subdatamat = splitdata(dataMat, i, value)
            prob = len(subdatamat)/float(len(dataMat))
            new_entropy += prob*calcEnt(subdatamat)
        infoGain = baseentropy - new_entropy
        if infoGain > bestinfoGain:
            bestinfoGain = infoGain
            bestfeature = i
    return bestfeature

def majorlity(classlist):
    """当所有的特征都使用完了，还没有将数据集分开，根据多数投票原则进行处理， 并归为类别最多的类"""
    vote = {}
    for label in classlist:
        if label not in vote.keys():
            vote[label] = 0
        vote[label] += 1
    votesorted = sorted(vote.items(), key=operator.itemgetter(1), reverse=True)
    return votesorted[0][0]

def creatTree(dataMat, datalabels):
    """训练， 建树过程， 递归实现, 假设每个特征使用一次"""
    labels = datalabels[:]
    classlist = [example[-1] for example in dataMat]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataMat[0]) == 1:
        return majorlity(dataMat)
    bestfeat = choosebestfeature(dataMat)
    bestfeatlabel = labels[bestfeat]
    mytree = {bestfeatlabel:{}}
    del(labels[bestfeat])
    featValues = [example[bestfeat] for example in dataMat]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        new_label = labels[:]    # copy,和C不一样， python 列表并不是传值参数，修改列表对后续结果会产生影响
        mytree[bestfeatlabel][value] = creatTree(splitdata(dataMat, bestfeat, value), new_label)
    return mytree

def storeTree(mytree, filename):
    """将训练好的模型进行保存"""
    fw = open(filename, 'w')
    pickle.dump(mytree, fw)
    fw.close()
    
def loadmodel(filename):
    """将保存好的模型导出，测试"""
    fw = open(filename)
    treemodel = pickle.load(fw)
    return treemodel

def classify(inputTree, featlabels, testVec):
    """测试模型效果"""
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featlabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classlabel = classify(secondDict[key], featlabels, testVec)
            else:
                classlabel = secondDict[key]
    return classlabel

if __name__ == '__main__':
    testdata = [[0, 1], [1, 1]]    # 'no' 'yes'
    dataMat, datalabels = loadData()
    mytree = creatTree(dataMat, datalabels)
    print(mytree)
    print('testing...')
    for i in range(len(testdata)):
        result = classify(mytree, datalabels, testdata[i])
        print(result)