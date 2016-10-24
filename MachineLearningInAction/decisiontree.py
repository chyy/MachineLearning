#coding:UTF-8
__author__ = 'lyj'
from math import log
import operator
import pickle

def createDataSet( ):
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def clacShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for vec in dataset:
        currentLabel = vec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0])-1
    baseEntropy = clacShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataSet(dataset,i,value)
            prob = len(subDataset) /float(len(dataset))
            newEntropy += prob*clacShannonEnt(subDataset)
        infoGain = baseEntropy-newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

#对标签进行统计,返回出现次数最多的标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0])==1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    #labels存储的是所有特征的标签,主要给出数据明确的含义
    myTree = {bestFeatLabel:{}}  #用字典类型存储树的信息
    #print len(labels)
    del(labels[bestFeat])
    #print len(labels)
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    #遍历当前选择的特征包含的所有属性值,在每个数据集划分上递归调用函数createTres()
    for value in uniqueVals:
        subLabels = labels[:]  #剩下的特征
        #得到的返回值被插入到字典变量myTree中
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat,value), subLabels)
    return myTree

#测试和存储分类器
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]   # no surfacing
    secondDict = inputTree[firstStr]   #{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    featIndex = featLabels.index(firstStr)  #取得特征名对应的特征序号
    for key in secondDict.keys():
        if testVec[featIndex] == key:  #判断测试向量对应的特征的值的走向
            if type(secondDict[key]).__name__=='dict':   #如果对应的特征的值是个字典,那么递归调用该函数
                classLabel = classify(secondDict[key],featLabels, testVec)
            else: classLabel = secondDict[key]    #如果对应的特征的值不是个字典,那么取出该特征对应的值
    return classLabel

#保存决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

myDat, labels = createDataSet()
print labels

myTree = createTree(myDat, labels)
print myTree

#print classify(myTree, ['no surfacing', 'flippers'], [1,0])