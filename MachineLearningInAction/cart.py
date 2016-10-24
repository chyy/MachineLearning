#coding:utf-8
__author__ = 'lyj'

from numpy import *

def treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

def loadDataSet(filename):
    dataMat =[]
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

#生成叶节点,回归树中是目标变量的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    #用均方差乘以数据个数,等于总方差
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType = regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    #如果所有值相等则退出,其实是统计不同剩余特征值的数目
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #如果切分出的数据集很小则不切分
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):continue
            newS = errType(mat0)+errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少不大,则退出
    if (S-bestS)<tolS:
        return None, leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex, bestValue)
    #如果切分出后的数据集很小则退出
    if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #如果是回归树,该模型是一个常数
    #如果是模型树,该模型是一个线性方程
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    #判断是否满足停止条件,如果满足返回叶节点
    if feat == None: return val
    retTree = {}
    #待切分的特征
    #待切分的特征值
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #进行切分,然后得到左,右树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet,leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

#用于测试输入变量是否是一棵树,返回布尔类型的结果,即判断当前处理的节点是否是叶节点
def isTree(obj):
    return (type(obj).__name__ == 'dict')

#从上到下遍历树直到叶节点为止,如果找到两个叶节点则计算它们的平均值
def getMean(tree):
    if isTree(tree['right']):tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['left'])/2.0

#待剪枝的树与剪枝所需的测试数据
def prune(tree, testData):
    #如果没有测试数据,则对树进行塌陷处理
    if shape(testData)[0]==0: return getMean(tree)
    #检查某个分支是子树还是节点
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    #分支是子树,则对子树进行剪枝
    if isTree(tree['left']):tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):tree['right'] = prune(tree['right'], rSet)
    #如果两个分支不再是子树,则进行合并,合并的时候要对合并前后的误差进行比较,合并后比合并前的误差小就进行合并,否则直接返回
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1]-tree['left'],2)) + sum(power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNoMerge:
            print "merging"
            return treeMean
        else: return tree
    else:return tree


path = '/Users/lyj/Programs/BookCode/MachineLearningInAction/Ch09/ex00.txt'
myDat = loadDataSet(path)
result = createTree(myDat)
print result