#coding:utf-8

__author__ = 'lyj'

import operator
from numpy import *
from os import listdir

#读取数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat, classLabelVector

'''#def createDataSet():
#    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
#    labels = ['A', 'A', 'B', 'B']
#    return group, labels'''

#对数据进行归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet /tile(ranges,(m,1))
    return normDataSet, ranges, minVals

#构建分类器进行预测
def classify0(inX, dataSet,labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  #tile复制inX,以(dataSetSize, 1)扩张
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest(path,percent, k):
    hoRatio = percent
    datingDataMat, datingLabels = file2matrix(path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
        print "The classifier came back with: %d, the real answer is: %d" % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]):errorCount += 1.0
    print "The total error rate is: %f" % (errorCount/float(numTestVecs))

def classifyPerson(path, k):
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("Percentage of time spent playing video games?"))
    ffMiles = float(raw_input("Frequent flier miles earned per year?"))
    iceCream = float(raw_input("Liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix(path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels,k)
    print "You will probably like this person: ", resultList[classifierResult-1]


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest(path1,path2,k):
    hwlabels = []
    trainingFileList = listdir(path1)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwlabels.append(classNumStr)
        trainingMat[i:] = img2vector(path1+('/%s' % fileNameStr))
    testFileList = listdir(path2)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(path2+('/%s' % fileNameStr))
        classifierResult = classify0(vectorUnderTest,trainingMat,hwlabels,k)
        print "The classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nThe total number of errors is: %d" % errorCount
    print "\nThe total error rate is: %f" % (errorCount/float(mTest))

#group, labels = createDataSet()
#label = classify0([0,0], group, labels, 3)
#print(label)

path0 = "/Users/lyj/Programs/machinelearninginaction/Ch02/datingTestSet2.txt"
#classifyPerson(path0, 3)

path1 = '/Users/lyj/Programs/machinelearninginaction/Ch02/digits/trainingDigits'
path2 = '/Users/lyj/Programs/machinelearninginaction/Ch02/digits/testDigits'
#testVector = img2vector(path2)
#print(testVector.shape[1])
#print(testVector[0,0:31])
handwritingClassTest(path1,path2,3)
