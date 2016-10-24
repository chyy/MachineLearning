#coding:utf-8
__author__ = 'lyj'

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))-1
    dataMat =[]
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:   #计算行列式
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)   #.I是求逆
    #ws = linalg.solve(xTx, xMat.T*yMat)
    return ws

def lwlr(testPoint, xArr, yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T)/(-2.0*(k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights*yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat -= yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    #对数据进行标准化处理
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat -= yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    m,n=shape(xMat)
    returnMat = zeros((numIt,n))
    #初始化权重向量,为了实现贪心算法建立了ws的两份副本
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        #贪心算法在所有特征上运行两次for循环,分别计算增加或减少该特征对误差的影响
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                #eps表示每次迭代需要调整的步长
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
    returnMat[i,:] = ws.T


path = '/Users/lyj/Programs/BookCode/MachineLearningInAction/Ch08/abalone.txt'
xArr, yArr = loadDataSet(path)
res = stageWise(xArr,yArr,0.01,200)
print res



