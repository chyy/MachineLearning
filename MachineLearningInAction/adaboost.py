#coding:utf-8
__author__ = 'lyj'
from numpy import *

def loadDataSet():
    datMat = matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3, 1. ],
                     [1. , 1. ],
                     [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

#单层决策树生成函数
#通过阀值比较对数据进行分类,阀值一侧的数据分类到类别-1,另一侧分类到类别1
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))  #列向量
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#遍历stumpClassify函数的所有可能值,并找到数据集上最佳的单层决策树,即最佳的权重向量D
def buildStump(dataArr, classLabels, D):
    dataMatrix=mat(dataArr)
    labelMat = mat(classLabels).T
    m,n =shape(dataMatrix)
    numSteps = 10.0
    #存储给定权重向量D时,得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    #三层嵌套循环,第一层遍历数据集的所有特征
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        #考虑使用多大的步长
        stepSize = (rangeMax-rangeMin)/numSteps
        #对每一步
        for j in range(-1,int(numSteps)+1):
            #在大于和小于之间切换不等式
            for inequal in ['lt','gt']:
                #通过步长,计算阀值
                threshVal = (rangeMin+float(j)*stepSize)
                #给定阀值,对分类结果进行预测
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                #计算预测结果和真实结果的误差,行向量*列向量,是基于权重向量D来评价的
                #如果要使用其它分类器,需要考虑D上最佳分类器所定义的计算过程
                weightedError = D.T*errArr
                print "split dim %d, thresh %.2f, thresh ineqal %s, the weighted error is %.3f" % \
                      (i, threshVal, inequal, weightedError)
                if weightedError<minError:
                    minError = weightedError
                    #类别估计值
                    bestClasEst = predictedVals.copy()
                    #用字典bestStump存储最佳单层决策数的信息
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    #单层决策树数组,存储每次得到的最佳弱分类器
    weakClassArr = []
    m = shape(dataArr)[0]
    #初始化权重向量
    D = mat(ones((m,1))/m)
    #初始化累计类别估计值
    aggClassEst = mat(zeros((m,1)))
    #迭代找出单层决策树
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D: ", D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "ClassEst: ",classEst.T
        #正确为-alpha,错误为alpha
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "Total error: ", errorRate, '\n'
        if errorRate == 0.0: break
    return weakClassArr

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

datMat, classLabels = loadDataSet()
d = mat(ones((5,1))/5)
tree= buildStump(datMat, classLabels, d)
result = adaBoostTrainDS(datMat, classLabels, 30)
results = adaClassify([0,0], tree)
print result
