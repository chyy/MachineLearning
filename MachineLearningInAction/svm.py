#coding:utf-8
__author__ = 'lyj'

#import os
from numpy import *
def loadDataSet(filepath):
    dataMat = []
    labelMat = []
    fr = open(filepath)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''选择alpha进行优化
'''
def selectJrand(i, m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

'''调整大于或者小于L的alpha值
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#数据集,类别标签,常数C,容错率,最大循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  #转置类别标签,标签的每一行对应数据矩阵的每一行
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1))) #构建alpha列矩阵,矩阵中的元素都初始化为0
    iter = 0  #纪录没有任何alpha改变的情况下遍历数据集的次数
    while (iter<maxIter):
        alphaPairsChanged = 0 #用于纪录alpha是否已经进行优化
        for i in range(m):   #遍历整个集合
            #multiply算的是两个列向量对应元素的乘积 (m*1).T * m*n * (1*n).T = 1*m * m*n * n*1
            fXi = float(multiply(alphas, labelMat).T *(dataMatrix*dataMatrix[i,:].T))+b  #预测的类别 w = sum(aiyixi)
            Ei = fXi - float(labelMat[i])  #误差
            #如果误差很大,对数据实例i所对应的alpha值进行优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i]<C)) or ((labelMat[i]*Ei > toler) and (alphas[i]>0)):
                j = selectJrand(i,m)   #随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                #计算第二个alpha的误差
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #保证alpha的范围在0到C之间,不能小于L,大于H
                if (labelMat[i]!=labelMat[j]): #如果实例i和j的标签不同
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+ alphas[j]- alphas[i])
                else:   #如果实例i和j的标签相同
                    L= max(0, alphas[j]+alphas[i]-C)
                    H = min(C,C+alphas[j]+alphas[i])
                #如果L和H相等,就不做任何改变,结束本次循环,运行下一次循环
                if L==H:print "L==H";continue
                #eta是alpha[j]的最优修改量, 2*a[i]*a[j]-a[i]*a[i]-a[j]*a[j];
                #如果eta为0,退出for循环的当前迭代过程(其实是计算出新的alpha就比较麻烦了,这里对该过程进行了简化)
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0: print "eta>=0";continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #检查alpha[j]是否有轻微的改变,如果有改变,就跳出循环对alpha[i]进行改变
                if (abs(alphas[j] - alphaJold) < 0.00001):print "j not moving enough"; continue
                #alpha[i]和alpha[j]同样进行改变,不过改变的方向正好相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #设置常数项b
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):b=b1
                elif (0<alphas[j]) and (C>alphas[j]):b=b2
                else: b = (b1+b2)/2.0
                alphaPairsChanged +=1

                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
        #检查alpha值是否做了更新,如果有更新则将iter设为0后,继续运行程序
        #只有在所有数据集上遍历maxIter次,且不再发生任何alpha修改之后,程序才会停止并退出while循环
        #因为alpha不修改时,iter才会自增1,所以迭代足够的次数之后,alpha的值都不再发生改变就退出循环
        if (alphaPairsChanged == 0): iter+=1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas
    print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)

dataArr,labelArr = loadDataSet("/Users/lyj/Programs/Codes&Data/MachineLearningInAction/Ch06/testSet.txt")
b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
mat1 = alphas[alphas>0]
mat2 = shape(alphas[alphas>0])
print mat1
print mat2
for i in range(100):
    if alphas[i]>0.0:print dataArr[i],labelArr[i]