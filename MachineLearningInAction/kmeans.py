#coding:utf-8
__author__ = 'lyj'

from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))

#构建簇质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))  #构建k个n维的行向量,初始化为0
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1) #生成5个0-1的随机数
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  #确定数据集中的数据点的总数
    clusterAssment = mat(zeros((m,2)))  #存储每个点的簇分配结果,[簇索引值,列存储误差]
    centroids = createCent(dataSet, k)  #初始化聚类中心
    clusterChanged = True   #判断数据点的簇分配是否还在改变,如果还在改变,继续迭代
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  #对数据集中的每个数据点
            minDist = inf
            minIndex = -1
            for j in range(k):   #对每个质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])  #计算点到质心的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex :  #如果任一点的簇分配结果发生改变,更新clusterChanged标志
                clusterChanged=True
            clusterAssment[i,:] = minIndex, minDist**2  #更新数据i的簇分配结果,和误差值
        print centroids
        #遍历所有质心,并更新它们的取值
        for cent in range(k):  #通过数组过滤来获得给定簇的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:]=mean(ptsInClust, axis=0)  #然后计算所有点的均值,axis = 0 表示沿列方向进行均值计算
    return centroids, clusterAssment  #返回所有的类质心和点分配结果

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    #创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m,2)))
    #创建一个初始类簇,计算整个数据集的质心,并使用一个列表保留所有的质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0]
    #遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    #对簇进行划分,直到得到想要的簇数目,用簇列表中的值来获得当前簇的数目
    while (len(centList) < k):
        lowestSSE = inf
        #遍历所有的簇,来决定最佳的簇进行划分,因此需要比较划分前后的SSE
        for i in range(len(centList)):
            #尝试划分每一簇
            #一开始把簇中的所有点看成一个小的数据集ptsInCurrCluster,然后进行k=2的划分
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])  #计算每个簇的误差值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])  #剩余数据集的误差
            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit
            #这些误差与剩余数据集的误差之和作为本次划分的误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                #划分操作,只要把要划分分簇中所有点的簇分配结果进行修改
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #数组过滤器
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'The bestCentToSplit is: ',bestCentToSplit
        print 'The len of bestCentToSplit is: ',len(bestClustAss)
        #新的簇分配结果被更新,新的质心会被添加到centList中
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    #返回质心列表和簇分配结果
    return mat(centList), clusterAssment

path = '/Users/lyj/Programs/BookCode/MachineLearningInAction/Ch10/testSet2.txt'
dataMat = mat(loadDataSet(path))  #记住这里是mat矩阵
#print(randCent(dataMat, 2))
#print distEclud(dataMat[0], dataMat[1])
#myCentroids, clusterAssing = kMeans(dataMat, 4)
#print myCentroids #, clusterAssing
#print '---------------------------'
centList, myNewAssment = biKmeans(dataMat,3)
print myNewAssment

