#coding:utf-8
__author__ = 'lyj'

from numpy import *
import re
import feedparser

def loadDataSet( ):
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park','stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'at', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

'''创建词汇表
   @:param list dataSet是一个列表,里面的列表也是列表
   @:return list 返回字典
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet |= set(document)  #集合的并
    return list(vocabSet)

'''文本表示
   @:param list, list 输入词汇表和某个文本
   @:return list  返回该文本的向量表示,1,0表示法
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print "The word: %s is not in my Vocabulary!" % word
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

'''从词向量计算概率,求p(ci),p(w|ci)
   @:param list, list
   @:return list,list,list
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0num = ones(numWords); p1num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]   #向量相加
            p1Denom += sum(trainMatrix[i])   #统计类别下总的词数
        else:
            p0num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = log(p1num/p1Denom)
    p0Vec = log(p0num/p0Denom)
    return p0Vec, p1Vec, pAbusive

#预测一个向量的类别
def classifyNB(vec2Classify, p0Vec, p1Vec, pAbusive):
    p1 = sum(vec2Classify*p1Vec)+log(pAbusive)
    p0 = sum(vec2Classify*p0Vec)+log(1.0 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()  #得到数据文件
    myVocabList = createVocabList(listOPosts)  #得到词汇表
    trainmat = []
    for postinDoc in listOPosts:
        trainmat.append(setOfWords2Vec(myVocabList,postinDoc))   #得到文本表示
    p0V,p1V,pAb = trainNB0(array(trainmat),array(listClasses))   #训练分类器
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))  #表示测试文件
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry1 = ['stupid','garbage']
    thisDoc1 = array(setOfWords2Vec(myVocabList,testEntry1))  #表示测试文件
    print testEntry1,'classified as: ',classifyNB(thisDoc1,p0V,p1V,pAb)

def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

#垃圾邮件分类器
def spamTest():
    docList = [];classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open(r'/Users/lyj/Programs/Codes&Data/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(r'/Users/lyj/Programs/Codes&Data/machinelearninginaction/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):   #留存交叉验证
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) !=classList[docIndex]:
            errorCount +=1
    print 'The error rate is: ', float(errorCount)/len(testSet)

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1),reverse = True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) !=classList[docIndex]:
            errorCount +=1
    print 'The error rate is: ', float(errorCount)/len(testSet)
    return vocabList, p0V,p1V

def getTopWords(ny,sf):
    vocabList, p0V,p1V = localWords(ny,sf)
    topNY = []; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **"
    for item in sortedNY:
        print item[0]

ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
getTopWords(ny,sf)
# if __name__ =='__main__':
#     spamTest()

