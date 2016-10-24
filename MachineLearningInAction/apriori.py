#coding:utf-8

#构建初始集合
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):  #构建大小为1的所有候选项集的集合
    C1 = []  #储存所有不重复的事项
    for transaction in dataSet:  #遍历数据集中的所有交易记录
        for item in transaction:  #遍历记录中的每一个项
            if [item] not in C1:  #添加没有出现的物品项,并且是作为一个列表添加,为每个物品项构建一个集合.
                C1.append([item])  #python不能创建只有一个整数的集合,必须使用列表,后续要做集合操作.
    C1.sort()   #对大列表进行排序,并将其中的每个单元素列表映射到frozenset()
    return map(frozenset, C1)  #对C1中每个项构建一个不变集合,后续作为字典健值使用

#从C1生成L1,返回包含支持度值的字典备用
def scanD(D, Ck, minSupport):  #记录数据集Ck,包含候选集合的列表, 感兴趣项集的最小支持度minSupport
    ssCnt = {}  #创建空字典
    for tid in D:  #遍历数据中的所有交易记录
        for can in Ck:  #遍历C1中的所有候选集
            if can.issubset(tid):  #如果C1中的集合是记录的一部分,那么增加字典中对应的计数值,这里字典的键就是集合
                if not ssCnt.has_key(can):ssCnt[can] = 1
                else:ssCnt[can] += 1
    #当扫描完数据集中的所有项以及所有候选集时,就需要计算支持度。不满足最小支持度要求的集合不会输出。
    numItems = float(len(D))  #记录数
    retList = []  #函数也会先构建一个空列表,该列表包含满足最小支持度要求的集合。
    supportData = {}
    for key in ssCnt:  #循环遍历字典中的每个元素并且计算支持度。如果支持度满足最小支持度要求,则将字典元素添加到retList中 。
        support = ssCnt[key]/numItems  #计算所有项集的支持度,次数/总次数
        if support >= minSupport:  #如果支持度满足最小支持度要求,则将字典元素添加到retList中
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData  #返回最频繁项集的支持度supportData

#主要是根据频繁项集Lk,创建候选项集Ck,创建的项集元素个数为k
def aprioriGen(Lk, k):  #creates Ck, 输入参数为频繁项集列表Lk,项集元素个数k
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            #频繁项集Lk的第i个元素(集合)和后面的若干个元素(集合)的前k-2项
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:  #如果两个集合的前k-2个元素相同时,将两个集合合并
                retList.append(Lk[i] | Lk[j])  #集合的并
    return retList

#生成候选集列表
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)  #使用aprioriGen()来创建Ck
        Lk, supK = scanD(D,Ck,minSupport)  #扫描数据集,从Ck得到Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData   #所有的候选集列表,以及支持度

def generateRules(L, supportData, minConf=0.7):  #频繁项集列表,包含频繁项集支持数据的字典,最小可信度阀值
    bigRuleList = []  #包含可信度的规则列表,可以基于可信度对它们进行排序
    for i in range(1,len(L)):  #从包含两个或者更多元素的项集开始规则构建过程
        for freqSet in L[i]:  #遍历L中的每一个频繁项集,并对每个频繁项集创建只包含单个元素集合的列表H1
            H1 = [frozenset([item]) for item in freqSet]
            if (i>1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  #如果项集中只有两个元素,那么使用函数calcConf()来计算可信度值。
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)  #计算频繁项集对应的规则
    return bigRuleList

#计算规则的可信度,对规则进行评估,以及找到满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []  #满足最小可信度要求的规则列表
    for conseq in H:  #遍历H中的所有项集,并计算可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq]  #计算的时候,使用supportData中的支持度数据
    if conf>=minConf:
        print freqSet-conseq, '-->', conseq, 'conf:', conf
        br1.append((freqSet-conseq, conseq, conf))
        prunedH.append(conseq)
    return prunedH

#从最初的项集中,生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])  #频繁项集大小m
    if (len(freqSet) > (m+1)):  #尝试进一步合并,査看该频繁项集是否大到可以移除大小为m的子集
        Hmp1 = aprioriGen(H, m+1)  #生成H中元素的无重复组合,结果存储在Hmp1中,这也是下一次迭代的H列表,创建Hm+1条新候选规则
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)  #测试它们的可信度以确定规则是否满足要求
        if (len(Hmp1) > 1):  #如果不止一条规则满足要求,那么使用Hmp1迭代调用函数来判断是否可以进一步组合这些规则。
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

dataSet = loadDataSet()
C1 = createC1(dataSet)
D = map(set, dataSet)
L1, suppData0 = scanD(D, C1, 0.5)
L,suppData = apriori(dataSet, minSupport=0.5)
rules = generateRules(L, suppData, minConf=0.7)
print rules
