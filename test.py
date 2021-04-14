from sklearn.ensemble import RandomForestClassifier
import re
import math


domainList = []#训练的域名对象列表
InputList = []#测试的域名矩阵
InputDomainName = []#测试的域名的名称
InputLabel = []#测试的域名的预测结果



def cal_entropy(text):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = text.lower()
    for i in range(len(text)):
        if text[i].isalpha():
            letter[ord(text[i]) - ord('a')] += 1
            sum += 1
    #print('\n', letter)
    if sum==0:
        return 1
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -(p * math.log(p, 2))
    return h

class Domain:
    def __init__(self,_domainName,_label,_genName):
        self.domainName = _domainName#完整域名
        self.label = _label#标签
        self.genName = _genName#最前部分的域名
        self.genNameLength = len(self.genName)#最前的长度
        self.numbers = len(re.findall('\d+',self.genName))#数字的数量
        self.domainNameEntropy = cal_entropy(self.genName)#字母熵
        self.segmentationn = len(re.findall('\.',self.domainName))#分段，及点数

    def return_value(self):
        return [self.genNameLength,self.numbers,self.domainNameEntropy,self.segmentationn]

    def return_label(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

def initData(filename):
    #with open(filename).read().splitlines() as f:
        f = open(filename).read().splitlines()
        for line in f:
            line = line.strip()
            tokens = line.split(',')
            tmp_domainName = domainName = tokens[0]
            tmp_domainName = tmp_domainName.strip()
            tmp_tokens = tmp_domainName.split('.')
            genName = tmp_tokens[0]
            label = tokens[1]
            domainList.append(Domain(domainName,label,genName))


def InputData(filename):
    #with open(filename).read().splitlines() as f:
        f = open(filename).read().splitlines()
        for line in f:
            strDomain = line#域名
            line = line.strip()
            tokens = line.split('.')
            genName = tokens[0]
            genNameLength = len(genName)#长度
            numbers = len(re.findall('\d+',genName))
            genNameEntropy = cal_entropy(genName)#
            segmentation = len(re.findall('\.',strDomain))
            InputList.append([genNameLength,numbers,genNameEntropy,segmentation])
            InputDomainName.append(strDomain)


def transformList(predictList):
    for i in predictList:
        if i == 0:
            InputLabel.append("notdga")
        else:
            InputLabel.append("dga")


def main():
    initData("train.txt")
    '''
    for i in range(len(domainList)):
        print(domainList[i].label,domainList[i].genNameLength,domainList[i].numbers,domainList[i].domainNameEntropy,domainList[i].segmentationn)
    '''
    featureMatrix = []
    labelList = []
    for items in domainList:
        featureMatrix.append(items.return_value())
        labelList.append(items.return_label())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    InputData("test.txt")
    predictList = clf.predict(InputList)
    transformList(predictList)

    file = open("result.txt",mode='w')
    for i in range(len(predictList)):
        #print(InputDomainName[i]+'\t',end="")
        #print(InputLabel[i])
        file.write(InputDomainName[i]+','+InputLabel[i]+'\n')

if __name__ == "__main__":
    main()