import numpy as np
from numpy import *
import datetime as dt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj



class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(j, oS, E2):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = -1
    for k in range(oS.m):   #loop through valid Ecache values and find the one that maximizes delta E
        if k == j: continue #don't calc for i, waste of time
        if((oS.alphas[k] < oS.C) and (oS.alphas[k] > 0)):
            Ek = calcEk(oS, k)
            deltaE = abs(E2 - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; 
    return maxK


def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
    
def takeStep(oS, i, j):
    alph1 = oS.alphas[i].copy()
    alph2 = oS.alphas[j].copy()
    #找到αi αj对应的标签
    y1 = oS.labelMat[i]
    y2 = oS.labelMat[j]
    E1 = calcEk(oS, i)
    E2 = calcEk(oS, j)
    #计算核函数K
    K11 = oS.X[i,:]*oS.X[i,:].T
    K22 = oS.X[j,:]*oS.X[j,:].T
    K12 = oS.X[i,:]*oS.X[j,:].T
    Eta = K11 + K22 - 2.0 * K12  #计算η
    s = y1 * y2
    if i == j :    #如果内层和外层循环选择的是同一个α，则返回重选
        return 0
    #计算L和H
    if (oS.labelMat[i] != oS.labelMat[j]):
        L = max(0, alph2 - alph1)
        H = min(oS.C, oS.C + alph2 - alph1)
    else:
        L = max(0, alph2 + alph1 - oS.C)
        H = min(oS.C, alph2 + alph1)
    if L==H:
        print("L==H")
        return 0
    #根据Eta的情况计算剪辑后的αj
    if (Eta > 0):
        oS.alphas[j] += oS.labelMat[j] * (E1 - E2) / Eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
    else:  #论文公式（19）
        f1 = y1 * (E1 * oS.b) - oS.alphas[i] * K11 - s * oS.alphas[j] * K12
        f2 = y2 * (E2 * oS.b) - s * oS.alphas[i] * K12 - oS.alphas[j] * K22
        L1 = oS.alphas[i] + s * (oS.alphas[j] - L)
        H1 = oS.alphas[i] + s * (oS.alphas[j] - H)
        FunL = L1 * f1 + L * f2 + 0.5 * L1 * L1 * K11 + 0.5 * L * L * K22 + s * L * L1 * K12
        FunH = H1 * f1 + H * f2 + 0.5 * H1 * H1 * K11 + 0.5 * H * H * K22 + s * H1 * H * K12
        if FunL < FunH - oS.tol:
            oS.alphas[j] = L
        elif FunL > FunH + oS.tol:
            oS.alphas[j] = H
        else:
            oS.alphas[j] = alph2
    updateEk(oS, j)
    if (abs(oS.alphas[j] - alph2) < 0.00001 * (oS.alphas[j] + alph2 + 0.00001)):
        return 0
    oS.alphas[i] = alph1 + s * (alph2 - oS.alphas[j])
    updateEk(oS, i)
    # 计算b
    b1 = oS.b - E1- y1*(oS.alphas[i]-alph1)*K11 - y2*(oS.alphas[j]-alph2)*K12
    b2 = oS.b - E2- y1*(oS.alphas[i]-alph1)*K12 - y2*(oS.alphas[j]-alph2)*K22
    if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
        oS.b = b1
    elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
        oS.b = b2
    else: 
        oS.b = (b1 + b2)/2.0
    return 1


    # 首先判断i2是否满足KKT条件，如果不满足，
    # 则根据启发式规则再选择i1样本，
    # 然后更新i和j的拉格朗日乘子。
def innerL(j, oS):
    E2 = calcEk(oS, j)  
    if ((oS.labelMat[j]*E2 < -oS.tol) and (oS.alphas[j] < oS.C)) or ((oS.labelMat[j]*E2 > oS.tol) and (oS.alphas[j] > 0)):
        i = selectJ(j, oS, E2) #this has been changed from selectJrand
        if i == -1: #第一轮执行，令i随机取值
            i = int(random.uniform(0,oS.m))
            i = selectJrand(j,oS.m)
            takeStep(oS,i,j)
            return 1
        else:
            takeStep(oS,i,j)
            return 1
    return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    # while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
    while ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] #acquire indexes of 0<alpha[i]<C
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w


'''
除了文件路径，请不要修改下面的代码
'''
dataArr, labelArr = loadDataSet('Ch6_testSet.txt')
starttime = dt.datetime.now()
b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
endtime = dt.datetime.now()
print("totol run time: %d" % (endtime-starttime).seconds)
print("alphas:")
print(alphas[alphas>0])
print(nonzero(alphas.A>0)[0])
ws = calcWs(alphas, dataArr, labelArr)
print("ws:")
print(ws)

m = len(dataArr)
X = mat(dataArr)
labelMat = mat(labelArr).transpose()
tol = 0.001; C = 0.6
for i in range(m):
    fXi = float(multiply(alphas,labelMat).T*(X*X[i,:].T)) + b
    Ei = fXi - float(labelMat[i])
    if ((labelMat[i]*Ei < -tol) and (alphas[i] < C)) or ((labelMat[i]*Ei > tol) and (alphas[i] > 0)):
        print("violate KKT: %d" % i)
        print(X[i,:], labelMat[i], Ei)
