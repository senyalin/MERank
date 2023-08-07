import json
import numpy as np
import random
from scipy.stats import weightedtau


with open("../result/MeRank.json", encoding = 'utf-8') as rk:
    rkp = json.load(rk)

with open("../result/GoldStandard.json", encoding = 'utf-8') as gd:
    gdp = json.load(gd)

def createRankList(rkp, gdp, topK = 999):
    rankList = []
    for i in rkp:
        tempList = []
        for j in rkp[i]:
            for k in gdp[i]:
                if j[0] == k[0] and j[0].find('=') != -1 and len(tempList) < topK:
                    tempList.append(k[1])
        rankList.append(tempList)
    return rankList


def calCorrectRate(rankList):
    scoreList = []
    for i in rankList:
        wrongPoint = 0
        for j in range(0, len(i)):
            if j + 1 < len(i):
                if i[j+1] > i[j]:
                    wrongPoint += 1
        scoreList.append((len(i) - wrongPoint)/len(i)*100)
    return round(sum(scoreList) / len(scoreList),3)



def ndcg(golden, current, n = -1):
    log2_table = np.log2(np.arange(2, 1002)) # The following parameter is for pre-setting the number of MEs. Setting it to a larger value is preferable.

    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n]
        dcg = np.sum(np.divide(np.power(2, rel) - 1, log2_table[:rel.shape[0]])) # np.power(2, rel) - 1
        return dcg
    ndcgs = []
    for i in range(len(current)):
        k = len(current[i]) if n == -1 else n
        idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k)
        dcg = dcg_at_n(current[i], n=k)
        tmp_ndcg = 0 if idcg == 0 else dcg / idcg
        ndcgs.append(tmp_ndcg)
    return 0 if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))

def calkendalltau(rankList):
    kdtList = []
    for item in rankList:
        sortedList = sorted(item, reverse = True)
        corr, _ = weightedtau(sortedList, item, rank=None)
        kdtList.append(corr)
#         print(item)
#         print(corr)
    return kdtList



def doShuffle(rankList):
    shuffleList = []
    for item in rankList:
        tmpeList = item.copy()
        random.shuffle(tmpeList)
        shuffleList.append(tmpeList)
    # print(shuffleList)
    return shuffleList


rankList = createRankList(rkp, gdp)

documentNum = len(rankList)
MeNum = 0
for i in rankList:
    MeNum += len(i)
print('documentNum = ', documentNum)
print('MeNum = ', MeNum)



kendalltauScore = calkendalltau(rankList)
print("our method kendall's tau = ", sum(kendalltauScore)/len(kendalltauScore))

test = []
tempScoreList = []
tempScore = 0

for i in range(50):
    tempScoreList = calkendalltau(doShuffle(rankList))
    tempScore += (sum(tempScoreList)/len(tempScoreList))

print("baseline kendall's tau = ", tempScore/50)



ndcgScoreat5 = ndcg(rankList, rankList,5)
ndcgScoreat10 = ndcg(rankList, rankList, 10)
ndcgScore = ndcg(rankList, rankList)

tempNdcgScore = 0
tempNdcgScoreat5 = 0
tempNdcgScoreat10 = 0
for i in range(50):
    tempNdcgat5 = ndcg(rankList, doShuffle(rankList),5)
    tempNdcgat10 = ndcg(rankList, doShuffle(rankList), 10)
    tempNdcg = ndcg(rankList, doShuffle(rankList))
    tempNdcgScoreat5 += tempNdcgat5
    tempNdcgScoreat10 += tempNdcgat10
    tempNdcgScore += tempNdcg

print("our method ndcg@5 = ", ndcgScoreat5)
print("our method ndcg@10 = ",ndcgScoreat10)
print("our method ndcg = ",ndcgScore)
print("----------------------------")
print("baseline ndcg@5 = ",tempNdcgScoreat5/50)
print("baseline ndcg@10 = ",tempNdcgScoreat10/50)
print("baseline ndcg = ",tempNdcgScore/50)