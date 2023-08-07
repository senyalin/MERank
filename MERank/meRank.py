from sentence_utils import sentence_segmentation
from sentence_utils import rm_empty
from sentence_utils import latex2mi
from sentence_utils import printMe
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import os
import json

def freqMo(content):
    """ 𝑓𝑟𝑒𝑞 - html ver (the occurrence of Mo throughout the whole document) """
    startPos = 0
    endPos = 0
    tempString = ""
    freqBox = {}

    if content:
        for line in content:
            while line.find('<latex>') != -1:
              startPos = line.find('<latex>')
              line = line[startPos + 7 :]
              endPos = line.find('<latex>')
              tempString = line[: endPos]
              if tempString in freqBox: 
                freqBox[tempString] += 1
              else:
                freqBox[tempString] = 1

              """ init for next run """
              startPos = 0
              endPos = 0
              tempString = ""
              line = line[line.find('<latex>') + 7:]
        
        tempBox = {}
        for i in freqBox:
            tempBox[i] = freqBox[i]
        freqBox = sorted(tempBox.items(), key=lambda k : k[1], reverse=True)
        
        return freqBox
    else:
        return None

def densMo(content):
    """ 𝑑𝑒𝑛𝑠 - html ver (the number of MO in the located sentence) """
    densBox = {}
    densStartPos = 0
    densEndPos = 0
    densTempString = ""
    densCount = 0
    densTempList = []
    
    if content:
        for t in content:
            """ select <latex>MO<latex> in a sentence """
            while t.find('<latex>') != -1: 
                densStartPos = t.find('<latex>')
                t = t[densStartPos + 7 :]
                densEndPos = t.find('<latex>')
                densTempString = t[: densEndPos]
                densCount += 1
                densTempList.append(densTempString)

                """ init for next run """
                densStartPos = 0
                densEndPos = 0
                densTempString = ""
                t = t[t.find('<latex>') + 7:]

            for item in densTempList:
                if item in densBox:
                    densBox[item] = densCount if densCount > densBox[item] else densBox[item]
                else:
                    densBox[item] = densCount
            densTempList = []
            densCount = 0
            
        tempBox = {}
        for i in densBox:
            tempBox[i] = densBox[i]
        densBox = sorted(tempBox.items(), key=lambda k : k[1], reverse=True)
        
#         for item in densBox:
#             print(item)
        return densBox
    else:
        return None

""" CMI problem """

def get_token_pr(target_token, token_list):
    target_total_pr = 0
    for token in token_list:
        target_total_pr += token.count(target_token)
    return target_total_pr
def get_token_freq(target_token, token_list):
    return token_list.count(target_token)
def cal_pmi(token_list, target):
#    alpha = 1e-10
    target_pr = get_token_freq(target, token_list) # get_token_pr(target, token_list)
    best_pmi = -10000
    best_idx = 0
    if "\\" in target:
        best_pmi = 1.0
    for i in range(1, len(target)):
        first_half = target[:i]
        second_half = target[i:]  
        if len(token_list) == 0:
            p_target = 0
            p_first = 0
            p_second = 0
        else:
            p_target = (target_pr) / len(token_list)
            p_first = (get_token_freq(first_half, token_list)) / (len(token_list))
            p_second = (get_token_freq(second_half, token_list)) / (len(token_list))
        if p_first * p_second == 0:
            tmp_pmi = 0
        else:
            tmp_pmi = np.log2(p_target / (p_first * p_second))
        if tmp_pmi > best_pmi:
            best_pmi = tmp_pmi
            best_idx = i
    # print("After Splitting: ", (target[:best_idx], target[best_idx:]))
    return [round(float(best_pmi), 3), best_idx]

def cal_mci(all_token_list, target_token):
    """
    This code estimate the target MO whether it is a mci or not,
    and find the best splitting point if possible
    """
    # alpha = 1e-10
    best_mci = 0
    best_idx = 0

    if "\\" in target_token:
        best_mci = 1.0
    else:
        for i in range(1, len(target_token)):
            first_half = target_token[:i]

            second_half = target_token[i:]


            freq_whole = 2*(get_token_freq(target_token, all_token_list))
            freq_first = get_token_freq(first_half, all_token_list)
            freq_second = get_token_freq(second_half, all_token_list)
            
            if freq_first + freq_second == 0:
                mci_prob = 1
            else:
                mci_prob = (freq_whole / (freq_first+freq_second))
            mci_prob = 1 if mci_prob > 1 else mci_prob
            
            """ old version """
            # mci_prob = (freq_whole / (freq_first+freq_second)*(alpha)/(alpha+all_token_list.count(target_token)))
            # if mci_prob < delta or mci_prob < best_mci:
            
            if mci_prob > best_mci:
                best_mci = mci_prob
                best_idx = i
            # print("{} MCI score at {}: ({} , {}), ({} , {})".format(target_token, i, mci_prob, best_mci,target_token[:i], target_token[i:]))
            # print("======================")
    # print("After Splitting: ", (target_token[:best_idx], target_token[best_idx:]))
    return [round(float(best_mci), 3), 0 if best_mci == 1 else best_idx]

"""
Higher ISS scores denote better segmentation certainty. 
If score exceeds the threshold, segmentation is required, otherwise not. 
For targets over length 2, the highest scoring term is recursively segmented.
"""
def cal_iss(all_token_list, target_token, threshold): # Identifier Segmentation Score
    best_iss = 0
    best_idx = 0
    result = []
    
    if "\\" in target_token or len(target_token) == 1:
        best_iss = 0
    else:
        for i in range(1, len(target_token)):
            first_half = target_token[:i]
            second_half = target_token[i:]
            
            freq_whole = get_token_freq(target_token, all_token_list)
            freq_first = get_token_freq(first_half, all_token_list)
            freq_second = get_token_freq(second_half, all_token_list)
            
            if freq_whole == 0 and (freq_first == 0 or freq_second == 0):
                iss_prob = -1
            else: #(freq_whole + freq_first * freq_second)
                if freq_whole == 0:
                    freq_whole = freq_whole + 0.1
                iss_prob = ((freq_first+freq_second) / (freq_whole))

            if iss_prob > best_iss and iss_prob > 0:
                best_iss = iss_prob
                best_idx = i   
            elif iss_prob <= 0:
                best_iss = iss_prob
                best_idx = 0 
                break
                
#             print("{} ISS at {}: (iss_prop = {} , best_iss = {}), ({} , {})"\
#                   .format(target_token, i, iss_prob, best_iss,target_token[:i], target_token[i:]))
#             print("======================")
    if best_iss <= threshold: # Set the threshold as needed.
#         print("Don\'t need to Split: " , target_token)
        result.append(target_token)
    else:
        if best_idx == 0 or len(target_token) == 1:
            inner_best_iss1 = 0
            inner_best_iss2 = 0
        else:
            inner_result1, inner_best_iss1, inner_best_idx1 = cal_iss(all_token_list, target_token[:best_idx], threshold)
            inner_result2, inner_best_iss2, inner_best_idx2 = cal_iss(all_token_list, target_token[best_idx:], threshold)

        if inner_best_iss1 <= threshold and inner_best_iss2 <= threshold:
#             print("After Splitting: " , target_token[:best_idx], ',', target_token[best_idx:])
            result.extend([target_token[:best_idx], target_token[best_idx:]])
        elif inner_best_iss1 > threshold and inner_best_iss2 <= threshold:
#             print("After Splitting: ", target_token[:best_idx][:inner_best_idx1], ','\
#                                      , target_token[:best_idx][inner_best_idx1:], ','\
#                                      , target_token[best_idx:])
            result.extend([target_token[:best_idx][:inner_best_idx1], target_token[:best_idx][inner_best_idx1:], target_token[best_idx:]])
        elif inner_best_iss1 <= threshold and inner_best_iss2 > threshold:
#             print("After Splitting: ", target_token[:best_idx], ','\
#                                      , target_token[best_idx:][:inner_best_idx2], ','\
#                                      , target_token[best_idx:][inner_best_idx2:])
            result.extend([target_token[:best_idx], target_token[best_idx:][:inner_best_idx2], target_token[best_idx:][inner_best_idx2:]])
        else:
#             print("After Splitting: ", target_token[:best_idx][:inner_best_idx1], ','\
#                                      , target_token[:best_idx][inner_best_idx1:], ','\
#                                      , target_token[best_idx:][:inner_best_idx2], ','\
#                                      , target_token[best_idx:][inner_best_idx2:])
            result.extend([target_token[:best_idx][:inner_best_idx1], target_token[:best_idx][inner_best_idx1:], target_token[best_idx:][:inner_best_idx2], target_token[best_idx:][inner_best_idx2:]])
#     print(best_idx)
#     print(target_token)
#     print("======================")
    return result, round(float(best_iss), 3), 0 if best_iss < 0 else best_idx

def degrMo(allLatex, word_token_list, function = "PMI"):
    """ 𝑑𝑒𝑔𝑟 - html ver (the number of MI the ME depends on) """
    function = function.upper()
    degrBox = {}
    for i in allLatex:
        tempList = []
        tempList.append(i)
        degrBox[i] = latex2mi(tempList)
    
    # for i in degrBox:print(i, '=', degrBox[i])

    tempDict = {}
    cmiList = []
    for i in degrBox:
        degr = 0
        tempList = degrBox[i]
        for j in tempList:
#             degr += Counter(word_token_list)[j] - 1
            if function == "ISS":
                """ cmi problem using iss """
                threshold = 1
                result, best_iss, split_index = cal_iss(word_token_list, j, threshold)
                if best_iss > threshold and len(j) > 1:
                    for term in result:
                        degr += Counter(word_token_list)[term] - 1
                    cmiList.append(j)
    #                 print(j, '->', j[:split_index],Counter(word_token_list)[j[:split_index]]\
    #                       , '+', j[split_index:], Counter(word_token_list)[j[split_index:]], cal_iss(word_token_list, j)[0])
                else:
    #                 print(cal_iss(word_token_list, j)[0])
                    degr += Counter(word_token_list)[j] - 1
            elif function == "PMI":
                """ cmi problem using pmi """
                if cal_pmi(word_token_list, j)[0] < 0 and len(j) > 1:
                    split_index = cal_pmi(word_token_list, j)[1]
                    degr += Counter(word_token_list)[j[:split_index]]                         +Counter(word_token_list)[j[split_index:]] - 2
                    cmiList.append(j)
#                     print(j, '->', j[:split_index],Counter(word_token_list)[j[:split_index]]\
#                           , '+', j[split_index:], Counter(word_token_list)[j[split_index:]], cal_pmi(word_token_list, j)[0])
                else:
    #                 print(cal_pmi(word_token_list, j)[0])
                    degr += Counter(word_token_list)[j] - 1
            elif function == "MCI":
                """ cmi problem using mci """
                if cal_mci(word_token_list, j)[0] < 1 and len(j) > 1:
                    split_index = cal_mci(word_token_list, j)[1]
                    degr += Counter(word_token_list)[j[:split_index]]                         +Counter(word_token_list)[j[split_index:]] - 2
                    cmiList.append(j)
    #                 print(j, '->', j[:split_index],Counter(word_token_list)[j[:split_index]]\
    #                       , '+', j[split_index:], Counter(word_token_list)[j[split_index:]], cal_mci(word_token_list, j)[0])
                else:
    #                 print(cal_mci(word_token_list, j)[0])
                    degr += Counter(word_token_list)[j] - 1
        tempDict[i] = degr
        
    degrBox = sorted(tempDict.items(), key=lambda k : k[1], reverse=True)
    return degrBox, set(cmiList)

def lengMo(allLatex, word_token_list, function = "PMI"):
    """ 𝑙𝑒𝑛𝑔 - html ver (the number of mathematical identifiers in an ME) """
    function = function.upper()
    lengBox = {}
    for i in allLatex:
        tempList = []
        tempList.append(i)
        lengBox[i] = latex2mi(tempList)

    tempDict = {}
    for i in lengBox:
        leng = 0
        tempList = lengBox[i]
        for j in tempList:
#             leng += 1
            if function == "ISS":
                """ cmi problem using iss """
                threshold = 1
                result, best_iss, split_index = cal_iss(word_token_list, j, threshold)
                leng += len(result)
            elif function == "PMI":
                """ cmi problem using pmi """
                if cal_pmi(word_token_list, j)[0] < 0 and len(j) > 1:
                    leng += 2
                else:
                    leng += 1
            elif function == "MCI":
                """ cmi problem using mci """
                if cal_mci(word_token_list, j)[0] < 1 and len(j) > 1:
                    leng += 2
                else:
                    leng += 1
        tempDict[i] = leng
        
    lengBox = sorted(tempDict.items(), key=lambda k : k[1], reverse=True)
    return lengBox


""" cal score for each metrics """

def calZscore(score, avg, std = 1e-10):
    std = 1e-10 if std == 0 else std
    return (score - avg)/std

def normalization(inputList):
    Lmax = max(inputList)
    Lmin = min(inputList)
    outputList = []
    for num in inputList:
        outputList.append(round((num - Lmin + 1e-10) / (Lmax - Lmin + 1e-10), 3))
    return outputList

def calScore(MeList, state='pos'):
    scoreList = []
    result = []
    for i in MeList:
        scoreList.append(i[1])
    avg = np.average(scoreList)
    std = np.std(scoreList)
    
#    if len(scoreList) > 0:
#        n_scoreList = normalization(scoreList)
    for i in MeList:
        if state == 'pos':
            result.append([i[0],calZscore(i[1], avg, std)])
        elif state == 'neg':
            result.append([i[0], 0 - calZscore(i[1], avg, std)])
    return result

def addSocre(scoreDict, scoreList):
    for i in scoreList:
        if i[0] in scoreDict:
            scoreDict[i[0]].append(i[1])
        else:
            scoreDict[i[0]] = [i[1]]
    return scoreDict


""" pre-processing """
def MeRank(path):
    soup = BeautifulSoup(open(path, encoding="utf-8"), "lxml") # Choose the "lxml" parser (requires additional installation) or the default "html.parser".
    """ remove authors because it may contains latex tag """
    if soup.find("div", class_="ltx_authors"):
        soup.find("div", class_="ltx_authors").decompose()
        
    """ sentence segmentation from all paragraph """
    allParagraph = soup.select("p,td")

    count = 0
    inlineLatex = []
    displayLatex = []
    p2s = []
    if allParagraph:
        for index, lines in enumerate(allParagraph):
            while lines.math and lines.math.get('alttext'):
                if lines.math.get('alttext')[0:13] == "\displaystyle":
                    newTag = soup.new_tag("displayLatex")
                else:
                    newTag = soup.new_tag("inlineLatex")
                newTag.string = '<latex>' + lines.math.get('alttext') + '<latex>'
                lines.math.replace_with(newTag)
            if lines.parent:
                if type(lines.parent.get('class')) == list:
                    if lines.name == "td" and 'ltx_equation' in lines.parent.get('class'):
                        line = sentence_segmentation(lines.text)
#                         print(lines,"\n")
                        for i in line:
                            p2s.append(i)
                        if lines.inlineLatex:
                            for tdInlineLatex in lines.find_all('inlineLatex'):
                                count += 1
                                inlineLatex.append(tdInlineLatex.text[7:-7])
    #                         print(tdInlineLatex.text, "\n")
                        if lines.displayLatex:
                            for tdDisplayLatex in lines.find_all('displayLatex'):
                                count += 1
                                displayLatex.append(tdDisplayLatex.text[7:-7])
    #                         print(tdDisplayLatex.text, "\n")
                if lines.name == "p":
                    line = sentence_segmentation(lines.text)
                    for i in line:
                        p2s.append(i)
                    if lines.inlineLatex:
                        for pInlineLatex in lines.find_all('inlineLatex'):
                            count += 1
                            inlineLatex.append(pInlineLatex.text[7:-7])
#                         print(pInlineLatex.text, "\n")
                    if lines.displayLatex:
                        for pDisplayLatex in lines.find_all('displayLatex'):
                            count += 1
                            displayLatex.append(pDisplayLatex.text[7:-7])
#                         print(pDisplayLatex.text, "\n")
    for index, i in enumerate(p2s):
        if index + 1 < len(p2s):
            if p2s[index + 1][:21] == "<latex>\displaystyle=" and p2s[index][:7] == "<latex>":
#                 print(p2s[index], " and " , p2s[index + 1])
                p2s[index] = p2s[index][:-7] + p2s[index+1][7:]
                p2s[index + 1] = ""
#                 print(p2s[index], " and " , p2s[index + 1])
    sentences = rm_empty(p2s) 
#     for s in sentences:
#         print(s, '\n')

    """ check no empty """
    inlineLatex = rm_empty(inlineLatex)
    displayLatex = rm_empty(displayLatex)


    allText =  rm_empty(sentences)
    for index, item in enumerate(displayLatex):
        if index + 1 < len(displayLatex):
            if displayLatex[index + 1][:14] == "\displaystyle=" and "\displaystyle=" not in displayLatex[index]:
                displayLatex[index] += displayLatex[index+1]
                displayLatex[index + 1] = ""
#                 print(displayLatex[index])

    """ combine display and inline Latex """

    allLatex = inlineLatex.copy()

    if displayLatex:
        allLatex.extend(displayLatex)
    
#     allLatex_no_dp = list(set(allLatex))
#     print(allLatex)
    word_token_list = latex2mi(allLatex)

    """ create a list without repeat element """ 
#    word_token_list_no_dp = list(set(word_token_list))
#     for i in word_token_list:print(i)
#     for i in word_token_list:print(i,'=' , Counter(word_token_list)[i])
#     for i in allLatex:print(i, len(allLatex),'\n')
#     for i in word_token_list_no_dp:
#         if len(i) > 0:
#             print(i)

#     freqMoList = freqMo(allText, displayLatex)
    freqMoList = []
    tempfreqMo = freqMo(allText)
    if tempfreqMo:
        for item in tempfreqMo:
            if item[0] != '':
                freqMoList.append(item)
        
    freqMeList = []
    [freqMeList.append(i) for i in freqMoList if i[0].find('=') != -1]
    
#     print(inlineMathML, sentences)
#     print(allText)
#     densMoList = densMo(allText, displayLatex)
    densMoList = []
    tempdensMo = densMo(allText)
    if tempdensMo:
        for item in tempdensMo:
            if item[0] != '':
                densMoList.append(item)
    densMeList = []
    [densMeList.append(i) for i in densMoList if i[0].find('=') != -1]
    
    
    degrMoList = []
    cmiList = []
    tempdegrMo, cmiList = degrMo(allLatex, word_token_list, "PMI")
    if tempdegrMo:
        for item in tempdegrMo:
            if item[0] != '':
                degrMoList.append(item)

#     print(tempdegrMo)
#     print(allLatex)

    lengMoList = []
    templengMo = lengMo(allLatex, word_token_list, "PMI")
    if templengMo:
        for item in templengMo:
            if item[0] != '':
                lengMoList.append(item)
    
    """ print the original(Before z-score and PCA) information of each feature """
#     print("\n\n\n"+"freq =====================================")
#     printMe(freqMoList, "ME")
#     print("\n\n\n"+"dens =====================================")
#     printMe(densMoList, "ME")
#     print("\n\n\n"+"degr =====================================")
#     printMe(degrMoList, "ME")
#     print("\n\n\n"+"leng =====================================")
#     printMe(lengMoList, "ME")

    freqScoreList = []
    densScoreList = []
    degrScoreList = []
    lengScoreList = []

    freqScoreList = calScore(freqMoList, 'neg')
    densScoreList = calScore(densMoList, 'neg')
    degrScoreList = calScore(degrMoList, 'pos')
    lengScoreList = calScore(lengMoList, 'pos')
    """ check list length """
    if len(freqMoList) != len(densMoList) or len(densMoList) != len(degrMoList) or len(degrMoList) != len(lengMoList):
        print("=======================================")
        print("There is/are some metric(s) goes wrong!")
        print("freqLen : " + str(len(freqScoreList)))
        print("densLen : " + str(len(densScoreList)))
        print("degrLen : " + str(len(degrScoreList)))
        print("lengLen : " + str(len(lengScoreList)))
        print("\nIt may be a problem caused by an error in the math tag on the html.")
        print("=======================================")
        return [], []
#     print(freqMoList)
#     print(freqScoreList,"\n", densScoreList,"\n", degrScoreList,"\n", lengScoreList,"\n")
    ScroeDict = {}
    ScroeDict = addSocre(ScroeDict, freqScoreList)
    ScroeDict = addSocre(ScroeDict, densScoreList)
    ScroeDict = addSocre(ScroeDict, degrScoreList)
    ScroeDict = addSocre(ScroeDict, lengScoreList)
    
#     for i in ScroeDict:
#         print(i, ScroeDict[i], "\n")
    
    """ Dimension Reduction by PCA (4 to 1) """
    scoreList_4d = []
    for item in ScroeDict:
        while len(ScroeDict[item]) < 4: # depend on how many features
            ScroeDict[item] += [0]
        scoreList_4d.append(ScroeDict[item])

    if len(scoreList_4d) > 0:
        scoreList_4d_np = np.array(scoreList_4d)
        pca = PCA(n_components = 1)
        pca = pca.fit(scoreList_4d_np)
        pca_result = pca.transform(scoreList_4d_np)
    
    for index, item in enumerate(ScroeDict):
        ScroeDict[item] = pca_result[index].tolist()
        
#     for i in scoreList_4d:
#         print(len(i),i,"\n")        
#     print(ScroeDict)

    MeRankList = sorted(ScroeDict.items(), key=lambda k : k[1], reverse=True)
    
    # freqScoreList, "--------------", densScoreList, "---------", degrScoreList, "--------", lengScoreList
    
    printMe(MeRankList)
#     print('=============')
#     printMe(freqScoreList)
#     print('-------------')
#     printMe(densScoreList)
#     print('-------------')
#     printMe(degrScoreList)
#     print('-------------')
#     printMe(lengScoreList)
    
    return MeRankList, cmiList


resultList = []
cmiListAll = []
invalidDoc = []

def rankFunction(item):
    tempMeList, tempcmiList = MeRank(item)
    if len(tempMeList) > 0 and len(tempcmiList) > 0:
        resultList.append(tempMeList)
        cmiListAll.append(tempcmiList)
    elif len(tempMeList) > 0:
        resultList.append(tempMeList)
    else:
        invalidDoc.append(item)

datasetDir = r"..\dataset\html"
datasetPathList = []
for dirPath, dirNames, fileNames in os.walk(datasetDir):
    for f in fileNames:
        datasetPathList.append(os.path.join(dirPath, f))
        

for item in tqdm(datasetPathList):
    print("now loading...", "doc", item)
    rankFunction(item)


cmiCount = 0
for item in cmiListAll:
    cmiCount+=len(item)
# print(cmiListAll, cmiCount)

resultDict = {}
count = 0
for index, i in enumerate(resultList):
    count += 1
    arxvID = datasetPathList[index][15:-5]
    resultDict[arxvID] = i
#     print(datasetPathList[index])
#     print(printMe(i, "MO"), '\n===============================\n')
print("total: ", count, " documents.")


with open("../result/MeRank.json", "w", encoding='utf-8') as f:
    json.dump(resultDict, f, ensure_ascii=False)
# with open("../result/invalidDoc.txt", "w", encoding='utf-8') as iDf:
#    for item in invalidDoc:
#        iDf.write(item)
#        iDf.write('\n')

