from sentence_utils import sentence_segmentation
from sentence_utils import rm_empty
from sentence_utils import printMe
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import json

def combine_content_gold_ver(content = None):
    if content:
        tempPos = []
        displayLatex = []

        for index, i in enumerate(content):
            if "\displaystyle=" in i:
                tempPos.append(index)
            index += 1

        needCombine = False
        tempString = ""

        for index, i in enumerate(content):
            needCombine = False
            if index + 1 in tempPos:
                needCombine = True
            if len(tempString) > 2:
                if tempString[0:6] == '<latex>' and tempString[-7:-1] == '<latex>':
                    tempString == tempString[1:len(tempString)-1]
            if i[0:6] == '<latex>' and i[-7:-1] == '<latex>':
                tempString += i[7:-7]
            else:
                tempString += i
            if not needCombine:
                displayLatex.append(tempString)
                tempString = ""
        return displayLatex
    else:
        return None

def scoreJudge(scoreDict): 
    """ 
    make sure scoreDict is distributed
    return False if no distribution, else return True
    """
    totalMe = 0
    zeroCount  = 0
    result = True
    for item in scoreDict:
        totalMe += scoreDict[item]
    for item in scoreDict:
        if scoreDict[item] > totalMe * 0.7:
            result = False
        if scoreDict[item] == 0:
            zeroCount += 1
    if zeroCount >= len(scoreDict) / 2 or len(scoreDict) <= 2 or totalMe < 5:
        result = False
    if 0 in scoreDict:
        if scoreDict[0] > totalMe * 0.5:
            result = False;
    return result

def GoldStandardRank(path):
    soup = BeautifulSoup(open(path, encoding="utf-8"), "lxml") # Choose the "lxml" parser (requires additional installation) or the default "html.parser".
    """ remove authors because it may contains latex tag """
    if soup.find("div", class_="ltx_authors"): 
        soup.find("div", class_="ltx_authors").decompose()

    allTitle = soup.select("h2.ltx_title")
#    allContent = soup.select("h2.ltx_title, p, math.ltx_Math[display='block'], math.ltx_Math[alttext*='displaystyle']")
    allParagraph = soup.select("h2.ltx_title,p,td")
    allcite = soup.select('cite')

    cites = []
    for i in allcite:
        cites.append(i.text)

    titles = []
    for i in allTitle:
        titles.append(i.text)
    
    count = 0    
    displayLatex = []
    inlineLatex = []
    p2s = []
    sentences = []

    if allParagraph:
        for lines in allParagraph:
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
            if lines.name == 'h2':
                p2s.append(lines.text)
    for index, i in enumerate(p2s):
        if index + 1 < len(p2s):
            if p2s[index + 1][:21] == "<latex>\displaystyle=" and p2s[index][:7] == "<latex>":
#                 print(p2s[index], " and " , p2s[index + 1])
                p2s[index] = p2s[index][:-7] + p2s[index+1][7:]
                p2s[index + 1] = ""
#                 print(p2s[index], " and " , p2s[index + 1])
    titles = rm_empty(titles)
    sentences = rm_empty(p2s)

    """ check no empty """
    inlineLatex = rm_empty(inlineLatex)
    displayLatex = rm_empty(displayLatex)

    for index, item in enumerate(displayLatex):
        if index + 1 < len(displayLatex):
            if displayLatex[index + 1][:14] == "\displaystyle=" and "\displaystyle=" not in displayLatex[index]:
                displayLatex[index] += displayLatex[index+1]
                displayLatex[index + 1] = ""

    """ remove empty lines and \n again """
    allText = sentences

    """ combine display and inline Latex """
    allLatex = []
    allLatex = inlineLatex.copy()
    
    if displayLatex:
        allLatex.extend(displayLatex)

    score1 = ['introduction', 'background', 'preliminar', 'related', 'motivati']
    score2 = ['experiment', 'evaluation', 'learn', 'analysis', 'result','appendix']
    score3 = ['method', 'approach', 'model', 'train', 'framework', 'proposed', 'formula']
#     score1 = ['introduction', 'related', 'background']
#     score2 = ['experiment', 'result', 'appendix']
#     score3 = ['method', 'approach', 'model']

    titleScore = {}
    for t in titles:
        for s1 in score1:
            if s1.lower() in t.lower():
                titleScore[t] = 1
        for s2 in score2:
            if s2.lower() in t.lower():
                titleScore[t] = 2
        for s3 in score3:
            if s3.lower() in t.lower():
                titleScore[t] = 3
        if not t in titleScore:
            titleScore[t] = 0

    allLatexSocre = {}
    titleCounter = 0
    nowScore = 0
    for text in allText:
        if titleCounter < len(titles) and text == titles[titleCounter]: 
            nowScore = titleScore[text]
            if len(titles) > titleCounter + 1:
                titleCounter += 1
        for latex in allLatex:
            if '<latex>'+latex+'<latex>' in text : 
                allLatexSocre[latex] = nowScore
    allLatexSocre = sorted(allLatexSocre.items(), key=lambda k : k[1], reverse=True)
    return allLatexSocre


datasetDir = r"..\dataset\html"
datasetPathList = []
for dirPath, dirNames, fileNames in os.walk(datasetDir):
    for f in fileNames:
        datasetPathList.append(os.path.join(dirPath, f))
        
resultList = []
for item in tqdm(datasetPathList):
# for index, item in enumerate(datasetPathList):
#     print("now loading...", "doc", "[", index+1, "]", item)
    resultList.append(GoldStandardRank(item))
    
resultDict = {}
printCount = 0
docFilter = []
for index, i in enumerate(resultList):
    arxivID = datasetPathList[index][15:-5]
    resultDict[arxivID] = i
#     MeScoreDict = {}
#     for Me, score in i: # Print out the distribution of ME count.
#         if Me.find("=") != -1: 
#             if score not in MeScoreDict:
#                 MeScoreDict[score] = 1
#             else:
#                 MeScoreDict[score] += 1
#     if scoreJudge(MeScoreDict):
    printCount += 1
#         docFilter.append(arxivID)
#         print(MeScoreDict, '\n')
    print(datasetPathList[index])
    print(printMe(i), '\n===============================\n')
print("total: ", printCount, " documents.")

with open("../result/GoldStandard.json", "w", encoding='utf-8') as f:
    json.dump(resultDict, f, ensure_ascii=False)

# with open("../result/doc.txt", "w", encoding='utf-8') as doc:
#     for i in docFilter:
#         doc.write(i+'\n')

