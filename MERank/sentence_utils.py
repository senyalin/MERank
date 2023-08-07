import re
def sentence_segmentation(content = None):
    """ Segment an arbitary contents into collections of sentences by regex. """
    sentences = []
    if content:
        lines = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s)', content)
        for line in lines:
            sentences.append(line)
        return sentences
    else:
        return sentences
def rm_empty(List):
    """ Remove empty lines and \n. """
    result = []
    for line in List:
        line = line.replace("\n", "")
        line = line.replace("%","")
        if line != "" and line != " " and line != '' and line != ' ': 
            result.append(line)
    return result


def latex2mi(content=None):
    """ Convert LaTeX code into mathematical identifiers. """
    MoBox = []
    for item in content:
        MoBox.append(item)
    identifierSet = {}
    for i in MoBox:
        identifierSet[i] = re.findall(r'(?<![\\\w])[a-zA-Z]+|\\Gamma|\\Delta|\\Theta|\\Lambda|\\beta|\\gamma|\\delta|\\epsilon|\\varepsilon|\\zeta|\\eta|\\theta|\\vartheta|\\iota|\\kappa|\\lambda|\\mu|\\nu|\\xi|\\pi|\\varpi|\\rho|\\varrho|\\sigma|\\varsigma|\\tau|\\upsilon|\\phi|\\varphi|\\chi|\\psi|\\omega', i)

    temp = []
    for i in identifierSet:
        temp.extend(identifierSet[i])

    temp = list(filter(None, temp))
    word_token_list = []
    for i in temp:
        if i.isdecimal() == False and not 'arg' in i:
            word_token_list.append(i)
    return word_token_list

def printMe(scoreList, style = "ME"):
    """ Print out the mathematical equation (or mathematical object). """
    style = style.upper()
    rank = 1
    if type(scoreList) == list :
        for i in scoreList:
            if style == "ME":
                if i[0].find('=') != -1:
                    print('rank',rank, ' : ', i[0], '=', i[1])
    #             print('rank',rank, ' : ')
    #             print(i[0], i[1])
                    rank += 1
            elif style == "MO":
                print('rank',rank, ' : ', i[0], '=', i[1])
    #             print('rank',rank, ' : ')
    #             print(i[0], i[1])
                rank += 1
    elif type(scoreList) == dict:
        for i in scoreList:
            if i.find('=') != -1:
                print(i, '=', scoreList[i])