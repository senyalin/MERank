""" Obtain the dataset (arXiv to HTML) using ar5iv. """

import urllib.request

""" Direct download by arxiv id. """ 
# urllib.request.urlretrieve("https://ar5iv.org/abs/1512.03385", "../dataset/html/1512.03385.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/1605.02019", "../dataset/html/1605.02019.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/1909.08041", "../dataset/html/1909.08041.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2012.07436", "../dataset/html/2012.07436.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.06560", "./dataset/html/2205.06560.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.05625", "../dataset/html/2205.05625.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.05476", "../dataset/html/2205.05476.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.05272", "../dataset/html/2205.05272.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.05270", "../dataset/html/2205.05270.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.05069", "../dataset/html/2205.05069.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04980", "../dataset/html/2205.04980.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04892", "../dataset/html/2205.04892.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04885", "../dataset/html/2205.04885.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04725", "../dataset/html/2205.04725.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04547", "../dataset/html/2205.04547.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04423", "../dataset/html/2205.04423.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04382", "../dataset/html/2205.04382.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04411", "../dataset/html/2205.04411.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04449", "../dataset/html/2205.04449.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.04061", "../dataset/html/2205.04061.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.03819", "../dataset/html/2205.03819.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.03817", "../dataset/html/2205.03817.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.03773", "../dataset/html/2205.03773.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.03656", "../dataset/html/2205.03656.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.03375", "../dataset/html/2205.03375.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.03273", "../dataset/html/2205.03273.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.03219", "../dataset/html/2205.03219.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.02998", "../dataset/html/2205.02998.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2112.04153", "../dataset/html/2112.04153.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2202.03609", "../dataset/html/2202.03609.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2205.14969", "../dataset/html/2205.14969.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2206.14597", "../dataset/html/2206.14597.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2202.13341", "../dataset/html/2202.13341.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2207.04881", "../dataset/html/2207.04881.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2109.04344", "../dataset/html/2109.04344.html")
# urllib.request.urlretrieve("https://ar5iv.org/abs/2206.13687", "../dataset/html/2206.13687.html")

""" Download using NTCIR-12 list. """
# path = '../dataset/urlList/NTCIR2_URL_list.txt' # output path
# arXivIdList = []
# with open(path, 'r') as AIL:
#     for line in AIL.readlines():
#         if line[:4] == "http" and line[21].isdigit():
#             arXivIdList.append(line[:-1])

# from tqdm import tqdm
# count= 0
# for item in tqdm(arXivIdList):
#     count+=1
#     if count >= 0:
#         urllib.request.urlretrieve(item[:9] + "5" + item[10:], "./dataset/ntcirAll/" + item[21:] + ".html")

