import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import re
import warnings

data=[]
with open('test2-u.csv',encoding = 'utf-8') as text: # 按行读取csv，并存入data数组，得到的形式是list of list
    row = csv.reader(text)
    for r in row:
        data.append(r)

pattern = r'[\。]'  # 涉及到一些特殊字符，就不用 ? ! 来分句子了  r'[\。|\？|\?|！|!]'
rs = []#新建一个列表
for i in range(len(data)):
    result_list = re.split(pattern, data[i][0])
    for r in result_list:
        rs.append(r)
rs = [i for i in rs if i != ''] # 去除空行

rs2=[]
for i in range(len(rs)):
    a=[x.strip() for x in rs[i] if x.strip() != '']
    b=''.join(a)
    rs2.append(b)

with open('rs20191130.csv','w',newline='',encoding = 'utf-8') as datacsv:
    csvwriter = csv.writer(datacsv,dialect=('excel'))  # https://www.cnblogs.com/unnameable/p/7366437.html
    for i in range(len(rs2)):
        csvwriter.writerow([rs2[i]])

print(0)




