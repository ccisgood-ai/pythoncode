import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import re
import warnings

with open('test2-u.csv',encoding = 'utf-8') as text: # 按行读取csv，并存入data数组，得到的形式是list of list
    reader = csv.reader(text,delimiter = ',')
    data = [row for row in reader]
text.close()

#  将用户id和发言内容对应存入数组中
userid=[]
content=[]
for i in range(len(data)):
    userid.append(data[i][0])
    content.append(data[i][1])

#对content列表中的内容进行预处理
content_rs=[] #暂存处理好的句子，元素为列表，一个用户对应一个列表
pattern = r'[\。]'  # 涉及到大学计算机课程中有一些特殊字符，就不用 ? ! 来分句子了  r'[\。|\？|\?|！|!]'
for i in range(len(content)):
    rs1 = ''.join([x.strip() for x in content[i] if x.strip() != ''])  # 去除头尾的换行符
    rs2 = [i for i in re.split(pattern, rs1) if i != '']  # 去除分句之后的空值
    content_rs.append(rs2)

# 将用户id和发布内容对应存入字典，用户id为key，内容为values
# 实现字典的一键多值情况，采用{key：[list]}形式
# 一位同学可能有多次回复，共计94个匿名用户，一个匿名用户算一位用户
dic={}
for i in range(len(userid)):
    for j in range(len(content_rs[i])):
        dic.setdefault(userid[i], []).append(content_rs[i][j])  # content_rs[i]为list形式，表示某一次贴子中发言句子的列表
    # print(dic['5434992'])  查看某位同学的发言

#创造list，存入[用户id，发言单句]
item=list(dic.items())
rs=[]
for i in range(len(item)):
    c=item[i]
    for j in range(len(c[1])):
        tmp=[]
        tmp.append(c[0])
        tmp.append(c[1][j])
        rs.append(tmp)

with open('rs20191130.csv','w',newline='',encoding = 'utf-8') as datacsv:
    csvwriter = csv.writer(datacsv,dialect=('excel'))  # https://www.cnblogs.com/unnameable/p/7366437.html
    # csvwriter.writerow(['userid', 'content'])
    csvwriter.writerows(rs)
datacsv.close()

print(0)





