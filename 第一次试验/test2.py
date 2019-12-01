import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt

file_ob = open('test2-utf.csv',encoding='UTF-8').read().split('\n')#逐行进行读取
rs = []#新建一个列表

for i in range(len(file_ob)):
    result=[]
    seg_li=jieba.cut(file_ob[i])
    for w in seg_li:
        result.append(w)
    rs.append(result)

file=open('rs2.txt','w',encoding="utf-8") #保存为txt文件
file.write(str(rs))
file.close()

txt = open("rs2.txt",encoding='utf-8').read() # 对txt文件进行读取

stopwords = [line.strip() for line in open('stop_words.txt',encoding='utf-8').readlines()] # 停用词文件
words = jieba.lcut(txt) #进行分词

counts={}  # 新建字典，用于存放各word出现的次数
for word in words:  # 判断是否在停用词内，去除冗余字符
    if word not in stopwords:
        if len(word) == 1:
            continue
        else:
            counts[word] = counts.get(word,0)+1
            # counts.get(word,0)：若word存在，则返回counts[word]；若不存在，则返回 0；

items = list(counts.items()) # 转换列表；counts.items()返回可遍历的(键, 值) 元组数组。
items.sort(key=lambda x:x[1], reverse=True) # 进行遍历访问，降序输出(https://www.cnblogs.com/zle1992/p/6271105.html)
for i in range(3000):
    word,count = items[i]

print(type(word))

    # # https://www.cnblogs.com/star12111/p/8848594.html
    # print('{:<10}{:>7}'.format(word,count))  # 打印最终结果

with open('rs2.csv','w',newline='') as datacsv:
    csvwriter = csv.writer(datacsv,dialect=('excel'))  # https://www.cnblogs.com/unnameable/p/7366437.html
    for i in items:
        csvwriter.writerow(i)

print('0')






