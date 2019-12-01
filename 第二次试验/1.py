import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import re
import warnings

data=[]
with open('rs.csv',encoding = 'utf-8') as text: # 按行读取csv，并存入data数组，得到的形式是list of list
    row = csv.reader(text)
    for r in row:
        data.append(r)

t=[]
t.append(data[1800][0])
t.append(data[1801][0])
print(t)
print(t[0].strip())
# t=[x.strip() for x in t[0] if x.strip() != '']
rs2=[]
for i in range(len(t)):
    a=[x.strip() for x in t[i] if x.strip() != '']
    b=''.join(a)
    rs2.append(b)

print(rs2)