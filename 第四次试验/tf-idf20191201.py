import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取评论数据集
df = pd.read_csv('test1-20191201-u.csv')

# jieba分词
df['content'] = df['content'].map(str)
df['cut'] = df['content'].map(lambda x: ' '.join(jieba.cut(x))) #以空格分隔jieba分词结果
# print(df.loc[3,'cut'])

def stop_words():
    with open('哈工大停用词表.txt') as f:
        lines = f.readlines()
        result = [i.strip('\n') for i in lines]
    return result

stopwords = stopwords_list()