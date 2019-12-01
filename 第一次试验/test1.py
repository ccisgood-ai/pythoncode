import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt

# 统计词频
def wordcount(text):
    # 文章字符串前期处理
    strl_ist = jieba.lcut(text)
    count_dict = {}
    all_num = 0;
    # 如果字典里有该单词则加1，否则添加入字典
    for str in strl_ist:
        if(len(str) <= 1):
            continue
        else:
            all_num+=1
        if str in count_dict.keys():
            count_dict[str] = count_dict[str] + 1
        else:
            count_dict[str] = 1
    #按照词频从高到低排列
    count_list=sorted(count_dict.items(),key=lambda x:x[1],reverse=True)
    return count_list, all_num


# dataframe格式
data = pd.read_csv('test0.csv',encoding='gbk')

# 分析 帖子内容 文本中的词频
content = data['帖子内容'].dropna(how = 'any').values
content_text = ""
for c in content:
    content_text += c
content_analyze = jieba.analyse.extract_tags(content_text, topK=50, withWeight=False, allowPOS=())
#question_lists = " ".join(question_analyze).split(' ')
#print(question_lists)
#print(content_text)