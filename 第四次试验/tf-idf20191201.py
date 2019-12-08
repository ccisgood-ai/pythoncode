import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import re
import warnings


# 读取评论数据集
df = pd.read_csv('test1-20191208-ver2-u.csv')
# df.groupby('label')['content'].count().plot.bar(ylim=0)
# print(plt.show())

# jieba分词
df['content'] = df['content'].map(str)
df['cut'] = df['content'].map(lambda x: ' '.join(jieba.cut(x))) #以空格分隔jieba分词结果
# print(df.loc[3,'cut'])


# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X = df['cut']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 变换器
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
# 词表数量
print(len(vect.vocabulary_))

# 交叉验证评估模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(),vect.transform(X_train), y_train, cv=5)
print('平均交叉验证准确率：{:.3f}'.format(np.mean(scores)))

# 去除停用词
def stop_words():
    with open('stop_words.txt',encoding='UTF-8') as f:
        lines = f.readlines()
        result = [i.strip('\n') for i in lines]
    return result
stopwords=stop_words()

# CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。
# 对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_df=0.8, min_df=3, stop_words=stopwords,token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')  #正则啥意思
vect.fit(X_train)
words_matrix = pd.DataFrame(vect.transform(X_train).toarray(),columns=vect.get_feature_names())
# 训练模型
lr = LogisticRegression()
lr.fit(vect.transform(X_train), y_train)
# 词表数量
print(len(vect.vocabulary_))
print('测试集准确率：{:.3f}'.format(lr.score(vect.transform(X_test), y_test)))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=3), LogisticRegression())
pipe.fit(X_train, y_train)
scores = cross_val_score(pipe, X_train, y_train, cv=5)
print('平均交叉验证准确率：{:.3f}'.format(np.mean(scores)))

vectorizer = pipe.named_steps['tfidfvectorizer']
# 找到每个特征中最大值
max_value = vectorizer.transform(X_train).max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# 获取特征名称
feature_names = np.array(vectorizer.get_feature_names())
print("tfidf较低的特征：\n{}".format(feature_names[sorted_by_tfidf[:20]]))
print()
print("tfidf较高的特征：\n{}".format( feature_names[sorted_by_tfidf[-20:]]))


from sklearn import metrics
# 预测值
y_pred = pipe.predict(X_test)
print('测试集准确率：{:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))
print('测试集准确率：{:.3f}'.format(pipe.score(X_test, y_test)))
print(metrics.confusion_matrix(y_test, y_pred))