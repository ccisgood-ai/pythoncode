import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import re
import warnings
import seaborn as sns

# 读取评论数据集
df = pd.read_csv('20191214u.csv')
# jieba分词
df['content'] = df['content'].map(str)
df['cut_content'] = df['content'].map(lambda x: ' '.join(jieba.cut(x))) #以空格分隔jieba分词结果
# df['label']=df['label'].astype(str)

# 去除停用词
def stop_words():
    with open('stop_words.txt',encoding='UTF-8') as f:
        lines = f.readlines()
        result = [i.strip('\n') for i in lines]
    return result
stopwords=stop_words()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.75, min_df=2, stop_words=stopwords,token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')
features = tfidf.fit_transform(df.cut_content).toarray()
labels = df.label

from io import StringIO
label_df = df[['label']].drop_duplicates().sort_values('label')


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X = df['cut_content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df.groupby('model_name').accuracy.mean())


from sklearn.model_selection import train_test_split
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.feature_selection import chi2
N = 2
for label in sorted(label_df['label'].values):
    label=label-1
    indices = np.argsort(model.coef_[label])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(label))
    print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
    print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

# from sklearn.metrics import confusion_matrix
# conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(8,6))
# sns.heatmap(conf_mat, annot=True, fmt='d',
#             xticklabels=label_df.label.values, yticklabels=label_df.label.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# print(plt.show())
# df['label']=df['label'].astype(str)
# # print(type(df['label'][0]))
# from sklearn import metrics
# print(metrics.classification_report(y_test, y_pred, target_names=df['label'].unique()))

# rs=[]
# test_df = pd.read_csv('other1u.csv')
# texts=test_df['content'].values.tolist()
# text_features = tfidf.transform(texts)
# predictions = model.predict(text_features)
# for predicted in predictions:
#   rs.append(predicted)
# test_df['predict']=rs
# test_df.to_csv('my_csv.csv', mode='a', header=True)