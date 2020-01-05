import csv as csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import re
import warnings
import os


# 读取评论数据集
test_df = pd.read_csv('other1u.csv')

tests=test_df['content'].values.tolist()
print(type(tests))