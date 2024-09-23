#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


train=pd.read_csv('data_train_edge.csv')
predict=pd.read_csv('predict.csv')
# answer=pd.read_csv('answer_RandomForest.csv')
# answer
ans500_ground_truth=pd.read_csv('ans500_ground_truth.csv')
# answer1=pd.read_csv('answer_GBDT.csv')


# In[11]:


#建立train data 答案之字典
edge={}
for n1,n2 in zip(train['node1'], train['node2']):
    edge[(n1,n2)]=1
#紀錄已有答案之test data 之index
t=[]
for i,n1,n2 in zip(range(len(predict)),predict['node1'], predict['node2']):
    if edge.get((n1,n2))==1:
        t+=[i]
        

#建立train data 答案+predict 中連到自己者之字典        
edge1={}
for n1,n2 in zip(predict['node1'], predict['node2']):
    if n1==n2:
        edge1[(n1,n2)]=1
edge1.update(edge)

#紀錄已有答案之test data +predict 中連到自己者之index
t1=[]
for i,n1,n2 in zip(range(len(predict)),predict['node1'], predict['node2']):
    if edge1.get((n1,n2))==1:
        t1+=[i]

# print(len(t))

# answer=pd.read_csv('answer_RandomForest_equal.csv')
# answer['ans'][t1]=1

# for i,ans in enumerate(ans500_ground_truth['ans']):
#     answer['ans'][i]=ans

# answer.to_csv('answer_RandomForest_equal_revised.csv',index=False)
# answer






# answer1=pd.read_csv('answer_GBDT_equal.csv')
# answer1['ans'][t1]=1

# for i,ans in enumerate(ans500_ground_truth['ans']):
#     answer1['ans'][i]=ans

# answer1.to_csv('answer_GBDT_equal_revised.csv',index=False)
# answer1



answer2=pd.read_csv('answer_RandomForest_gt.csv')
answer2['ans'][t1]=1

for i,ans in enumerate(ans500_ground_truth['ans']):
    answer2['ans'][i]=ans

answer2.to_csv('answer_RandomForest_gt_revised.csv',index=False)
answer2






answer3=pd.read_csv('answer_GBDT_gt.csv')
answer3['ans'][t1]=1

for i,ans in enumerate(ans500_ground_truth['ans']):
    answer3['ans'][i]=ans

answer3.to_csv('answer_GBDT_gt_revised.csv',index=False)
answer3


# In[23]:


# # 先random 設定值,再將答案填上去
# random.seed(1)
# random_answer=[random.randint(0, 1) for i in range(len(predict))]
# random_answer=np.array(random_answer)
# random_answer[t1]=1

# for i,ans in enumerate(ans500_ground_truth['ans']):
#     random_answer[i]=ans
    

# random_answer=pd.DataFrame({'predict_nodepair_id':list(range(10200)),'ans':random_answer})
# random_answer.to_csv('random_answer_revised.csv',index=False)
# random_answer


# In[24]:


# #先把所有直設成0,再將答案填上去
# initial_0=[0 for i in range(len(predict))]
# initial_0=np.array(initial_0)
# initial_0[t1]=1

# for i,ans in enumerate(ans500_ground_truth['ans']):
#     initial_0[i]=ans
    
    
# initial_0=pd.DataFrame({'predict_nodepair_id':list(range(10200)),'ans':initial_0})
# initial_0.to_csv('initial_0_revised.csv',index=False)
# initial_0

