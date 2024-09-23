#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm


# In[2]:


train=pd.read_csv('train.csv')
train.to_csv('train_woheader.csv',header=False,index=False)
G=nx.read_edgelist('train_woheader.csv',delimiter=',',create_using=nx.Graph(),nodetype=int)


# In[3]:


predict=pd.read_csv('test.csv')


# In[4]:


len(G.nodes), len(G.edges)


# In[5]:


# resolution=1.1
# coms = algorithms.louvain(G, weight='weight', resolution=resolution, randomize=False)


# In[6]:


# partition = community_louvain.best_partition(G)


# In[7]:


# partition={}
# for i,c in enumerate(coms.communities):
#     for n in c:
#         partition[n]=i
        

# # partition        


# In[8]:


# Id=[i for i in range(len(predict))]

# ans=[0 for i in range(len(predict))]



# for nodepair in predict.values.tolist():
#     if partition[nodepair[1]]==partition[nodepair[2]]:
#         ans[nodepair[0]]=1


# In[9]:


# answer=pd.DataFrame({'Id':Id,'Category':ans})


# In[10]:


# answer.to_csv(f'louvain_{resolution}.csv',index=False)


# # 設定參數resolution/threshold/seed

# In[11]:


resolution=[1]
threshold=[1e-4]
seed=[7]


# # 基於modularity 選出partition

# In[12]:


for r,t,s in zip(resolution,threshold,seed):
    coms = nx_comm.louvain_communities(G, weight='weight', resolution=r,threshold=t,seed=s)
    partition={}
    for i,c in enumerate(coms):
        for n in c:
            partition[n]=i

    Id=[i for i in range(len(predict))]

    ans=[0 for i in range(len(predict))]



    for nodepair in predict.values:
#         print(nodepair)
        if partition[nodepair[1]]==partition[nodepair[2]]:
            ans[nodepair[0]]=1

    answer=pd.DataFrame({'Id':Id,'Category':ans})
    answer.to_csv(f'louvain_resolution_{r}_threshold_{t}_seed_{s}.csv',index=False)


# # 測試取不同level 之情況

# In[14]:

'''
for r,t,s in zip(resolution,threshold,seed):
    dendo=nx_comm.louvain_partitions(G, weight='weight', resolution=r,threshold=t,seed=s)
    #會return list of sets(top level(很多community)->bottom level(比較少community))
    for level,coms in enumerate(dendo) : 
        partition={}
        for i,c in enumerate(coms):
            for n in c:
                partition[n]=i

        Id=[i for i in range(len(predict))]
        ans=[0 for i in range(len(predict))]

        for nodepair in predict.values:
            if partition[nodepair[1]]==partition[nodepair[2]]:
                ans[nodepair[0]]=1
        answer=pd.DataFrame({'Id':Id,'Category':ans})
        answer.to_csv(f'louvain_level_{level}_resolution_{r}_threshold_{t}_seed_{s}.csv',index=False)
'''
