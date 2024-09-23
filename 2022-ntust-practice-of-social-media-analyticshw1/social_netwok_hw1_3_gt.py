#!/usr/bin/env python
# coding: utf-8

# # 參考https://github.com/GopiSumanth/Facebook-Link-Prediction 進行修改

# # 結合各種metric 當成feature 進行預測(把 500 筆test data with ground true 加進train data,使用test data 所計算出的feature)

# In[1]:


#Importing Libraries
# please do go through this python notebook: 
import warnings
warnings.filterwarnings("ignore")

import csv
import pandas as pd#pandas to create small dataframes 
import datetime #Convert to unix time
import time #Convert to unix time
# if numpy is not installed already : pip3 install numpy
import numpy as np#Do aritmetic operations on arrays
# matplotlib: used to plot graphs
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns#Plots
from matplotlib import rcParams#Size of plots  
from sklearn.cluster import MiniBatchKMeans, KMeans#Clustering
from  sklearn.ensemble import RandomForestClassifier
import math
import pickle
import os
import os.path
import shutil


# to install xgboost: pip3 install xgboost
import xgboost as xgb

import warnings
import networkx as nx
import pdb
import pickle


# In[2]:


#convert data to facebook data format

#移除data 資料夾
if os.path.isdir('./data'):
    shutil.rmtree('./data')
# 建立資料夾
os.makedirs('./data/after_eda', exist_ok=True)
os.makedirs('./data/fea_sample', exist_ok=True)

data_train_edge=pd.read_csv('data_train_edge.csv')
predict=pd.read_csv('predict.csv')
test_data_gt=pd.read_csv('ans500_ground_truth.csv')
index=test_data_gt[test_data_gt['ans']==1].index
data_train_edge=pd.concat([data_train_edge,predict.iloc[index]]).drop_duplicates().reset_index(drop=True)
data_train_edge.rename(columns = {'node1':'source_node', 'node2':'destination_node'}, inplace = True)
predict.rename(columns = {'node1':'source_node', 'node2':'destination_node'}, inplace = True)
predict_500=predict[:500]
predict_500['indicator_link']=test_data_gt['ans']
data_train_edge.to_csv('./data/train.csv',index=False)
# predict.to_csv('./data/predict.csv',index=False)


# # 檢查predict 中是否有不在train data 中的node

# In[3]:


# for i in pd.concat([predict['source_node'], predict['destination_node']], ignore_index=True).unique():
#     if i not in pd.concat([predict['source_node'], predict['destination_node']], ignore_index=True).unique():
#         print("yes")
#         break


# # 檢查是否有連到自己的node

# In[4]:


# flag1='No'
# self1=[]
# for n1,n2 in zip(data_train_edge['source_node'], data_train_edge['destination_node']):
#     if n1==n2:
#         flag1='Yes'
#         self1+=[n1]
        
# print(flag1)
# print(len(self1))

# self2=[]
# flag2='No'
# for n1,n2 in zip(predict['source_node'], predict['destination_node']):
#     if n1==n2:
#         flag2='Yes'
#         self2+=[n2]
# print(flag2)
# print(self2)


# # 檢查是否要預測已經有的邊

# In[5]:


# edge={}
# for n1,n2 in zip(data_train_edge['source_node'], data_train_edge['destination_node']):
#     edge[(n1,n2)]=1

# t=[]
# for n1,n2 in zip(predict['source_node'], predict['destination_node']):
#     if edge.get((n1,n2))==1:
#         t+=[(n1,n2)]
# print(len(t))


# In[6]:


#reading graph
if not os.path.isfile('data/after_eda/train_woheader.csv'):
    traincsv = pd.read_csv('data/train.csv')
    print(traincsv[traincsv.isna().any(1)])
    print(traincsv.info())
    print("Number of diplicate entries: ",sum(traincsv.duplicated()))
    traincsv.to_csv('data/after_eda/train_woheader.csv',header=False,index=False)
    print("saved the graph into file")
    g=nx.read_edgelist('data/after_eda/train_woheader.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
else:
    g=nx.read_edgelist('data/after_eda/train_woheader.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)


# > Displaying a sub graph

# In[7]:


if not os.path.isfile('train_woheader_sample.csv'):
    pd.read_csv('data/train.csv', nrows=50).to_csv('train_woheader_sample.csv',header=False,index=False)
    
subgraph=nx.read_edgelist('train_woheader_sample.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
# https://stackoverflow.com/questions/9402255/drawing-a-huge-graph-with-networkx-and-matplotlib

pos=nx.spring_layout(subgraph)
nx.draw(subgraph,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
plt.savefig("graph_sample.pdf")
print(nx.info(subgraph))


# # 1. Exploratory Data Analysis

# In[8]:


# No of Unique persons 
print("The number of unique persons",len(g.nodes()))


# ## 1.1 No of followers for each person

# In[9]:


# print(dict(g.in_degree()))
indegree_dist = list(dict(g.in_degree()).items())
indegree_dist.sort()
indegree_dist
no_of_node=[k for k,v in indegree_dist]
indegree_dist=[v for k,v in indegree_dist]
plt.figure(figsize=(10,6))
plt.plot(no_of_node,indegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()


# In[10]:


indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist[0:1500000])
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()


# In[11]:


plt.boxplot(indegree_dist)
plt.ylabel('No Of Followers')
plt.show()


# In[12]:


### 90-100 percentile
for i in range(0,11):
    print(90+i,'percentile value is',np.percentile(indegree_dist,90+i))


# 99% of data having followers of 40 only.

# In[13]:


### 99-100 percentile
for i in range(10,110,10):
    print(99+(i/100),'percentile value is',np.percentile(indegree_dist,99+(i/100)))


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.distplot(indegree_dist, color='#16A085')
plt.xlabel('PDF of Indegree')
sns.despine()
#plt.show()


# ## 1.2 No of people each person is following

# In[15]:


outdegree_dist = list(dict(g.out_degree()).items())
outdegree_dist.sort()
no_of_node=[k for k,v in outdegree_dist]
outdegree_dist=[v for k,v in outdegree_dist]
plt.figure(figsize=(10,6))
plt.plot(no_of_node,outdegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following')
plt.show()


# In[16]:


indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(outdegree_dist[0:1500000])
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following')
plt.show()


# In[17]:


plt.boxplot(outdegree_dist)
plt.ylabel('No Of people each person is following')
plt.show()


# In[18]:


### 90-100 percentile
for i in range(0,11):
    print(90+i,'percentile value is',np.percentile(outdegree_dist,90+i))


# In[19]:


### 99-100 percentile
for i in range(10,110,10):
    print(99+(i/100),'percentile value is',np.percentile(outdegree_dist,99+(i/100)))


# In[20]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.distplot(outdegree_dist, color='#16A085')
plt.xlabel('PDF of Outdegree')
sns.despine()


# In[21]:


print('No of persons those are not following anyone are' ,sum(np.array(outdegree_dist)==0),'and % is',
                                sum(np.array(outdegree_dist)==0)*100/len(outdegree_dist) )


# In[22]:


print('No of persons having zero followers are' ,sum(np.array(indegree_dist)==0),'and % is',
                                sum(np.array(indegree_dist)==0)*100/len(indegree_dist) )


# In[23]:


count=0
for i in g.nodes():
    if len(list(g.predecessors(i)))==0 :
        if len(list(g.successors(i)))==0:
            count+=1
print('No of persons those are not not following anyone and also not having any followers are',count)


# ## 1.3 both followers + following 

# In[24]:


from collections import Counter
dict_in = dict(g.in_degree())
# print(dict_in)
dict_out = dict(g.out_degree())
# print(dict_out)
d = Counter(dict_in) + Counter(dict_out)
in_out_degree = np.array(list(d.items()))
in_out_degree


# In[25]:


in_out_degree_sort = sorted(in_out_degree, key=lambda x:x[0]) 
in_out_degree_sort[0][0]

no_of_node=[in_out_degree_sort[i][0] for i in range(len(in_out_degree_sort))]
in_out_degree_sort=[in_out_degree_sort[i][1] for i in range(len(in_out_degree_sort))]
plt.figure(figsize=(10,6))
plt.plot(no_of_node,in_out_degree_sort)
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following + followers')
plt.show()


# In[26]:


### 90-100 percentile
for i in range(0,11):
    print(90+i,'percentile value is',np.percentile(in_out_degree_sort,90+i))


# In[27]:


### 99-100 percentile
for i in range(10,110,10):
    print(99+(i/100),'percentile value is',np.percentile(in_out_degree_sort,99+(i/100)))


# In[28]:


print('Min of no of followers + following is',min(in_out_degree_sort))
print(np.sum(in_out_degree_sort==min(in_out_degree_sort)),' persons having minimum no of followers + following')


# In[29]:


print('Max of no of followers + following is',max(in_out_degree_sort))
print(np.sum(in_out_degree_sort==max(in_out_degree_sort)),' persons having maximum no of followers + following')


# In[30]:


print('No of persons having followers + following less than 10 are',np.sum(np.array(in_out_degree_sort)<10))


# In[31]:


print('No of weakly connected components',len(list(nx.weakly_connected_components(g))))
count=0
for i in list(nx.weakly_connected_components(g)):
    if len(i)==2:
        count+=1
print('weakly connected components wit 2 nodes',count)


# # 2. Posing a problem as classification problem 

# ## 2.1 Generating some edges which are not present in graph for supervised learning  
# Generated Bad links from graph which are not in graph and whose shortest path is greater than 2. 

# In[32]:


# %%time
# ###generating bad edges from given graph
# import random
# random.seed(1)

# if not os.path.isfile('data/after_eda/missing_edges_final.p'):
#     #getting all set of edges
#     r = csv.reader(open('data/after_eda/train_woheader.csv','r'))
#     edges = dict()
#     for edge in r:
#         edges[(int(edge[0]), int(edge[1]))] = 1
        
        
        
#     missing_edges = set([])
    
    
#     for i,nodepair in predict[:500].iterrows():
#         tmp = edges.get((nodepair[0],nodepair[1]),-1)
#         if tmp == -1:
#             missing_edges.add((nodepair[0],nodepair[1]))
            
#     p=predict[500:].values.tolist()
#     while (len(missing_edges)<41020):
#         a=random.choice(list(g.nodes))
#         b=random.choice(list(g.nodes))
#         tmp = edges.get((a,b),-1)
#         if [n1,n2] in p:
#             continue
#         if tmp == -1 and a!=b:
#             try:
#                 if nx.shortest_path_length(g,source=a,target=b) > 2: 

#                     missing_edges.add((a,b))
#                 else:
#                     continue  
#             except:  
#                     missing_edges.add((a,b))              
#         else:
#             continue
#     pickle.dump(missing_edges,open('data/after_eda/missing_edges_final.p','wb'))
# else:
#     missing_edges = pickle.load(open('data/after_eda/missing_edges_final.p','rb'))
    


# # missing_edges


# In[33]:


get_ipython().run_cell_magic('time', '', "###generating bad edges from given graph\nimport random\nif not os.path.isfile('data/after_eda/missing_edges_final.p'):\n    #getting all set of edges\n    r = csv.reader(open('data/after_eda/train_woheader.csv','r'))\n    edges = dict()\n    for edge in r:\n        #print(edge)\n        edges[(int(edge[0]), int(edge[1]))] = 1\n        \n    p=predict[500:].values.tolist()   \n    missing_edges = set([])\n    for n1 in g.nodes:\n        for n2 in g.nodes:\n            if [n1,n2] in p:\n                continue\n#             if n1==n2:\n#                 edges[(n1, n2)] = 1\n            tmp = edges.get((n1,n2),-1)\n            if tmp == -1:\n#                 try:\n#                     if nx.shortest_path_length(g,source=n1,target=n2) > 2:\n#                         missing_edges.add((n1,n2))\n#                 except:\n#                     pass\n                missing_edges.add((n1,n2))\n    pickle.dump(missing_edges,open('data/after_eda/missing_edges_final.p','wb'))\nelse:\n    missing_edges = pickle.load(open('data/after_eda/missing_edges_final.p','rb'))")


# In[34]:


len(missing_edges)


# ## 2.2 Training and Test data split:  
# Removed edges from Graph and used as test data and after removing used that graph for creating features for Train and test data

# In[35]:


from sklearn.model_selection import train_test_split
if (not os.path.isfile('data/after_eda/train_pos_after_eda.csv')) and (not os.path.isfile('data/after_eda/test_pos_after_eda.csv')):
    #reading total data df
    df_pos = pd.read_csv('data/train.csv')
    df_neg = pd.DataFrame(list(missing_edges), columns=['source_node', 'destination_node'])
    
    print("Number of nodes in the graph with edges", df_pos.shape[0])
    print("Number of nodes in the graph without edges", df_neg.shape[0])
    
    #Trian test split 
    #Spiltted data into 80-20 
    #positive links and negative links seperatly because we need positive training data only for creating graph 
    #and for feature generation
#     X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(df_pos,np.ones(len(df_pos)),test_size=0.2, random_state=9)
#     X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(df_neg,np.zeros(len(df_neg)),test_size=0.2, random_state=9)
    X_train_pos, X_train_neg, y_train_pos, y_train_neg=df_pos,df_neg,np.ones(len(df_pos)),np.zeros(len(df_neg))
    X_test_pos,y_test_pos=predict[:500],np.ones(500)
    
    
    print('='*60)
    print("Number of nodes in the train data graph with edges", X_train_pos.shape[0],"=",y_train_pos.shape[0])
    print("Number of nodes in the train data graph without edges", X_train_neg.shape[0],"=", y_train_neg.shape[0])
    print('='*60)
    print("Number of nodes in the test data graph with edges", X_test_pos.shape[0],"=", y_test_pos.shape[0])

    #removing header and saving
    X_train_pos.to_csv('data/after_eda/train_pos_after_eda.csv',header=False, index=False)
    X_train_neg.to_csv('data/after_eda/train_neg_after_eda.csv',header=False, index=False)
    X_test_pos.to_csv('data/after_eda/test_pos_after_eda.csv',header=False, index=False)
else:
    #Graph from Traing data only 
    del missing_edges


# In[36]:


predict


# In[37]:


if (os.path.isfile('data/after_eda/train_pos_after_eda.csv')) and (os.path.isfile('data/after_eda/test_pos_after_eda.csv')):
    train_graph=nx.read_edgelist('data/after_eda/train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
    test_graph=nx.read_edgelist('data/after_eda/test_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
    for n in pd.concat([predict['source_node'], predict['destination_node']], ignore_index=True).unique():
        test_graph.add_node(n)
    print(nx.info(train_graph))
    print(nx.info(test_graph))

    # finding the unique nodes in the both train and test graphs
    train_nodes_pos = set(train_graph.nodes())
    test_nodes_pos = set(test_graph.nodes())

    trY_teY = len(train_nodes_pos.intersection(test_nodes_pos))
    trY_teN = len(train_nodes_pos - test_nodes_pos)
    teY_trN = len(test_nodes_pos - train_nodes_pos)

    print('no of people common in train and test -- ',trY_teY)
    print('no of people present in train but not present in test -- ',trY_teN)

    print('no of people present in test but not present in train -- ',teY_trN)
    print(' % of people not there in Train but exist in Test in total Test data are {} %'.format(teY_trN/len(test_nodes_pos)*100))


# > we have a cold start problem here

# In[38]:


#final train and test data sets
if (not os.path.isfile('data/after_eda/train_after_eda.csv')) and (not os.path.isfile('data/after_eda/test_after_eda.csv')) and (not os.path.isfile('data/train_y.csv')) and (os.path.isfile('data/after_eda/train_pos_after_eda.csv')) and (os.path.isfile('data/after_eda/test_pos_after_eda.csv')) and (os.path.isfile('data/after_eda/train_neg_after_eda.csv')):
    
    X_train_pos = pd.read_csv('data/after_eda/train_pos_after_eda.csv', names=['source_node', 'destination_node'])
    X_test_pos = pd.read_csv('data/after_eda/test_pos_after_eda.csv', names=['source_node', 'destination_node'])
    X_train_neg = pd.read_csv('data/after_eda/train_neg_after_eda.csv', names=['source_node', 'destination_node'])


    print('='*60)
    print("Number of nodes in the train data graph with edges", X_train_pos.shape[0])
    print("Number of nodes in the train data graph without edges", X_train_neg.shape[0])
    print('='*60)
    print("Number of nodes in the test data graph with edges", X_test_pos.shape[0])


    X_train = X_train_pos.append(X_train_neg,ignore_index=True)
    y_train = np.concatenate((y_train_pos,y_train_neg))
    X_test = predict
    
    X_train.to_csv('data/after_eda/train_after_eda.csv',header=False,index=False)
    X_test.to_csv('data/after_eda/test_after_eda.csv',header=False,index=False)
    pd.DataFrame(y_train.astype(int)).to_csv('data/train_y.csv',header=False,index=False)


# In[39]:


print("Data points in train data",X_train.shape)
print("Data points in test data",X_test.shape)
print("Shape of traget variable in train",y_train.shape)


# **Computed and store the data for featurization**

# # Features definition

# In[40]:


#Importing Libraries
# please do go through this python notebook: 
import warnings
warnings.filterwarnings("ignore")

import csv
import pandas as pd#pandas to create small dataframes 
import datetime #Convert to unix time
import time #Convert to unix time
# if numpy is not installed already : pip3 install numpy
import numpy as np#Do aritmetic operations on arrays
# matplotlib: used to plot graphs
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns#Plots
from matplotlib import rcParams#Size of plots  
from sklearn.cluster import MiniBatchKMeans, KMeans#Clustering
import math
import pickle
import os
# to install xgboost: pip3 install xgboost
import xgboost as xgb

import warnings
import networkx as nx
import pdb
import pickle
from pandas import HDFStore,DataFrame
from pandas import read_hdf
from scipy.sparse.linalg import svds, eigs
import gc
from tqdm import tqdm


# # 1. Reading Data

# In[41]:


if os.path.isfile('data/after_eda/train_pos_after_eda.csv'):
    train_graph=nx.read_edgelist('data/after_eda/train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
    print(nx.info(train_graph))
else:
    print("please run the FB_EDA.ipynb or download the files from drive")


# # 2. Similarity measures

# ## 2.1 Jaccard Distance:
# http://www.statisticshowto.com/jaccard-index/

# \begin{equation}
# j = \frac{|X\cap Y|}{|X \cup Y|} 
# \end{equation}

# In[42]:


#for followees
def jaccard_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/                                    (len(set(train_graph.successors(a)).union(set(train_graph.successors(b)))))
    except:
        return 0
    return sim


# In[43]:


#one test case
print(jaccard_for_followees(133,568))


# In[44]:


#node 1635354 not in graph 
print(jaccard_for_followees(4,15))


# In[45]:


#for followers
def jaccard_for_followers(a,b):
    try:
        if len(set(train_graph.predecessors(a))) == 0  | len(set(g.predecessors(b))) == 0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/                                 (len(set(train_graph.predecessors(a)).union(set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0


# In[46]:


print(jaccard_for_followers(133,568))


# In[47]:


#node 1635354 not in graph 
print(jaccard_for_followees(4,15))


# ## 2.2 Cosine distance

# \begin{equation}
# CosineDistance = \frac{|X\cap Y|}{|X|\cdot|Y|} 
# \end{equation}

# In[48]:


#for followees
def cosine_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/                                    (math.sqrt(len(set(train_graph.successors(a)))*len((set(train_graph.successors(b))))))
        return sim
    except:
        return 0


# In[49]:


print(cosine_for_followees(133,568))


# In[50]:


print(cosine_for_followees(4,15))


# In[51]:


def cosine_for_followers(a,b):
    try:
        
        if len(set(train_graph.predecessors(a))) == 0  | len(set(train_graph.predecessors(b))) == 0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/                                     (math.sqrt(len(set(train_graph.predecessors(a))))*(len(set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0


# In[52]:


print(cosine_for_followers(133,568))


# In[53]:


print(cosine_for_followers(4,15))


# ## 3. Ranking Measures

# https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
# 
# PageRank computes a ranking of the nodes in the graph G based on the structure of the incoming links.
# 
# <img src='PageRanks-Example.jpg'/>
# 
# Mathematical PageRanks for a simple network, expressed as percentages. (Google uses a logarithmic scale.) Page C has a higher PageRank than Page E, even though there are fewer links to C; the one link to C comes from an important page and hence is of high value. If web surfers who start on a random page have an 85% likelihood of choosing a random link from the page they are currently visiting, and a 15% likelihood of jumping to a page chosen at random from the entire web, they will reach Page E 8.1% of the time. <b>(The 15% likelihood of jumping to an arbitrary page corresponds to a damping factor of 85%.) Without damping, all web surfers would eventually end up on Pages A, B, or C, and all other pages would have PageRank zero. In the presence of damping, Page A effectively links to all pages in the web, even though it has no outgoing links of its own.</b>

# ## 3.1 Page Ranking
# 
# https://en.wikipedia.org/wiki/PageRank
# 

# In[54]:


if not os.path.isfile('data/fea_sample/page_rank.p'):
    pr = nx.pagerank(train_graph, alpha=0.85)
    pickle.dump(pr,open('data/fea_sample/page_rank.p','wb'))
else:
    pr = pickle.load(open('data/fea_sample/page_rank.p','rb'))


# In[55]:


print('min',pr[min(pr, key=pr.get)])
print('max',pr[max(pr, key=pr.get)])
print('mean',float(sum(pr.values())) / len(pr))


# In[56]:


#for imputing to nodes which are not there in Train data
mean_pr = float(sum(pr.values())) / len(pr)
print(mean_pr)


# # 4. Other Graph Features

# ## 4.1 Shortest path:

# Getting Shortest path between twoo nodes, if nodes have direct path i.e directly connected then we are removing that edge and calculating path. 

# In[57]:


#if has direct edge then deleting that edge and calculating shortest path
def compute_shortest_path_length(a,b):
    p=-1
    try:
        if train_graph.has_edge(a,b):
            train_graph.remove_edge(a,b)
            p= nx.shortest_path_length(train_graph,source=a,target=b)
            train_graph.add_edge(a,b)
        else:
            p= nx.shortest_path_length(train_graph,source=a,target=b)
        return p
    except:
        return -1


# In[58]:


#testing
compute_shortest_path_length(133, 568)


# In[59]:


#testing
compute_shortest_path_length(4,15)


# ## 4.2 Checking for same community

# In[60]:


#getting weekly connected edges from graph 
wcc=list(nx.weakly_connected_components(train_graph))
def belongs_to_same_wcc(a,b):
    index = []
    if train_graph.has_edge(b,a):
        return 1
    if train_graph.has_edge(a,b):
            for i in wcc:
                if a in i:
                    index= i
                    break
            if (b in index):
                train_graph.remove_edge(a,b)
                if compute_shortest_path_length(a,b)==-1:
                    train_graph.add_edge(a,b)
                    return 0
                else:
                    train_graph.add_edge(a,b)
                    return 1
            else:
                return 0
    else:
            for i in wcc:
                if a in i:
                    index= i
                    break
            if(b in index):
                return 1
            else:
                return 0


# In[61]:


belongs_to_same_wcc(133, 568)


# In[62]:


belongs_to_same_wcc(4,15)


# ## 4.3 Adamic/Adar Index:
# Adamic/Adar measures is defined as inverted sum of degrees of common neighbours for given two vertices.
# $$A(x,y)=\sum_{u \in N(x) \cap N(y)}\frac{1}{log(|N(u)|)}$$

# In[63]:


#adar index
def calc_adar_in(a,b):
    sum=0
    try:
        n=list(set(train_graph.successors(a)).intersection(set(train_graph.successors(b))))
        if len(n)!=0:
            for i in n:
                sum=sum+(1/np.log10(len(list(train_graph.predecessors(i)))))
            return sum
        else:
            return 0
    except:
        return 0


# In[64]:


calc_adar_in(133,568)


# In[65]:


calc_adar_in(4,15)


# ## 4.4 Is persion was following back:

# In[66]:


def follows_back(a,b):
    if train_graph.has_edge(b,a):
        return 1
    else:
        return 0


# In[67]:


follows_back(133,568)


# In[68]:


follows_back(4,15)


# ## 4.5 Katz Centrality:
# https://en.wikipedia.org/wiki/Katz_centrality
# 
# https://www.geeksforgeeks.org/katz-centrality-centrality-measure/
#  Katz centrality computes the centrality for a node 
#     based on the centrality of its neighbors. It is a 
#     generalization of the eigenvector centrality. The
#     Katz centrality for node `i` is
#  
# $$x_i = \alpha \sum_{j} A_{ij} x_j + \beta,$$
# where `A` is the adjacency matrix of the graph G 
# with eigenvalues $$\lambda$$.
# 
# The parameter $$\beta$$ controls the initial centrality and 
# 
# $$\alpha < \frac{1}{\lambda_{max}}.$$

# In[69]:


if not os.path.isfile('data/fea_sample/katz.p'):
    katz = nx.katz.katz_centrality(train_graph,alpha=0.005,beta=1)
    pickle.dump(katz,open('data/fea_sample/katz.p','wb'))
else:
    katz = pickle.load(open('data/fea_sample/katz.p','rb'))


# In[70]:


print('min',katz[min(katz, key=katz.get)])
print('max',katz[max(katz, key=katz.get)])
print('mean',float(sum(katz.values())) / len(katz))


# In[71]:


mean_katz = float(sum(katz.values())) / len(katz)
print(mean_katz)


# ## 4.6 Hits Score
# The HITS algorithm computes two numbers for a node. Authorities estimates the node value based on the incoming links. Hubs estimates the node value based on outgoing links.
# 
# https://en.wikipedia.org/wiki/HITS_algorithm

# In[72]:


if not os.path.isfile('data/fea_sample/hits.p'):
    hits = nx.hits(train_graph, max_iter=100, tol=1e-08, nstart=None, normalized=True)
    pickle.dump(hits,open('data/fea_sample/hits.p','wb'))
else:
    hits = pickle.load(open('data/fea_sample/hits.p','rb'))


# In[73]:


print('min',hits[0][min(hits[0], key=hits[0].get)])
print('max',hits[0][max(hits[0], key=hits[0].get)])
print('mean',float(sum(hits[0].values())) / len(hits[0]))


# # 5. Featurization

# ## 5. 1 Reading a sample of Data from both train and test

# In[74]:


import random
if os.path.isfile('data/after_eda/train_after_eda.csv'):
    filename = "data/after_eda/train_after_eda.csv"
    # you uncomment this line, if you dont know the lentgh of the file name
    # here we have hardcoded the number of lines as 15100030
    n_train = sum(1 for line in open(filename)) #number of records in file (excludes header)
#     n_train =  15100028
#     s = 100000 #desired sample size
#     skip_train = sorted(random.sample(range(1,n_train+1),n_train-s))
    skip_train=[]
    #https://stackoverflow.com/a/22259008/4084039


# In[75]:


if os.path.isfile('data/after_eda/train_after_eda.csv'):
    filename = "data/after_eda/test_after_eda.csv"
    # you uncomment this line, if you dont know the lentgh of the file name
    # here we have hardcoded the number of lines as 3775008
    n_test = sum(1 for line in open(filename)) #number of records in file (excludes header)
#     n_test = 3775006
#     s = 50000 #desired sample size
#     skip_test = sorted(random.sample(range(1,n_test+1),n_test-s))
    skip_test=[]
    #https://stackoverflow.com/a/22259008/4084039


# In[76]:


print("Number of rows in the train data file:", n_train)
print("Number of rows we are going to elimiate in train data are",len(skip_train))
print("Number of rows in the test data file:", n_test)
print("Number of rows we are going to elimiate in test data are",len(skip_test))


# In[77]:


df_final_train = pd.read_csv('data/after_eda/train_after_eda.csv', skiprows=skip_train, names=['source_node', 'destination_node'])
df_final_train['indicator_link'] = pd.read_csv('data/train_y.csv', skiprows=skip_train, names=['indicator_link'])
print("Our train matrix size ",df_final_train.shape)
df_final_train.head(2)


# In[78]:


df_final_test = pd.read_csv('data/after_eda/test_after_eda.csv', skiprows=skip_test, names=['source_node', 'destination_node'])
# df_final_test['indicator_link'] = pd.read_csv('data/test_y.csv', skiprows=skip_test, names=['indicator_link'])
print("Our test matrix size ",df_final_test.shape)
df_final_test.head(2)


# ## 5.2 Adding a set of features
# 
# __we will create these each of these features for both train and test data points__
# <ol>
# <li>jaccard_followers</li>
# <li>jaccard_followees</li>
# <li>cosine_followers</li>
# <li>cosine_followees</li>
# <li>num_followers_s</li>
# <li>num_followees_s</li>
# <li>num_followers_d</li>
# <li>num_followees_d</li>
# <li>inter_followers</li>
# <li>inter_followees</li>
# </ol>

# In[79]:


if not os.path.isfile('data/fea_sample/storage_sample_stage1.h5'):
    #mapping jaccrd followers to train and test data
    df_final_train['jaccard_followers'] = df_final_train.apply(lambda row:
                                            jaccard_for_followers(row['source_node'],row['destination_node']),axis=1)
    df_final_test['jaccard_followers'] = df_final_test.apply(lambda row:
                                            jaccard_for_followers(row['source_node'],row['destination_node']),axis=1)

    #mapping jaccrd followees to train and test data
    df_final_train['jaccard_followees'] = df_final_train.apply(lambda row:
                                            jaccard_for_followees(row['source_node'],row['destination_node']),axis=1)
    df_final_test['jaccard_followees'] = df_final_test.apply(lambda row:
                                            jaccard_for_followees(row['source_node'],row['destination_node']),axis=1)
    

    #mapping jaccrd followers to train and test data
    df_final_train['cosine_followers'] = df_final_train.apply(lambda row:
                                            cosine_for_followers(row['source_node'],row['destination_node']),axis=1)
    df_final_test['cosine_followers'] = df_final_test.apply(lambda row:
                                            cosine_for_followers(row['source_node'],row['destination_node']),axis=1)

    #mapping jaccrd followees to train and test data
    df_final_train['cosine_followees'] = df_final_train.apply(lambda row:
                                            cosine_for_followees(row['source_node'],row['destination_node']),axis=1)
    df_final_test['cosine_followees'] = df_final_test.apply(lambda row:
                                            cosine_for_followees(row['source_node'],row['destination_node']),axis=1)


# In[80]:


def compute_features_stage1(df_final):
    #calculating no of followers followees for source and destination
    #calculating intersection of followers and followees for source and destination
    num_followers_s=[]
    num_followees_s=[]
    num_followers_d=[]
    num_followees_d=[]
    inter_followers=[]
    inter_followees=[]
    for i,row in df_final.iterrows():
        try:
            s1=set(train_graph.predecessors(row['source_node']))
            s2=set(train_graph.successors(row['source_node']))
        except:
            s1 = set()
            s2 = set()
        try:
            d1=set(train_graph.predecessors(row['destination_node']))
            d2=set(train_graph.successors(row['destination_node']))
        except:
            d1 = set()
            d2 = set()
        num_followers_s.append(len(s1))
        num_followees_s.append(len(s2))

        num_followers_d.append(len(d1))
        num_followees_d.append(len(d2))

        inter_followers.append(len(s1.intersection(d1)))
        inter_followees.append(len(s2.intersection(d2)))
    
    return num_followers_s, num_followers_d, num_followees_s, num_followees_d, inter_followers, inter_followees


# In[81]:


if not os.path.isfile('data/fea_sample/storage_sample_stage1.h5'):
    df_final_train['num_followers_s'], df_final_train['num_followers_d'],     df_final_train['num_followees_s'], df_final_train['num_followees_d'],     df_final_train['inter_followers'], df_final_train['inter_followees']= compute_features_stage1(df_final_train)
    
    df_final_test['num_followers_s'], df_final_test['num_followers_d'],     df_final_test['num_followees_s'], df_final_test['num_followees_d'],     df_final_test['inter_followers'], df_final_test['inter_followees']= compute_features_stage1(df_final_test)
    
    hdf = HDFStore('data/fea_sample/storage_sample_stage1.h5')
    hdf.put('train_df',df_final_train, format='table', data_columns=True)
    hdf.put('test_df',df_final_test, format='table', data_columns=True)
    hdf.close()
else:
    df_final_train = read_hdf('data/fea_sample/storage_sample_stage1.h5', 'train_df',mode='r')
    df_final_test = read_hdf('data/fea_sample/storage_sample_stage1.h5', 'test_df',mode='r')


# ## 5.3 Adding new set of features
# 
# __we will create these each of these features for both train and test data points__
# <ol>
# <li>adar index</li>
# <li>is following back</li>
# <li>belongs to same weakly connect components</li>
# <li>shortest path between source and destination</li>
# </ol>

# In[82]:


if not os.path.isfile('data/fea_sample/storage_sample_stage2.h5'):
    #mapping adar index on train
    df_final_train['adar_index'] = df_final_train.apply(lambda row: calc_adar_in(row['source_node'],row['destination_node']),axis=1)
    #mapping adar index on test
    df_final_test['adar_index'] = df_final_test.apply(lambda row: calc_adar_in(row['source_node'],row['destination_node']),axis=1)

    #--------------------------------------------------------------------------------------------------------
    #mapping followback or not on train
    df_final_train['follows_back'] = df_final_train.apply(lambda row: follows_back(row['source_node'],row['destination_node']),axis=1)

    #mapping followback or not on test
    df_final_test['follows_back'] = df_final_test.apply(lambda row: follows_back(row['source_node'],row['destination_node']),axis=1)

    #--------------------------------------------------------------------------------------------------------
    #mapping same component of wcc or not on train
    df_final_train['same_comp'] = df_final_train.apply(lambda row: belongs_to_same_wcc(row['source_node'],row['destination_node']),axis=1)

    ##mapping same component of wcc or not on train
    df_final_test['same_comp'] = df_final_test.apply(lambda row: belongs_to_same_wcc(row['source_node'],row['destination_node']),axis=1)
    
    #--------------------------------------------------------------------------------------------------------
    #mapping shortest path on train 
    df_final_train['shortest_path'] = df_final_train.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)
    #mapping shortest path on test
    df_final_test['shortest_path'] = df_final_test.apply(lambda row: compute_shortest_path_length(row['source_node'],row['destination_node']),axis=1)

    hdf = HDFStore('data/fea_sample/storage_sample_stage2.h5')
    hdf.put('train_df',df_final_train, format='table', data_columns=True)
    hdf.put('test_df',df_final_test, format='table', data_columns=True)
    hdf.close()
else:
    df_final_train = read_hdf('data/fea_sample/storage_sample_stage2.h5', 'train_df',mode='r')
    df_final_test = read_hdf('data/fea_sample/storage_sample_stage2.h5', 'test_df',mode='r')


# ## 5.4 Adding new set of features
# 
# __we will create these each of these features for both train and test data points__
# <ol>
# <li>Weight Features
#     <ul>
#         <li>weight of incoming edges</li>
#         <li>weight of outgoing edges</li>
#         <li>weight of incoming edges + weight of outgoing edges</li>
#         <li>weight of incoming edges * weight of outgoing edges</li>
#         <li>2*weight of incoming edges + weight of outgoing edges</li>
#         <li>weight of incoming edges + 2*weight of outgoing edges</li>
#     </ul>
# </li>
# <li>Page Ranking of source</li>
# <li>Page Ranking of dest</li>
# <li>katz of source</li>
# <li>katz of dest</li>
# <li>hubs of source</li>
# <li>hubs of dest</li>
# <li>authorities_s of source</li>
# <li>authorities_s of dest</li>
# </ol>

# #### Weight Features

# In order to determine the similarity of nodes, an edge weight value was calculated between nodes. Edge weight decreases as the neighbor count goes up. Intuitively, consider one million people following a celebrity on a social network then chances are most of them never met each other or the celebrity. On the other hand, if a user has 30 contacts in his/her social network, the chances are higher that many of them know each other. 
# `credit` - Graph-based Features for Supervised Link Prediction
# William Cukierski, Benjamin Hamner, Bo Yang

# \begin{equation}
# W = \frac{1}{\sqrt{1+|X|}}
# \end{equation}

# it is directed graph so calculated Weighted in and Weighted out differently

# In[83]:


#weight for source and destination of each link
Weight_in = {}
Weight_out = {}
for i in  tqdm(train_graph.nodes()):
    s1=set(train_graph.predecessors(i))
    w_in = 1.0/(np.sqrt(1+len(s1)))
    Weight_in[i]=w_in
    
    s2=set(train_graph.successors(i))
    w_out = 1.0/(np.sqrt(1+len(s2)))
    Weight_out[i]=w_out
    
#for imputing with mean
mean_weight_in = np.mean(list(Weight_in.values()))
mean_weight_out = np.mean(list(Weight_out.values()))


# In[84]:


if not os.path.isfile('data/fea_sample/storage_sample_stage3.h5'):
    #mapping to pandas train
    df_final_train['weight_in'] = df_final_train.destination_node.apply(lambda x: Weight_in.get(x,mean_weight_in))
    df_final_train['weight_out'] = df_final_train.source_node.apply(lambda x: Weight_out.get(x,mean_weight_out))

    #mapping to pandas test
    df_final_test['weight_in'] = df_final_test.destination_node.apply(lambda x: Weight_in.get(x,mean_weight_in))
    df_final_test['weight_out'] = df_final_test.source_node.apply(lambda x: Weight_out.get(x,mean_weight_out))


    #some features engineerings on the in and out weights
    df_final_train['weight_f1'] = df_final_train.weight_in + df_final_train.weight_out
    df_final_train['weight_f2'] = df_final_train.weight_in * df_final_train.weight_out
    df_final_train['weight_f3'] = (2*df_final_train.weight_in + 1*df_final_train.weight_out)
    df_final_train['weight_f4'] = (1*df_final_train.weight_in + 2*df_final_train.weight_out)

    #some features engineerings on the in and out weights
    df_final_test['weight_f1'] = df_final_test.weight_in + df_final_test.weight_out
    df_final_test['weight_f2'] = df_final_test.weight_in * df_final_test.weight_out
    df_final_test['weight_f3'] = (2*df_final_test.weight_in + 1*df_final_test.weight_out)
    df_final_test['weight_f4'] = (1*df_final_test.weight_in + 2*df_final_test.weight_out)


# In[85]:


if not os.path.isfile('data/fea_sample/storage_sample_stage3.h5'):
    
    #page rank for source and destination in Train and Test
    #if anything not there in train graph then adding mean page rank 
    df_final_train['page_rank_s'] = df_final_train.source_node.apply(lambda x:pr.get(x,mean_pr))
    df_final_train['page_rank_d'] = df_final_train.destination_node.apply(lambda x:pr.get(x,mean_pr))

    df_final_test['page_rank_s'] = df_final_test.source_node.apply(lambda x:pr.get(x,mean_pr))
    df_final_test['page_rank_d'] = df_final_test.destination_node.apply(lambda x:pr.get(x,mean_pr))
    #================================================================================

    #Katz centrality score for source and destination in Train and test
    #if anything not there in train graph then adding mean katz score
    df_final_train['katz_s'] = df_final_train.source_node.apply(lambda x: katz.get(x,mean_katz))
    df_final_train['katz_d'] = df_final_train.destination_node.apply(lambda x: katz.get(x,mean_katz))

    df_final_test['katz_s'] = df_final_test.source_node.apply(lambda x: katz.get(x,mean_katz))
    df_final_test['katz_d'] = df_final_test.destination_node.apply(lambda x: katz.get(x,mean_katz))
    #================================================================================

    #Hits algorithm score for source and destination in Train and test
    #if anything not there in train graph then adding 0
    df_final_train['hubs_s'] = df_final_train.source_node.apply(lambda x: hits[0].get(x,0))
    df_final_train['hubs_d'] = df_final_train.destination_node.apply(lambda x: hits[0].get(x,0))

    df_final_test['hubs_s'] = df_final_test.source_node.apply(lambda x: hits[0].get(x,0))
    df_final_test['hubs_d'] = df_final_test.destination_node.apply(lambda x: hits[0].get(x,0))
    #================================================================================

    #Hits algorithm score for source and destination in Train and Test
    #if anything not there in train graph then adding 0
    df_final_train['authorities_s'] = df_final_train.source_node.apply(lambda x: hits[1].get(x,0))
    df_final_train['authorities_d'] = df_final_train.destination_node.apply(lambda x: hits[1].get(x,0))

    df_final_test['authorities_s'] = df_final_test.source_node.apply(lambda x: hits[1].get(x,0))
    df_final_test['authorities_d'] = df_final_test.destination_node.apply(lambda x: hits[1].get(x,0))
    #================================================================================

    hdf = HDFStore('data/fea_sample/storage_sample_stage3.h5')
    hdf.put('train_df',df_final_train, format='table', data_columns=True)
    hdf.put('test_df',df_final_test, format='table', data_columns=True)
    hdf.close()
else:
    df_final_train = read_hdf('data/fea_sample/storage_sample_stage3.h5', 'train_df',mode='r')
    df_final_test = read_hdf('data/fea_sample/storage_sample_stage3.h5', 'test_df',mode='r')


# ## 5.5 Adding new set of features
# 
# __we will create these each of these features for both train and test data points__
# <ol>
# <li>SVD features for both source and destination</li>
# </ol>

# In[86]:


def svd(x, S):
    try:
        z = sadj_dict[x]
        return S[z]
    except:
        return [0,0,0,0,0,0]


# In[87]:


#for svd features to get feature vector creating a dict node val and index in svd vector
sadj_col = sorted(train_graph.nodes())
sadj_dict = { val:idx for idx,val in enumerate(sadj_col)}


# In[88]:


Adj = nx.adjacency_matrix(train_graph,nodelist=sorted(train_graph.nodes())).asfptype()


# In[89]:


U, s, V = svds(Adj, k = 6)
print('Adjacency matrix Shape',Adj.shape)
print('U Shape',U.shape)
print('V Shape',V.shape)
print('s Shape',s.shape)


# In[90]:


if not os.path.isfile('data/fea_sample/storage_sample_stage4.h5'):
    #===================================================================================================
    
    df_final_train[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] =     df_final_train.source_node.apply(lambda x: svd(x, U)).apply(pd.Series)
    
    df_final_train[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5','svd_u_d_6']] =     df_final_train.destination_node.apply(lambda x: svd(x, U)).apply(pd.Series)
    #===================================================================================================
    
    df_final_train[['svd_v_s_1','svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6',]] =     df_final_train.source_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)

    df_final_train[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5','svd_v_d_6']] =     df_final_train.destination_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)
    #===================================================================================================
    
    df_final_test[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] =     df_final_test.source_node.apply(lambda x: svd(x, U)).apply(pd.Series)
    
    df_final_test[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5','svd_u_d_6']] =     df_final_test.destination_node.apply(lambda x: svd(x, U)).apply(pd.Series)

    #===================================================================================================
    
    df_final_test[['svd_v_s_1','svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6',]] =     df_final_test.source_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)

    df_final_test[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5','svd_v_d_6']] =     df_final_test.destination_node.apply(lambda x: svd(x, V.T)).apply(pd.Series)
    #===================================================================================================

    hdf = HDFStore('data/fea_sample/storage_sample_stage4.h5')
    hdf.put('train_df',df_final_train, format='table', data_columns=True)
    hdf.put('test_df',df_final_test, format='table', data_columns=True)
    hdf.close()


# **Store the data as final for machine learning models**

# In[91]:


#reading
from pandas import read_hdf
df_final_train = read_hdf('data/fea_sample/storage_sample_stage4.h5', 'train_df',mode='r')
df_final_test = read_hdf('data/fea_sample/storage_sample_stage4.h5', 'test_df',mode='r')


# In[92]:


df_final_train=df_final_train.replace([np.inf,-np.inf],np.nan)
df_final_train=df_final_train.fillna(0)
df_final_test=df_final_test.replace([np.inf,-np.inf],np.nan)
df_final_test=df_final_test.fillna(0)


# In[93]:


# # 選出確定的neg smaple
# # predict_500[predict_500['indicator_link']==0]
# # df_final_train[df_final_train['indicator_link']==0]
# neg1=pd.merge(predict_500[predict_500['indicator_link']==0],df_final_train,how='inner')
# neg1

# #將neg1從原本training data中去除
# drop_index=[df_final_train[(df_final_train['source_node']==nodepair[0]) & (df_final_train['destination_node']==nodepair[1])].index[0] for i,nodepair in neg1.iterrows()]
# df_final_train.drop(drop_index,inplace=True)

# neg2=df_final_train[df_final_train['indicator_link']==0].sample(n=20510-len(neg1), random_state=1)
# neg=pd.concat([neg1, neg2], ignore_index=True)


# In[94]:


# neg=df_final_train[df_final_train['indicator_link']==0].sample(n=20510, random_state=1)


# In[95]:


# pos=df_final_train[df_final_train['indicator_link']==1]


# In[96]:


# df_final_train=pd.concat([pos, neg], ignore_index=True)
df_final_train


# In[97]:


df_final_test


# In[98]:


df_final_train.columns


# In[99]:


y_train = df_final_train.indicator_link
# y_test = df_final_test.indicator_link


# In[100]:


# df_final_train.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)
# df_final_test.drop(['source_node', 'destination_node'],axis=1,inplace=True)

df_final_train=df_final_train[['cosine_followers','inter_followers','adar_index']]
df_final_test=df_final_test[['cosine_followers','inter_followers','adar_index']]

# df_final_train=df_final_train[['adar_index']]
# df_final_test=df_final_test[['adar_index']]


# In[101]:


from sklearn.metrics import f1_score
estimators = [10,50,100,250,450]
train_scores = []
test_scores = []
for i in estimators:
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=52, min_samples_split=120,
            min_weight_fraction_leaf=0.0, n_estimators=i, n_jobs=-1,random_state=25,verbose=0,warm_start=False)
    clf.fit(df_final_train,y_train)
    train_sc = f1_score(y_train,clf.predict(df_final_train))
#     test_sc = f1_score(y_test,clf.predict(df_final_test))
#     test_scores.append(test_sc)
    train_scores.append(train_sc)
    print('Estimators = ',i,'Train Score',train_sc)
plt.plot(estimators,train_scores,label='Train Score')
# plt.plot(estimators,test_scores,label='Test Score')
plt.xlabel('Estimators')
plt.ylabel('Score')
plt.title('Estimators vs score at depth of 5')


# In[102]:


depths = [3,9,11,15,20,35,50,70,130]
train_scores = []
test_scores = []
for i in depths:
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=i, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=52, min_samples_split=120,
            min_weight_fraction_leaf=0.0, n_estimators=115, n_jobs=-1,random_state=25,verbose=0,warm_start=False)
    clf.fit(df_final_train,y_train)
    train_sc = f1_score(y_train,clf.predict(df_final_train))
#     test_sc = f1_score(y_test,clf.predict(df_final_test))
#     test_scores.append(test_sc)
    train_scores.append(train_sc)
    print('depth = ',i,'Train Score',train_sc)
plt.plot(depths,train_scores,label='Train Score')
# plt.plot(depths,test_scores,label='Test Score')
plt.xlabel('Depth')
plt.ylabel('Score')
plt.title('Depth vs score at depth of 5 at estimators = 115')
plt.show()


# In[103]:


from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

param_dist = {"n_estimators":sp_randint(105,125),
              "max_depth": sp_randint(10,15),
              "min_samples_split": sp_randint(110,190),
              "min_samples_leaf": sp_randint(25,65)}

clf = RandomForestClassifier(random_state=25,n_jobs=-1)

rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=5,cv=10,scoring='f1',random_state=25)

rf_random.fit(df_final_train,y_train)
print('mean test scores',rf_random.cv_results_['mean_test_score'])
# print('mean train scores',rf_random.cv_results_['mean_train_score'])


# In[104]:


print(rf_random.best_estimator_)


# In[105]:


clf=rf_random.best_estimator_


# In[106]:


# clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=14, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=28, min_samples_split=111,
#             min_weight_fraction_leaf=0.0, n_estimators=121, n_jobs=-1,
#             oob_score=False, random_state=25, verbose=0, warm_start=False)


# In[107]:


clf.fit(df_final_train,y_train)
y_train_pred = clf.predict(df_final_train)
y_test_pred = clf.predict(df_final_test)
y_test_pred


# In[108]:


#產生答案
answer=pd.DataFrame({'predict_nodepair_id':predict.index, 'ans':y_test_pred})
answer=answer.to_csv('answer_RandomForest_gt.csv',index=False)
answer


# In[109]:


from sklearn.metrics import f1_score
print('Train f1 score',f1_score(y_train,y_train_pred))
# print('Test f1 score',f1_score(y_test,y_test_pred))


# In[110]:


from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    
    A =(((C.T)/(C.sum(axis=1))).T)
    
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(20,4))
    
    labels = [0,1]
    # representing A in heatmap format
    cmap=sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()


# In[111]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
# print('Test confusion_matrix')
# plot_confusion_matrix(y_test,y_test_pred)


# In[112]:


# from sklearn.metrics import roc_curve, auc
# fpr,tpr,ths = roc_curve(y_test,y_test_pred)
# auc_sc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic with test data')
# plt.legend()
# plt.show()


# In[113]:


features = df_final_train.columns
importances = clf.feature_importances_
indices = (np.argsort(importances))[-25:]
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# Before using GBDT, lets create a some new features which are used in research papers!!!

# In[114]:


#reading
from pandas import read_hdf
df_final_train = read_hdf('./data/fea_sample/storage_sample_stage4.h5', 'train_df',mode='r')
df_final_test = read_hdf('./data/fea_sample/storage_sample_stage4.h5', 'test_df',mode='r')


# 
# 2.   Add another feature called Preferential Attachment with followers and followees data of vertex. you can check about Preferential Attachment in below link http://be.amazd.com/link-prediction/
# 
# 

# In[115]:


df_final_train['preferential_attachment'] = (df_final_train['num_followers_s'] + df_final_train['num_followees_s']) * ( df_final_train['num_followers_d'] + df_final_train['num_followees_d'])
df_final_test['preferential_attachment'] = (df_final_test['num_followers_s'] + df_final_test['num_followees_s']) * ( df_final_train['num_followers_d'] + df_final_test['num_followees_d'])


# In[116]:


df_final_train=df_final_train.replace([np.inf,-np.inf],np.nan)
df_final_train=df_final_train.fillna(0)
df_final_test=df_final_test.replace([np.inf,-np.inf],np.nan)
df_final_test=df_final_test.fillna(0)


# In[117]:


# #選出確定的neg smaple
# neg1=pd.merge(predict_500[predict_500['indicator_link']==0],df_final_train,how='inner')
# neg1

# #將neg1從原本training data中去除
# drop_index=[df_final_train[(df_final_train['source_node']==nodepair[0]) & (df_final_train['destination_node']==nodepair[1])].index[0] for i,nodepair in neg1.iterrows()]
# df_final_train.drop(drop_index,inplace=True)

# neg2=df_final_train[df_final_train['indicator_link']==0].sample(n=20510-len(neg1), random_state=1)
# neg=pd.concat([neg1, neg2], ignore_index=True)


# In[118]:


# pos=df_final_train[df_final_train['indicator_link']==1]


# In[119]:


# df_final_train=pd.concat([pos, neg], ignore_index=True)
df_final_train


#   3. Add feature called svd_dot. you can calculate svd_dot as Dot product between sourse node svd and destination node svd features. you can read about this in below pdf https://storage.googleapis.com/kaggle-forum-message-attachments/2594/supervised_link_prediction.pdf

# In[120]:


def svd_dot(S,D):
    list_var = []
    A = S.values
    B = D.values
    for i in range(len(A)):
        list_var.append(np.dot(A[i],B[i]))
    df = pd.DataFrame(list_var)
    return df


# In[121]:


u_s = df_final_train[['svd_u_s_1', 'svd_u_s_2', 'svd_u_s_3', 'svd_u_s_4','svd_u_s_5', 'svd_u_s_6']]
u_d = df_final_train[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4','svd_u_d_5', 'svd_u_d_6']]
df_final_train['svd_dot_u'] = svd_dot(u_s,u_d)

v_s = df_final_train[['svd_v_s_1', 'svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4','svd_v_s_5', 'svd_v_s_6']]
v_d = df_final_train[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4','svd_v_d_5', 'svd_v_d_6']]
df_final_train['svd_dot_v'] = svd_dot(v_s,v_d)

u_s = df_final_test[['svd_u_s_1', 'svd_u_s_2', 'svd_u_s_3', 'svd_u_s_4','svd_u_s_5', 'svd_u_s_6']]
u_d = df_final_test[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4','svd_u_d_5', 'svd_u_d_6']]
df_final_test['svd_dot_u'] = svd_dot(u_s,u_d)

v_s = df_final_test[['svd_v_s_1', 'svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4','svd_v_s_5', 'svd_v_s_6']]
v_d = df_final_test[['svd_v_s_1', 'svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4','svd_v_s_5', 'svd_v_s_6']]
df_final_test['svd_dot_v'] = svd_dot(v_s,v_d)


# In[122]:


df_final_train.columns


# In[123]:


y_train = df_final_train.indicator_link
# y_test = df_final_test.indicator_link


# In[124]:


# df_final_train.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)
# df_final_test.drop(['source_node', 'destination_node'],axis=1,inplace=True)

df_final_train=df_final_train[['adar_index']]
df_final_test=df_final_test[['adar_index']]


# In[125]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score


# In[126]:


param_dist = {"n_estimators":[10,50,100,250],
              "max_depth": [3,9,11,15,20,35,50,70,130]}

clf = GradientBoostingClassifier(random_state=25,verbose=1)

rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=5,cv=3,scoring='f1',random_state=25)

rf_random.fit(df_final_train,y_train)
print('mean test scores',rf_random.cv_results_['mean_test_score'])


# In[127]:


print(rf_random.best_estimator_)


# In[128]:


clf = rf_random.best_estimator_


# In[129]:


clf.fit(df_final_train,y_train)
y_train_pred = clf.predict(df_final_train)
y_test_pred = clf.predict(df_final_test)
y_test_pred 


# In[130]:


#產生答案
answer=pd.DataFrame({'predict_nodepair_id':predict.index, 'ans':y_test_pred})
answer=answer.to_csv('answer_GBDT_gt.csv',index=False)
answer


# In[131]:


from sklearn.metrics import f1_score
print('Train f1 score',f1_score(y_train,y_train_pred))
# print('Test f1 score',f1_score(y_test,y_test_pred))


# In[132]:


from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    
    A =(((C.T)/(C.sum(axis=1))).T)
    
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(20,4))
    
    labels = [0,1]
    # representing A in heatmap format
    sns.set()
    cmap=sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()


# In[133]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,y_train_pred)
# print('Test confusion_matrix')
# plot_confusion_matrix(y_test,y_test_pred)


# In[134]:


# from sklearn.metrics import roc_curve, auc
# fpr,tpr,ths = roc_curve(y_test,y_test_pred)
# auc_sc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic with test data')
# plt.legend()
# plt.show()


# In[135]:


features = df_final_train.columns
importances = clf.feature_importances_
indices = (np.argsort(importances))[-25:]
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:




