#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df=pd.read_csv("./results/RandomForest_f1_transposed.csv")
df.T.to_csv("./results/RandomForest_f1.csv",header=False)

df=pd.read_csv("./results/GBDT_f1_transposed.csv")
df.T.to_csv("./results/GBDT_f1.csv",header=False)

