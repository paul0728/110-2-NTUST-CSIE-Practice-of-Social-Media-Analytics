#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
#RandomForest
df=pd.read_csv("./results/RandomForest_Train_f1_transposed.csv")
df.T.to_csv("./results/RandomForest_Train_f1.csv",header=False)
os.remove("./results/RandomForest_Train_f1_transposed.csv")

df=pd.read_csv("./results/RandomForest_Test_f1_transposed.csv")
df.T.to_csv("./results/RandomForest_Test_f1.csv",header=False)
os.remove("./results/RandomForest_Test_f1_transposed.csv")

#GBDT
df=pd.read_csv("./results/GBDT_Train_f1_transposed.csv")
df.T.to_csv("./results/GBDT_Train_f1.csv",header=False)
os.remove("./results/GBDT_Train_f1_transposed.csv")

df=pd.read_csv("./results/GBDT_Test_f1_transposed.csv")
df.T.to_csv("./results/GBDT_Test_f1.csv",header=False)
os.remove("./results/GBDT_Test_f1_transposed.csv")
