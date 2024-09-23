import os
import pandas as pd
import numpy as np
import csv
import networkx as nx
import random
random.seed(25)

def LoadData(filedir, filename):
    path = os.path.join(filedir, filename+'.csv')
    data = pd.read_csv(path)
    return data

def SaveLink(data):
    link = []
    for i in range(len(data)):
        link.append((data['node_1'][i], data['node_2'][i]))
    return link

def NodeListGen(data):
    nodes ={}
    for i in range(len(data)):
        nodes[data[i][0]] = 1
        nodes[data[i][1]] = 1
    #for i in range(len(data1)):
    #    nodes[data1[i][0]] = 1
    #    nodes[data1[i][1]] = 1
    node_list = []
    for i in nodes:
        node_list.append(i)
    node_list = sorted(node_list)
    return node_list

def GraphConstruct(link, node_num):
    G = nx.Graph()
    G.add_nodes_from(range(node_num))
    G.add_edges_from(link)
    #nx.draw(G, with_labels=True)
    return G

def UnconnectedLinkFind(G, node_list, num):
    adj_G = nx.to_numpy_matrix(G, nodelist = node_list)
    unconnected_link = []
    cnt = 0
    while cnt < num:
        i = random.randint(0, len(node_list)-1)
        j = random.randint(0, len(node_list)-1)
        if adj_G[i, j] == 0 and i!=j:
            adj_G[i, j] = 1
            unconnected_link.append([node_list[i],node_list[j]])
            cnt += 1
    return unconnected_link

def dfGenerate(link, isPos):
    node_1_linked = [link[i][0] for i in range(len(link))]
    node_2_linked = [link[i][1] for i in range(len(link))]
    df = pd.DataFrame()
    df['node_1'] = node_1_linked
    df['node_2'] = node_2_linked
    if isPos:
        df['link'] = 1
    else:
        df['link'] = 0
    print(df['link'].value_counts())
    return df

def SavedfLink(df):
    node1 = df['node_1'].tolist()
    node2 = df['node_2'].tolist()
    link = []
    for i in range(len(node1)):
        link.append((node1[i], node2[i]))
    print("# of Link:", len(link))
    return link

def SaveEmbedding(data, idx = 'Embed'):
    embedding = []
    for i in range(len(data)):
        embedding.append(data[str(idx)][i])
    return embedding

def SaveHEmbedding(data, dim = 16):
    e = list(range(dim))
    for i in range(dim):
        e[i] = SaveEmbedding(data, idx = i)
    return e
    