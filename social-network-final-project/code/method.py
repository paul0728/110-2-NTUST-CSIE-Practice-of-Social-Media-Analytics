import numpy as np
import networkx as nx
import community
from sklearn.decomposition import PCA

def jaccard_distance(G, link):
    score = nx.jaccard_coefficient(G, link)
    result = []
    for u, v, p in score:
        result.append(p)
    return result

def cosine_distance(G, link):
    result = []
    for n1, n2 in link:
        X = set( G.neighbors(n1))
        Y = set( G.neighbors(n2))
        if len(X)*len(Y) == 0:
            result.append(0)
        else:
            result.append(len( X&Y ) / ( len(X)*len(Y) ))
    return result

def shortest_path_length(G, link):
    result = []
    for n1, n2 in link:
        try:
            result.append(nx.shortest_path_length(G, source=n1, target=n2))
        except:
            result.append(0)
    return result

def same_community(G, link, cc = False):
    result = []
    if cc:
        cc=list(nx.connected_components(G))
        dic={}
        for i,c in enumerate(cc):
            for node in c:
                dic[node]=i
        for n1, n2 in link:
            if dic.get(n1,-1)==dic.get(n2,-1) and dic.get(n1,-1)!=-1:
                result.append(1)
            else:
                result.append(0)
    else:
        partition = community.best_partition(G)
        for n1, n2 in link:
            if partition[n1] == partition[n2]:
                result.append(1)
            else:
                result.append(0)
    return result

def common_neighbors(G, link):
    result = []
    for n1, n2 in link:
        result.append(len(sorted(nx.common_neighbors(G, n1, n2))))
    return result

def adamic_adar(G, link):
    #filter out node1 = node2
    tmp_link = []
    same_node_index_list = []
    for idx in range(len(link)):
        if link[idx][0] == link[idx][1]:
            same_node_index_list.append(idx)
        else:
            tmp_link.append(link[idx])
    
    s = []
    score = nx.adamic_adar_index(G, tmp_link)
    for u, v, p in score:
        s.append(p)
    result = []
    cnt = 0
    for idx in range(len(link)):
        if idx in same_node_index_list:
            result.append(0)
        else:
            result.append(s[cnt])
            cnt += 1
    return result

def resource_allocation_index(G, link):
    score = nx.resource_allocation_index(G, link)
    result = []
    for u, v, p in score:
        result.append(p)
    return result

def preferential_attachment(G, link):
    score = nx.preferential_attachment(G, link)
    result = []
    for u, v, p in score:
        result.append(p)
    return result

def n2v(model, link):
    result = []
    for n1, n2 in link:
        result.append(list(model.wv[str(n1)] + model.wv[str(n2)]))
    return result

def n2v_pca(model, link):
    X = []
    for n1, n2 in link:
        X.append(model.wv[str(n1)] + model.wv[str(n2)])
    pca = PCA(n_components=1)
    pca.fit(np.array(X))
    X_pca = pca.transform(X)
    print(X_pca.shape)
    result = list(X_pca)
    print(len(result))
    return result

def get_embedding(embed, link):
    result = []
    for n1, n2 in link:
        result.append(embed[n1] + embed[n2])
    return result

def get_n2v_embedding(embed, link, dim_idx):
    result = []
    for i in range(len(link)):
        result.append(embed[i][dim_idx])
    return result
