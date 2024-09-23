import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import svds

def svd(x, S, dic):
    try:
        z = dic[x]
        return S[z]
    except:
        return [0,0,0,0,0,0]
    
def svd_dot(S,D):
    result = []
    A = S.values
    B = D.values
    for i in range(len(A)):
        result.append(np.dot(A[i],B[i]))
    return result

def SVD(G, df):
    sadj_col = sorted(G.nodes())
    sadj_dict = {val: idx for idx, val in enumerate(sadj_col)}
    Adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).asfptype()
    U, s, V = svds(Adj, k=6)
    
    df[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] = \
    df['node_1'].apply(lambda x: svd(x, U, sadj_dict)).apply(pd.Series)

    df[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5','svd_u_d_6']] = \
    df['node_2'].apply(lambda x: svd(x, U, sadj_dict)).apply(pd.Series)

    df[['svd_v_s_1','svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6']] = \
    df['node_1'].apply(lambda x: svd(x, V.T, sadj_dict)).apply(pd.Series)

    df[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5','svd_v_d_6']] = \
    df['node_2'].apply(lambda x: svd(x, V.T, sadj_dict)).apply(pd.Series)
    
    u_s = df[['svd_u_s_1', 'svd_u_s_2', 'svd_u_s_3', 'svd_u_s_4','svd_u_s_5', 'svd_u_s_6']]
    u_d = df[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4','svd_u_d_5', 'svd_u_d_6']]
    result1 = svd_dot(u_s,u_d)

    v_s = df[['svd_v_s_1', 'svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4','svd_v_s_5', 'svd_v_s_6']]
    v_d = df[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4','svd_v_d_5', 'svd_v_d_6']]
    result2 = svd_dot(v_s,v_d)
    
    return result1, result2

