import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import csv
import random
import networkx as nx
import pickle
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

from utils import parse_args
from graph_construct import LoadData, SaveLink, NodeListGen, GraphConstruct, UnconnectedLinkFind, dfGenerate, SavedfLink, SaveEmbedding
from method import jaccard_distance, cosine_distance, shortest_path_length, same_community, common_neighbors, adamic_adar, resource_allocation_index, preferential_attachment, n2v, get_embedding
from method_svd import SVD
from plot import Confusion_Matrix, ROC_Curve, Importance

if __name__ == "__main__":
    args = parse_args()
    random.seed(25)


    #load data
    filedir = "facebook_clean_data"
    dataset = str(args.dataset)
    train_data = LoadData(filedir, dataset + "_edges")
    
    print("Data loaded")
    link = SaveLink(train_data)
    
    node_list = NodeListGen(link)
    with open('node_list.pickle', 'wb') as f:
        pickle.dump(node_list, f)
    # print(node_list)
    node_num = len(node_list)

    #construct graph
    print("Graph Construct")
    all_G = GraphConstruct(link, node_num)
    
    #find unconnected link
    print("neg+pos")
    unconnected_link = UnconnectedLinkFind(all_G, node_list, len(link))
    print('unconnected_link =',len(unconnected_link ))
    with open('unconnected_link.pickle', 'wb') as f:
        pickle.dump(unconnected_link, f)
    # print(unconnected_link)
    
    #dataframe = pos + neg
    #df = dfGenerate(link, unconnected_link)
    pos_df = dfGenerate(link, True)
    neg_df = dfGenerate(unconnected_link, False)
    
    pos_xtrain, pos_xtest, pos_ytrain, pos_ytest = train_test_split(pos_df, pos_df['link'], test_size = 0.2, random_state = 9)
    neg_xtrain, neg_xtest, neg_ytrain, neg_ytest = train_test_split(neg_df, neg_df['link'], test_size = 0.2, random_state = 9)
    with open('pos_xtrain.pickle', 'wb') as f:
        pickle.dump(pos_xtrain, f)

    xtrain_frames = [pos_xtrain, neg_xtrain]
    ytrain_frames = [pos_ytrain, neg_ytrain]
    xtrain = pd.concat(xtrain_frames)

    ytrain = pd.concat(ytrain_frames)
    xtest_frames = [pos_xtest, neg_xtest]
    ytest_frames = [pos_ytest, neg_ytest]
    xtest = pd.concat(xtest_frames)

    ytest = pd.concat(ytest_frames)
        
    print("Train Graph Construct")
    train_link = SavedfLink(xtrain)
    test_link = SavedfLink(xtest)
    all_link = train_link + test_link
    
    pos_train_link = SavedfLink(pos_xtrain)
    G = GraphConstruct(pos_train_link, node_num)
    with open('train_graph.pickle', 'wb') as f:
        pickle.dump(G, f) 
    Base = []

    if args.all:
        print("jaccard_distance")
        xtrain['jaccard_distance'] = jaccard_distance(G, train_link)
        xtest['jaccard_distance'] = jaccard_distance(G, test_link)
    
        print("cosine_distance")
        xtrain['cosine_distance'] = cosine_distance(G, train_link)
        xtest['cosine_distance'] = cosine_distance(G, test_link)
        
        print("shortest_path_length")
        xtrain['shortest_path_length'] = shortest_path_length(G, train_link)
        xtest['shortest_path_length'] = shortest_path_length(G, test_link)
        
        print("same_community")
        xtrain['same_community'] = same_community(G, train_link)
        xtest['same_community'] = same_community(G, test_link)
        
        print("common_neighbors")
        xtrain['common_neighbors'] = common_neighbors(G, train_link)
        xtest['common_neighbors'] = common_neighbors(G, test_link)
        
        print("adamic_adar")
        xtrain['adamic_adar'] = adamic_adar(G, train_link)
        xtest['adamic_adar'] = adamic_adar(G, test_link)
        
        print("resource_allocation_index")
        xtrain['resource_allocation_index'] = resource_allocation_index(G, train_link)
        xtest['resource_allocation_index'] = resource_allocation_index(G, test_link)
        
        print("preferential_attachment")
        xtrain['preferential_attachment'] = preferential_attachment(G, train_link)
        xtest['preferential_attachment'] = preferential_attachment(G, test_link)
        
        print("svd")
        xtrain['svd_dot_u'], xtrain['svd_dot_v'] = SVD(G, xtrain)
        xtest['svd_dot_u'], xtest['svd_dot_v'] = SVD(G, xtest)
        
        Base += ['jaccard_distance', 'cosine_distance', 'shortest_path_length', 'same_community', 
                'common_neighbors', 'adamic_adar', 'resource_allocation_index', 'preferential_attachment',
                'svd_dot_u', 'svd_dot_v']
        
    with open('xtrain.pickle', 'wb') as f:
        pickle.dump(xtrain, f)
    with open('xtest.pickle', 'wb') as f:
        pickle.dump(xtest, f) 
        
    if args.n2v:
        print("Node2Vec")
        node2vec = Node2Vec(G, dimensions=16, walk_length=16, num_walks=50)
        model = node2vec.fit(window=7, min_count=1)
        xtrain['node2vec'] = n2v(model, train_link)
        xtest['node2vec'] = n2v(model, test_link)
        Base += ['node2vec']
    
    if args.p_gemsec:
        print("Gemsec")
        filedir = "embedding/GEMSEC_PCA"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed = SaveEmbedding(embedding_data)
        xtrain['Gemsec'] = get_embedding(Embed, train_link)
        xtest['Gemsec'] = get_embedding(Embed, test_link)
        Base += ['Gemsec']
    
    if args.p_gemsecReg:
        print("GemsecReg")
        filedir = "embedding/GEMSECWithRegularization_PCA"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed = SaveEmbedding(embedding_data)
        xtrain['GemsecWithRegularization'] = get_embedding(Embed, train_link)
        xtest['GemsecWithRegularization'] = get_embedding(Embed, test_link)
        Base += ['GemsecWithRegularization']
        
    if args.p_deepwalk:
        print("Deepwalk")
        filedir = "embedding/Deepwalk_PCA"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed = SaveEmbedding(embedding_data)
        xtrain['Deepwalk'] = get_embedding(Embed, train_link)
        xtest['Deepwalk'] = get_embedding(Embed, test_link)
        Base += ['Deepwalk']
        
    if args.p_deepwalkReg:
        print("DeepwalkReg")
        filedir = "embedding/DeepWalkWithRegularization_PCA"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed = SaveEmbedding(embedding_data)
        xtrain['DeepwalkWithRegularization'] = get_embedding(Embed, train_link)
        xtest['DeepwalkWithRegularization'] = get_embedding(Embed, test_link)
        Base += ['DeepwalkWithRegularization']
    
    if args.randomforest:
        print("Random Forest")
        
        param_dist = {"n_estimators":[105,125],
                      "max_depth": [10,15],
                      "min_samples_split": [110,190],
                      "min_samples_leaf": [25,65]}
        
        RDForest = RandomForestClassifier(random_state=25,n_jobs=-1)
        
        rf_random = RandomizedSearchCV(RDForest, param_distributions = param_dist,
                                           n_iter=5,cv=10,scoring='f1',random_state=25)
        rf_random.fit(xtrain[Base], ytrain)
        RDForest = rf_random.best_estimator_
        RDForest.fit(xtrain[Base], ytrain)
        y_train_pred = RDForest.predict(xtrain[Base])
        train_score = f1_score(ytrain, y_train_pred)
        y_test_pred = RDForest.predict(xtest[Base])
        test_score = f1_score(ytest, y_test_pred)
        print("train score is", train_score)
        print("test score is", test_score)
    
        if args.visualize:
            Confusion_Matrix(ytrain, y_train_pred, 'RF', 'Train')
            Confusion_Matrix(ytest, y_test_pred, 'RF', 'Test')
            ROC_Curve(ytrain, y_train_pred, 'RF', 'Train')
            ROC_Curve(ytest, y_test_pred, 'RF', 'Test')
            Importance(RDForest, Base, 'RF')
    
    if args.gradientboost:
        print("Gradient Boost")
        param_dist = {"n_estimators":[100],
                      "max_depth": [5,25,50]}
        
        clf = GradientBoostingClassifier(random_state=25,verbose=1)
        
        rf_random = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=3, scoring='f1',random_state=25)
        rf_random.fit(xtrain[Base], ytrain)
        clf = rf_random.best_estimator_
        clf.fit(xtrain[Base], ytrain)
        y_train_pred = clf.predict(xtrain[Base])
        train_score = f1_score(ytrain, y_train_pred)
        y_test_pred = clf.predict(xtest[Base])
        test_score = f1_score(ytest, y_test_pred)
        print("train score is", train_score)
        print("test score is", test_score)
    
        if args.visualize:
            Confusion_Matrix(ytrain, y_train_pred, 'GB', 'Train')
            Confusion_Matrix(ytest, y_test_pred, 'GB', 'Test')
            ROC_Curve(ytrain, y_train_pred, 'GB', 'Train')
            ROC_Curve(ytest, y_test_pred, 'GB', 'Test')
            Importance(clf, Base, 'GB')