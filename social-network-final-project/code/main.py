import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import csv
import networkx as nx
import random
import pickle
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

from utils import parse_args
from graph_construct import LoadData, SaveLink, NodeListGen, GraphConstruct, UnconnectedLinkFind, dfGenerate, SavedfLink, SaveEmbedding, SaveHEmbedding
from method import jaccard_distance, cosine_distance, shortest_path_length, same_community, common_neighbors, adamic_adar, resource_allocation_index, preferential_attachment, n2v, n2v_pca, get_embedding, get_n2v_embedding
from method_svd import SVD
from plot import Confusion_Matrix, ROC_Curve, Importance

if __name__ == "__main__":
    args = parse_args()
    random.seed(25)


    if args.gemsec:
        embedding="Gemsec"
    elif args.gemsecReg:
        embedding="GemsecWithRegularization"
    elif args.deepwalk:
        embedding="Deepwalk"
    elif args.deepwalkReg:
        embedding="DeepwalkWithRegularization"
    elif args.n2v:
        embedding="Node2vector"
    elif args.p_gemsec:
        embedding="PCA"
    else:
        embedding="noPCA"

    # 建立data資料夾
    os.makedirs(f'./data/{args.dataset}',exist_ok=True)
    #建立results資料夾
    os.makedirs(f"./results/{args.dataset}/{embedding}/RandomForest", exist_ok=True)
    os.makedirs(f"./results/{args.dataset}/{embedding}/GBDT", exist_ok=True)
    #load data
    filedir = "facebook_clean_data"
    dataset = str(args.dataset)
    train_data = LoadData(filedir, dataset + "_edges")
    
    print("Data loaded")
    link = SaveLink(train_data)
    
    node_list = NodeListGen(link)
    node_num = len(node_list)

    #construct graph
    print("Graph Construct")
    all_G = GraphConstruct(link, node_num)
    
    #find unconnected link
    print("neg+pos")
    if not os.path.isfile(f'./data/{args.dataset}/unconnected_link.pickle'):
        unconnected_link = UnconnectedLinkFind(all_G, node_list, len(link))
        with open(f'./data/{args.dataset}/unconnected_link.pickle', 'wb') as f:
            pickle.dump(unconnected_link,f)
    else:
        with open(f'./data/{args.dataset}/unconnected_link.pickle', 'rb') as f:
            unconnected_link = pickle.load(f)
    
    # print(unconnected_link) 
    #dataframe = pos + neg
    #df = dfGenerate(link, unconnected_link)
    pos_df = dfGenerate(link, True)
    neg_df = dfGenerate(unconnected_link, False)
    
    pos_xtrain, pos_xtest, pos_ytrain, pos_ytest = train_test_split(pos_df, pos_df['link'], test_size = 0.2, random_state = 9)
    neg_xtrain, neg_xtest, neg_ytrain, neg_ytest = train_test_split(neg_df, neg_df['link'], test_size = 0.2, random_state = 9)
    
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

    Base = []

    if args.all:
        if (not os.path.isfile(f'./data/{args.dataset}/xtrain_all.pickle')) or \
            (not os.path.isfile(f'./data/{args.dataset}/xtest_all.pickle')) or \
            (not os.path.isfile(f'./data/{args.dataset}/Base_all.pickle')):
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

            with open(f'./data/{args.dataset}/xtrain_all.pickle', 'wb') as f:
                pickle.dump(xtrain[['jaccard_distance', 'cosine_distance', 'shortest_path_length', 'same_community', 
                    'common_neighbors', 'adamic_adar', 'resource_allocation_index', 'preferential_attachment',
                    'svd_dot_u', 'svd_dot_v']],f)
            with open(f'./data/{args.dataset}/xtest_all.pickle', 'wb') as f:
                pickle.dump(xtest[['jaccard_distance', 'cosine_distance', 'shortest_path_length', 'same_community', 
                    'common_neighbors', 'adamic_adar', 'resource_allocation_index', 'preferential_attachment',
                    'svd_dot_u', 'svd_dot_v']],f)
            with open(f'./data/{args.dataset}/Base_all.pickle', 'wb') as f:
                pickle.dump(['jaccard_distance', 'cosine_distance', 'shortest_path_length', 'same_community', 
                    'common_neighbors', 'adamic_adar', 'resource_allocation_index', 'preferential_attachment',
                    'svd_dot_u', 'svd_dot_v'],f)

        else:
            with open(f'./data/{args.dataset}/xtrain_all.pickle', 'rb') as f:
                xtrain[['jaccard_distance', 'cosine_distance', 'shortest_path_length', 'same_community', 
                    'common_neighbors', 'adamic_adar', 'resource_allocation_index', 'preferential_attachment',
                    'svd_dot_u', 'svd_dot_v']] = pickle.load(f)
            with open(f'./data/{args.dataset}/xtest_all.pickle', 'rb') as f:
                xtest[['jaccard_distance', 'cosine_distance', 'shortest_path_length', 'same_community', 
                    'common_neighbors', 'adamic_adar', 'resource_allocation_index', 'preferential_attachment',
                    'svd_dot_u', 'svd_dot_v']] = pickle.load(f)
            with open(f'./data/{args.dataset}/Base_all.pickle', 'rb') as f:
                Base += pickle.load(f)

    #############################################
    ###             High Dimension            ###
    #############################################
    
    if args.gemsec:
        print("Gemsec")
        filedir = "embedding/GEMSEC"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed_all = SaveHEmbedding(embedding_data, dim = args.dim)
        for i in range(args.dim):
            xtrain[str('Gemsec_'+ str(i))] = get_embedding(Embed_all[i], train_link)
            xtest[str('Gemsec_'+ str(i))] = get_embedding(Embed_all[i], test_link)
            Base += [str('Gemsec_'+ str(i))]

        
    if args.gemsecReg:
        print("GemsecReg")
        filedir = "embedding/GEMSECWithRegularization"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed_all = SaveHEmbedding(embedding_data, dim = args.dim)
        for i in range(args.dim):
            xtrain[str('GemsecReg_'+ str(i))] = get_embedding(Embed_all[i], train_link)
            xtest[str('GemsecReg_'+ str(i))] = get_embedding(Embed_all[i], test_link)
            Base += [str('GemsecReg_'+ str(i))]


    if args.deepwalk:
        print("Deepwalk")
        filedir = "embedding/Deepwalk"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed_all = SaveHEmbedding(embedding_data, dim = args.dim)
        for i in range(args.dim):
            xtrain[str('Deepwalk_'+ str(i))] = get_embedding(Embed_all[i], train_link)
            xtest[str('Deepwalk_'+ str(i))] = get_embedding(Embed_all[i], test_link)
            Base += [str('Deepwalk_'+ str(i))]


    if args.deepwalkReg:
        print("DeepwalkReg")
        filedir = "embedding/DeepWalkWithRegularization"
        embedding_data = LoadData(filedir, dataset + "_embedding")
        Embed_all = SaveHEmbedding(embedding_data, dim = args.dim)
        for i in range(args.dim):
            xtrain[str('DeepwalkReg_'+ str(i))] = get_embedding(Embed_all[i], train_link)
            xtest[str('DeepwalkReg_'+ str(i))] = get_embedding(Embed_all[i], test_link)
            Base += [str('DeepwalkReg_'+ str(i))]


    if args.n2v:
        print("Node2Vec")
        node2vec = Node2Vec(G, dimensions=args.dim, walk_length=16, num_walks=10)
        model = node2vec.fit(window=7, min_count=1)
        Embed_train = n2v(model, train_link)
        Embed_test = n2v(model, test_link)
        for i in range(args.dim):
            xtrain[str('node2vec'+ str(i))] = get_n2v_embedding(Embed_train, train_link, i)
            xtest[str('node2vec'+ str(i))] = get_n2v_embedding(Embed_test, test_link, i)
            Base += [str('node2vec'+ str(i))]


    
    #############################################
    ###        High Dimension With PCA        ###
    #############################################
    if embedding=='PCA':
        if (not os.path.isfile(f'./data/{args.dataset}/xtrain_PCA.pickle')) or \
            (not os.path.isfile(f'./data/{args.dataset}/xtest_PCA.pickle')) or \
            (not os.path.isfile(f'./data/{args.dataset}/Base_PCA.pickle')):
            if args.p_n2v:
                print("Node2Vec(PCA)")
                node2vec = Node2Vec(G, dimensions=16, walk_length=16, num_walks=50)
                model = node2vec.fit(window=7, min_count=1)
                xtrain['node2vec(PCA)'] = n2v_pca(model, train_link)
                xtest['node2vec(PCA)'] = n2v_pca(model, test_link)
                Base += ['node2vec(PCA)']
            
            if args.p_gemsec:
                print("Gemsec(PCA)")
                filedir = "embedding_PCA/GEMSEC_PCA"
                embedding_data = LoadData(filedir, dataset + "_embedding")
                Embed = SaveEmbedding(embedding_data)
                xtrain['Gemsec(PCA)'] = get_embedding(Embed, train_link)
                xtest['Gemsec(PCA)'] = get_embedding(Embed, test_link)
                Base += ['Gemsec(PCA)']
            
            if args.p_gemsecReg:
                print("GemsecReg(PCA)")
                filedir = "embedding_PCA/GEMSECWithRegularization_PCA"
                embedding_data = LoadData(filedir, dataset + "_embedding")
                Embed = SaveEmbedding(embedding_data)
                xtrain['GemsecWithRegularization(PCA)'] = get_embedding(Embed, train_link)
                xtest['GemsecWithRegularization(PCA)'] = get_embedding(Embed, test_link)
                Base += ['GemsecWithRegularization(PCA)']
                
            if args.p_deepwalk:
                print("Deepwalk(PCA)")
                filedir = "embedding_PCA/Deepwalk_PCA"
                embedding_data = LoadData(filedir, dataset + "_embedding")
                Embed = SaveEmbedding(embedding_data)
                xtrain['Deepwalk(PCA)'] = get_embedding(Embed, train_link)
                xtest['Deepwalk(PCA)'] = get_embedding(Embed, test_link)
                Base += ['Deepwalk(PCA)']
                
            if args.p_deepwalkReg:
                print("DeepwalkReg(PCA)")
                filedir = "embedding_PCA/DeepWalkWithRegularization_PCA"
                embedding_data = LoadData(filedir, dataset + "_embedding")
                Embed = SaveEmbedding(embedding_data)
                xtrain['DeepwalkWithRegularization(PCA)'] = get_embedding(Embed, train_link)
                xtest['DeepwalkWithRegularization(PCA)'] = get_embedding(Embed, test_link)
                Base += ['DeepwalkWithRegularization(PCA)']



            with open(f'./data/{args.dataset}/xtrain_PCA.pickle', 'wb') as f:
                pickle.dump(xtrain[['node2vec(PCA)','Gemsec(PCA)','GemsecWithRegularization(PCA)','Deepwalk(PCA)','DeepwalkWithRegularization(PCA)']],f)
            with open(f'./data/{args.dataset}/xtest_PCA.pickle', 'wb') as f:
                pickle.dump(xtest[['node2vec(PCA)','Gemsec(PCA)','GemsecWithRegularization(PCA)','Deepwalk(PCA)','DeepwalkWithRegularization(PCA)']],f)
            with open(f'./data/{args.dataset}/Base_PCA.pickle', 'wb') as f:
                pickle.dump(['node2vec(PCA)','Gemsec(PCA)','GemsecWithRegularization(PCA)','Deepwalk(PCA)','DeepwalkWithRegularization(PCA)'],f)

        else:
            with open(f'./data/{args.dataset}/xtrain_PCA.pickle', 'rb') as f:
                xtrain[['node2vec(PCA)','Gemsec(PCA)','GemsecWithRegularization(PCA)','Deepwalk(PCA)','DeepwalkWithRegularization(PCA)']]= pickle.load(f)
            with open(f'./data/{args.dataset}/xtest_PCA.pickle', 'rb') as f:
                xtest[['node2vec(PCA)','Gemsec(PCA)','GemsecWithRegularization(PCA)','Deepwalk(PCA)','DeepwalkWithRegularization(PCA)']]= pickle.load(f)
            with open(f'./data/{args.dataset}/Base_PCA.pickle', 'rb') as f:
                Base += pickle.load(f)


    
    #############################################
    ###             Learning Model            ###
    #############################################
    
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
            Confusion_Matrix(ytrain, y_train_pred, f"./results/{args.dataset}/{embedding}/RandomForest/", 'Train')
            Confusion_Matrix(ytest, y_test_pred, f"./results/{args.dataset}/{embedding}/RandomForest/", 'Test')
            ROC_Curve(ytrain, y_train_pred, f"./results/{args.dataset}/{embedding}/RandomForest/", 'Train')
            ROC_Curve(ytest, y_test_pred, f"./results/{args.dataset}/{embedding}/RandomForest/", 'Test')
            Importance(RDForest, Base, f"./results/{args.dataset}/{embedding}/RandomForest/")



        #將Train f1 score 寫入 csv　檔案
        if embedding=='PCA' or embedding=='noPCA':
            with open("./results/RandomForest_Train_f1_transposed.csv", "a") as f:
                f.write(str(train_score))

                if embedding=='noPCA':
                    f.write('\n')
                else:
                    f.write(',')
        else:
            with open("./results/RandomForest_Train_f1_transposed.csv", "a") as f:
                
                #如果embedding 為GEMSEC 且 類型為artist,write csv header
                if embedding=="Gemsec" and args.dataset=='artist':
                    f.write('RandomForest_f1,GEMSEC,GEMSECWithRegularization,DeepWalk,DeepWalkWithRegularization,Node2vector,all other feature(with PCA),all other feature(without PCA)\n')
                
                #如果embedding 為GEMSEC write row names
                if embedding=="Gemsec":
                    f.write(args.dataset+',')  
                f.write(str(train_score)+',')


        #將Test f1 score 寫入 csv　檔案
        if embedding=='PCA' or embedding=='noPCA':
            with open("./results/RandomForest_Test_f1_transposed.csv", "a") as f:
                f.write(str(test_score))

                if embedding=='noPCA':
                    f.write('\n')
                else:
                    f.write(',')
        else:
            with open("./results/RandomForest_Test_f1_transposed.csv", "a") as f:
                
                #如果embedding 為GEMSEC 且 類型為artist,write csv header
                if embedding=="Gemsec" and args.dataset=='artist':
                    f.write('RandomForest_f1,GEMSEC,GEMSECWithRegularization,DeepWalk,DeepWalkWithRegularization,Node2vector,all other feature(with PCA),all other feature(without PCA)\n')
                
                #如果embedding 為GEMSEC write row names
                if embedding=="Gemsec":
                    f.write(args.dataset+',')  
                f.write(str(test_score)+',')
                


    if args.gradientboost:
        print("Gradient Boost")
        param_dist = {"n_estimators":[10,30,50],"max_depth": [3,9,11,15,20,35,50]}
        
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
            Confusion_Matrix(ytrain, y_train_pred, f"./results/{args.dataset}/{embedding}/GBDT/", 'Train')
            Confusion_Matrix(ytest, y_test_pred, f"./results/{args.dataset}/{embedding}/GBDT/", 'Test')
            ROC_Curve(ytrain, y_train_pred, f"./results/{args.dataset}/{embedding}/GBDT/", 'Train')
            ROC_Curve(ytest, y_test_pred, f"./results/{args.dataset}/{embedding}/GBDT/", 'Test')
            Importance(clf, Base, f"./results/{args.dataset}/{embedding}/GBDT/")
        
        
        #將Train f1 score 寫入 csv　檔案
        if embedding=='PCA' or embedding=='noPCA':
            with open("./results/GBDT_Train_f1_transposed.csv", "a") as f:
                f.write(str(train_score))

                if embedding=='noPCA':
                    f.write('\n')
                else:
                    f.write(',')
        else:
            with open("./results/GBDT_Train_f1_transposed.csv", "a") as f:
                
                #如果embedding 為GEMSEC 且 類型為artist,write csv header
                if embedding=="Gemsec" and args.dataset=='artist':
                    f.write('GBDT_f1,GEMSEC,GEMSECWithRegularization,DeepWalk,DeepWalkWithRegularization,Node2vector,all other feature(with PCA),all other feature(without PCA)\n')
                
                #如果embedding 為GEMSEC write row names
                if embedding=="Gemsec":
                    f.write(args.dataset+',')  
                f.write(str(train_score)+',')


        #將Test f1 score 寫入 csv　檔案
        if embedding=='PCA' or embedding=='noPCA':
            with open("./results/GBDT_Test_f1_transposed.csv", "a") as f:
                f.write(str(test_score))

                if embedding=='noPCA':
                    f.write('\n')
                else:
                    f.write(',')
        else:
            with open("./results/GBDT_Test_f1_transposed.csv", "a") as f:
                
                #如果embedding 為GEMSEC 且 類型為artist,write csv header
                if embedding=="Gemsec" and args.dataset=='artist':
                    f.write('GBDT_f1,GEMSEC,GEMSECWithRegularization,DeepWalk,DeepWalkWithRegularization,Node2vector,all other feature(with PCA),all other feature(without PCA)\n')
                
                #如果embedding 為GEMSEC write row names
                if embedding=="Gemsec":
                    f.write(args.dataset+',')
                f.write(str(test_score)+',')
                
