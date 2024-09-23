import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
        default="tvshow",
    )
    
    parser.add_argument("-a", "--all", help="noPCA",action='store_true')
    
    parser.add_argument("-e", "--gemsec", help="Gemsec", action='store_true')
    parser.add_argument("-s", "--gemsecReg", help="GemsecWithRegularization", action='store_true')
    parser.add_argument("-l", "--deepwalk", help="Deepwalk", action='store_true')
    parser.add_argument("-k", "--deepwalkReg",  help="DeepwalkWithRegularization", action='store_true')
    parser.add_argument("-x", "--n2v",  help="Node2vector", action='store_true')
    
    parser.add_argument("--dim", help="Dimension of High Dim Embedding", type=float, default=16)
    
    parser.add_argument("-p", "--p_gemsec", help="Gemsec with PCA", action='store_true')
    parser.add_argument("-m", "--p_gemsecReg", help="GemsecWithRegularization with PCA", action='store_true')
    parser.add_argument("-d", "--p_deepwalk", help="Deepwalk with PCA", action='store_true')
    parser.add_argument("-w", "--p_deepwalkReg",  help="DeepwalkWithRegularization with PCA", action='store_true')
    parser.add_argument("-n", "--p_n2v", help="node2vec with PCA",action='store_true')
    
    parser.add_argument("-v", "--visualize", action='store_true')
    parser.add_argument("-r", "--randomforest", action='store_true')
    parser.add_argument("-g", "--gradientboost", action='store_true')

    args = parser.parse_args()
    return args
