import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="File name of dataset.",
        default="tvshow",
    )
    
    parser.add_argument("-a", "--all", action='store_true')
    parser.add_argument("-n", "--n2v", action='store_true')
    parser.add_argument("-s", "--svd", action='store_true')
    
    parser.add_argument("-p", "--p_gemsec", action='store_true')
    parser.add_argument("-m", "--p_gemsecReg", action='store_true')
    parser.add_argument("-d", "--p_deepwalk", action='store_true')
    parser.add_argument("-w", "--p_deepwalkReg", action='store_true')
    
    parser.add_argument("-v", "--visualize", action='store_true')
    parser.add_argument("-r", "--randomforest", action='store_true')
    parser.add_argument("-g", "--gradientboost", action='store_true')

    args = parser.parse_args()
    return args
