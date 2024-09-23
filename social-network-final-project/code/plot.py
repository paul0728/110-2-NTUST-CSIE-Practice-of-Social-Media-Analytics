import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def Confusion_Matrix(test_y, predict_y, save_dir, split):
    C = confusion_matrix(test_y, predict_y)
    
    A =(((C.T)/(C.sum(axis=1))).T)
    
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(20,5))
    
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
    
    plt.savefig(f"{save_dir}{split}_confusion_matrix.png")
    
def ROC_Curve(test_y, predict_y, save_dir, split):
    fpr,tpr,ths = roc_curve(test_y,predict_y)
    auc_sc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic with test data')
    plt.legend()
    plt.savefig(f"{save_dir}{split}_roc_curve.png")
    
def Importance(clf, features, save_dir):
    importances = clf.feature_importances_
    indices = (np.argsort(importances))[-25:]
    plt.figure(figsize=(10,12))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='r', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f"{save_dir}importance.png")