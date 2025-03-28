from sklearn.metrics import roc_curve, roc_auc_score 
import matplotlib.pyplot as plt

def roc_plot(y_test, y_pred_prob, roc_auc):
    # Calculate the false positive rate(fpr) and true positive rate(tpr) for different classification thresholds 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    # Plot the ROC curve 
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    # roc curve for tpr = fpr  
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve') 
    plt.legend(loc="lower right") 
    plt.show()
'''
This function plots the auc(receiver operating characteristic) curve of a classifier.

Parameters
----------
y_test: 1D Pandas Series
Test set y for model evalutation.

y_pred_prob: 2D NumPy Array
Model classification probabilities.

roc_auc: float
auc value of model.
'''