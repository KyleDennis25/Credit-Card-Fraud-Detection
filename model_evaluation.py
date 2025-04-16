from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    # Create metric dictionary
    metric_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1 score": f1_score(y_true, y_pred)
    }
    return metric_dict
'''
This function calculates accuracy, precision, recall, and f1-score of model 
predictions.

Parameters
----------
y_true: 1D NumPy array
Actual y values(actual class labels).

y_pred: 1D NumPy array
Predicted y values(predicted class labels).

Returns
-------
metric_dict: dictionary(4 key-value pairs)
Dictionary containing various accuracy metrics of model predictions.
'''

def stratified_cv(model, X, y, num_splits):
    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=14)
    # Calculate stratified cv accuracies
    cv_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')  
    return cv_scores.mean()
'''
This function calculates average cross validated accuracy of some scikit-learn models.

Parameters
----------
model: object
A scikit-learn model object

X: 2D Pandas dataframe
Predictor variables for model training.

y: 1D Pandas Series
Response variable for model training.

numSplits: int
Number of folds in cross validation

Returns
-------
Scores.mean(): float
Average accuracy from cross validation.
'''

def auc_plot(y_test, y_pred_prob, pr_auc):
    # Calculate precision and recall for PR-AUC curve
    precision, recall, thr = precision_recall_curve(y_test, y_pred_prob, pos_label=1)
    # Plot the PR curve
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    # Save plot to computer
    plt.savefig("pr_curve.png", dpi=300)
    plt.show()

'''
This function plots the roc(receiver operating characteristic) curve of a binary classifier.

Parameters
----------
y_test: 1D Pandas Series
Test set y for model evalutation.

y_pred_prob: 2D NumPy Array
Model classification probabilities.

roc_auc: float
auc value of model.
'''