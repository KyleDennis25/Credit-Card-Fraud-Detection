from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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