from imblearn.over_sampling import SMOTE

def smote_sampling(X, y, sample_proportion):
    # Oversample minority class based on "sample_proportion"(see description below)
    smote = SMOTE(random_state=16, sampling_strategy=sample_proportion)
    # Convert sampled data into new dataframes
    X_sampled, y_sampled = smote.fit_resample(X, y)
    return X_sampled, y_sampled
'''
This function performs SMOTE oversampling on data.

Parameters
----------
X: 2D Pandas dataframe
Predictor variables for model training.

y: 1D Pandas Series
Response variable for model training.

sample_proportion: float
Proportion that the minority class will be, relative to the majority class, after SMOTE oversampling.

Returns
-------
X_sampled: 2D Pandas dataframe
SMOTE sampled predictor variables for model training.

y_sampled: 1D Pandas Series
SMOTE sampled response variables for model training.
'''