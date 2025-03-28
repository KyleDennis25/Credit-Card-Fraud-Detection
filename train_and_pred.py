def train_and_predict(model, X_train, y_train, X_test):
    # fit model
    model.fit(X_train, y_train)
    # Predict probabilities for the test set
    predicted_probs = model.predict_proba(X_test)[:, 1]
    # predict test set
    y_pred = model.predict(X_test)  
    
    return predicted_probs, y_pred
'''
This function trains the given model and makes predictions on the test set.

Parameters
----------
model: object
A scikit-learn model object.

X_train: 2D Pandas dataframe
Training set X for model fitting.

y_train: 1D Pandas Series
Training set y for model fitting.

X_test: 2D Pandas dataframe
Training set X for model evalutation.

Returns
-------
predicted_probs: 1D NumPy Array
Model classification probabilities(of fraud) for the test set.

y_pred: 1D NumPy Array
Model predictions on test set.
'''