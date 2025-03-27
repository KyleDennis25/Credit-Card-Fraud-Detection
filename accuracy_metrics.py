from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    # Create metric dictionary
    metric_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision(y=1)": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1-score": f1_score(y_true, y_pred, average="weighted")
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