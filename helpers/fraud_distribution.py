def fraud_distr(y):
    # find how many fraud transactions are in y
    fraud_count = (y == 1).sum()
    # find percentage
    fraud_percentage = (fraud_count/len(y))*100 
    fraud_percentage_rounded = round(fraud_percentage, 4)
    return fraud_count, fraud_percentage_rounded
'''
This function calculates the fraud distribution of y.

Parameters
----------
y: 1D Pandas Series
Response variable of dataset.

Returns
-------
fraud_count: int
Total number of fraudulent transactions in y.

fraud_percentage_rounded: float
Rounded percentage of fraudulent transactions in y.
'''