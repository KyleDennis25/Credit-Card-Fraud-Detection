# Evaluating the Performance of Machine Learning Models for Credit Card Fraud Detection Using SMOTE

This is my senior capstone project from Quinnipiac. In this project, I evaluate the performance of several machine learning models when used to predict credit card fraud. I used SMOTE oversampling to address the dataset imbalance, and investigated how this affects model performance. 

# File Descriptions
**Report.docx**- Complete project report, in which I discuss the background to this problem, and I evaluate and interpret the results (and their real-world implications).

**fraud_distribution.py**- Python file containing a helper function that calculates the distribution of fraudulent credit card transactions in a given 'y' vector.

**model_evaluation.py**- Python file containing helper functions for different model evaluation tasks, such as computing accuracy metrics, and plotting Precision-Recall Curves.

**project.ipynb**- Jupyter notebook containing all project steps, from data cleaning to model evaluation.

**smote.py**- Python file containing a helper function for performing SMOTE oversampling on a given training set.

**train_and_pred.py**- Python file containing a helper function that trains a given model, and makes predictions on a given test set.

# Dataset
The dataset used is from Kaggle- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
