# Evaluating the Performance of Machine Learning Models for Credit Card Fraud Detection Using SMOTE

This project was completed as part of my senior capstone at Quinnipiac University. I evaluated the performance of several machine learning models when used to predict credit card fraud using real transaction data. To address the dataset’s severe class imbalance, I applied SMOTE oversampling and analyzed how this affected model performance.

This project highlights the critical trade-offs between metrics like precision, recall, and F1 score when dealing with imbalanced data. By experimenting with different SMOTE samples, I demonstrated how this sampling technique significantly impacts a model’s effectiveness — a key insight for deploying machine learning models in high-stakes, real-world scenarios like fraud detection. 

## File Descriptions
**Report.docx**- Complete project report, in which I discuss the background to the problem and evaluate and interpret the results, including their real-world implications.

**fraud_distribution.py**- Python file containing a helper function that calculates the distribution of fraudulent credit card transactions in a given 'y' vector.

**model_evaluation.py**- Python file containing helper functions for different model evaluation tasks, such as computing accuracy metrics, and plotting Precision-Recall Curves.

**project.ipynb**- Jupyter notebook containing all project steps, from data cleaning to model evaluation.

**smote.py**- Python file containing a helper function for performing SMOTE oversampling on a given training set.

**train_and_pred.py**- Python file containing a helper function that trains a given model, and makes predictions on a given test set.

## Dataset
The dataset used is from Kaggle- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
