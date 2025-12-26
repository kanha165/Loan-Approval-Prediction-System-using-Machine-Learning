# Loan-Approval-Prediction-System-using-Machine-Learning
Loan Approval Prediction System
Project Overview

This project is a Machine Learning–based Loan Approval Prediction System that predicts whether a loan application will be Approved or Rejected based on applicant details.
The model is trained using the Random Forest Classifier on a real-world loan dataset.

Objective

To automate the loan approval process by analyzing applicant information and predicting loan eligibility accurately.

Dataset

Source: Kaggle (Loan Prediction Dataset)

Files used:

train.csv – used for training the model

test.csv – used for testing and prediction

Note:
The Loan_ID column is removed during training and testing because it is only an identifier and does not contribute to prediction.

Features Used

Gender

Married

Dependents

Education

Self_Employed

ApplicantIncome

CoapplicantIncome

LoanAmount

Loan_Amount_Term

Credit_History

Property_Area

Target Variable

Loan_Status

Approved

Rejected

Machine Learning Algorithm

Random Forest Classifier

Chosen because it:

Reduces overfitting

Handles both numerical and categorical data

Provides good accuracy on tabular datasets

Project Workflow

Load training dataset

Remove Loan_ID column

Handle missing values

Encode categorical variables

Split data into training and testing sets

Train Random Forest model

Evaluate model accuracy

Save trained model as .pkl file

Load model and predict results on test data

Model Performance

Achieved accuracy: ~78%

Evaluation metric: Accuracy Score

Files in the Project
loan-approval-prediction/
│
├── train.csv
├── test.csv
├── loan_approval_model.pkl
├── loan_test_results.csv
├── training.ipynb
├── testing.ipynb
└── README.md

How to Run the Project
1. Install Required Libraries
pip install pandas scikit-learn joblib

2. Train the Model

Run the training notebook/script

The trained model will be saved as:

loan_approval_model.pkl

3. Test the Model

Run the testing notebook/script

Predictions will be saved as:

loan_test_results.csv

Output

The final output file contains predicted loan status:

Loan_Status_Predicted
Approved / Rejected

Applications

Banking and Financial Institutions

Loan Eligibility Systems

Credit Risk Analysis

Conclusion

This project demonstrates how machine learning can be used to automate loan approval decisions. By using a Random Forest model and proper data preprocessing, the system provides reliable predictions and reduces manual effort.

Author

Kanha Patidar
B.Tech (CSIT)
Chamelidevi Group of Institutions, Indore
