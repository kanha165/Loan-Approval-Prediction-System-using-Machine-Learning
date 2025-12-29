# Loan Approval Prediction using Random Forest

## Project Overview
This project implements a **Loan Approval Prediction System** using the **Random Forest machine learning algorithm**.  
The system predicts whether a loan application will be **Approved** or **Rejected** based on applicant details.

The complete project demonstrates an **end-to-end machine learning workflow**, including:
- Training the model using a training dataset
- Testing the model on an entire test dataset at once (batch testing)
- Saving the trained model in `.pkl` format
- Testing the model using **manual user input**

---

## Folder Structure
random forest/
│
├── train.csv
├── test.csv
├── loan_approval_model.pkl
├── loan_test_results.csv
│
├── train_model.ipynb
├── test_model_data.ipynb
├── test_model_user.ipynb
└── README.md



## Dataset Description

### Training Dataset (`train.csv`)
The training dataset is used to train the Random Forest model.

**Input Features:**
- Gender  
- Married  
- Dependents  
- Education  
- Self_Employed  
- ApplicantIncome  
- CoapplicantIncome  
- LoanAmount  
- Loan_Amount_Term  
- Credit_History  
- Property_Area  

**Target Variable:**
- Loan_Status (Approved / Rejected)

**Important Note:**  
The `Loan_ID` column is removed during training because it is only an identifier and does not contribute to the prediction.

---

### Testing Dataset (`test.csv`)
The test dataset is used to evaluate the trained model on unseen data.

- Contains the same features as the training dataset
- `Loan_ID` column is removed to maintain feature consistency
- Used for batch prediction (entire file tested at once)

---

## Machine Learning Algorithm
**Random Forest Classifier**

### Reason for Selection
- Handles both numerical and categorical data
- Reduces overfitting compared to a single decision tree
- Performs well on tabular datasets
- Provides stable and reliable predictions

---

## Project Workflow

### Step 1: Model Training (`train_model.ipynb`)
1. Load `train.csv`
2. Remove `Loan_ID` column
3. Handle missing values using mean and mode
4. Encode categorical features using Label Encoding
5. Split data into training and testing sets
6. Train the Random Forest model
7. Evaluate model accuracy
8. Save the trained model as `loan_approval_model.pkl`

---

### Step 2: Batch Testing using Test Dataset (`test_model_data.ipynb`)
1. Load the trained model from `loan_approval_model.pkl`
2. Load `test.csv`
3. Remove `Loan_ID` column
4. Apply the same preprocessing steps used during training
5. Predict loan approval status for the **entire test dataset in one run**
6. Save predictions in `loan_test_results.csv`

This step validates the model’s performance on unseen data using batch processing.

---

### Step 3: Manual User Input Testing (`test_model_user.ipynb`)
1. Load the trained model from the `.pkl` file
2. Accept applicant details manually from the user
3. Convert user input into a DataFrame
4. Predict loan approval status
5. Display result as **Approved** or **Rejected**

This step demonstrates real-time prediction using user-provided inputs.

---

## Testing and Evaluation

### Batch Testing
- The complete `test.csv` file is tested in a single execution.
- Predictions for all loan applications are generated at once.
- Results are stored in `loan_test_results.csv`.

### Manual Testing
- Users can manually enter applicant details.
- The model predicts loan approval status in real time.

### Testing Summary
- Batch testing was performed using the entire test dataset.
- Manual testing was performed using user input.
- Both testing methods produced consistent and reliable results.

---

## Model Performance
- Accuracy achieved: **~78%**
- Evaluation Metric: Accuracy Score

---

## Output Files
- **Trained Model:**  
loan_approval_model.pkl


- **Prediction Results:**  
loan_test_results.csv



## How to Run the Project

### Install Required Libraries
```bash
pip install pandas scikit-learn joblib
Train the Model
Run:


train_model.ipynb
Test Using Test Dataset
Run:


test_model_data.ipynb
Test Using Manual User Input
Run:


test_model_user.ipynb
Applications
Banking and Financial Institutions

Loan Eligibility Checking Systems

Credit Risk Analysis

Decision Support Systems

Conclusion
This project demonstrates a complete machine learning pipeline for loan approval prediction.
By combining batch testing and manual user input testing, the system proves both accuracy and real-world usability.

Author
Kanha Patidar
B.Tech (CSIT)
Chamelidevi Group of Institutions, Indore
