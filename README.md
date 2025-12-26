# Loan Approval Prediction System

## 📌 Project Description
The Loan Approval Prediction System is a machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details.  
The system uses a **Random Forest Classifier** trained on a real-world loan dataset.

---

## 🎯 Objective
To automate the loan approval process by analyzing applicant information and predicting loan eligibility accurately and efficiently.

---

## 📊 Dataset
- Source: Kaggle Loan Prediction Dataset
- Files:
  - `train.csv` – used for training the model
  - `test.csv` – used for testing the trained model
- Note:
  - The `Loan_ID` column is removed as it is only an identifier and does not affect predictions.

---

## 🧾 Features Used
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

---

## 🎯 Target Variable
- Loan_Status  
  - Approved  
  - Rejected  

---

## 🤖 Machine Learning Algorithm
- Random Forest Classifier

**Reason for selection:**
- Reduces overfitting
- Works well with tabular data
- Provides good accuracy

---

## ⚙️ Project Workflow
1. Load training dataset  
2. Remove Loan_ID column  
3. Handle missing values  
4. Encode categorical features  
5. Split data into training and testing sets  
6. Train Random Forest model  
7. Evaluate model accuracy  
8. Save trained model as `.pkl` file  
9. Load model and predict results on test data  

---

## 📈 Model Performance
- Accuracy achieved: ~78%
- Evaluation metric: Accuracy Score

---

## 📁 Project Structure

loan-approval-prediction/
│
├── train.csv
├── test.csv
├── loan_approval_model.pkl
├── loan_test_results.csv
├── train_model.ipynb
├── test_model.ipynb
└── README.md


---

## ▶️ How to Run the Project

### Step 1: Install Required Libraries
pip install pandas scikit-learn joblib

markdown
Copy code

### Step 2: Train the Model
- Run the training notebook/script
- The trained model will be saved as:
loan_approval_model.pkl

markdown
Copy code

### Step 3: Test the Model
- Run the testing notebook/script
- Predictions will be saved as:
loan_test_results.csv

yaml
Copy code

---

## 📤 Output
The output file contains predicted loan status:
Loan_Status_Predicted
Approved / Rejected

yaml
Copy code

---

## 🏦 Applications
- Banking systems
- Loan eligibility verification
- Credit risk analysis

---

## ✅ Conclusion
This project demonstrates the use of machine learning for automating loan approval decisions.  
The Random Forest model provides reliable predictions and reduces manual processing effort.

---

**Author**
Developed by **Kanha Patidar**

Branch: B.Tech CSIT

Semester: 5th Sem

College: Chameli Devi Group of Institutions, Indore


Machine Learning inten at technorizen software solution. indore 


