# 💼 Loan Approval Prediction using Random Forest

## 📌 Project Overview

This project implements a **Loan Approval Prediction System** using the **Random Forest Machine Learning Algorithm**.

The system predicts whether a loan application will be **Approved** or **Rejected** based on applicant details.

It demonstrates a complete **end-to-end machine learning workflow**, including:

* Model training using dataset
* Batch testing on unseen data
* Saving trained model (`.pkl`)
* Real-time prediction using manual user input

---

## 📂 Folder Structure

```id="a1b2c3"
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
```

---

## 📊 Dataset Description

### 🔹 Training Dataset (`train.csv`)

Used for training the model.

**Input Features:**

* Gender
* Married
* Dependents
* Education
* Self_Employed
* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Loan_Amount_Term
* Credit_History
* Property_Area

**Target Variable:**

* Loan_Status (Approved / Rejected)

**Note:**
`Loan_ID` is removed during training as it is only an identifier.

---

### 🔹 Testing Dataset (`test.csv`)

Used for evaluating the model.

* Same features as training dataset
* `Loan_ID` removed for consistency
* Used for batch prediction

---

## 🧠 Machine Learning Algorithm

### 🌲 Random Forest Classifier

**Why Random Forest?**

* Handles both numerical and categorical data
* Reduces overfitting compared to decision trees
* Works well on tabular datasets
* Provides stable and accurate predictions

---

## 🔄 Project Workflow

### 📌 Step 1: Model Training (`train_model.ipynb`)

1. Load dataset
2. Remove `Loan_ID`
3. Handle missing values (mean/mode)
4. Encode categorical data
5. Split dataset (train/test)
6. Train Random Forest model
7. Evaluate accuracy
8. Save model → `loan_approval_model.pkl`

---

### 📌 Step 2: Batch Testing (`test_model_data.ipynb`)

1. Load trained model
2. Load `test.csv`
3. Remove `Loan_ID`
4. Apply preprocessing
5. Predict for entire dataset
6. Save results → `loan_test_results.csv`

---

### 📌 Step 3: Manual Testing (`test_model_user.ipynb`)

1. Load trained model
2. Take user input
3. Convert to DataFrame
4. Predict result
5. Output: **Approved / Rejected**

---

## 🧪 Testing & Evaluation

### ✅ Batch Testing

* Entire dataset tested in one run
* Results stored in CSV file

### ✅ Manual Testing

* User inputs data manually
* Real-time prediction

### 📌 Summary

* Both methods give consistent results
* Suitable for real-world use cases

---

## 📈 Model Performance

* **Accuracy:** ~78%
* **Metric Used:** Accuracy Score

---

## 📁 Output Files

* **Trained Model:**
  `loan_approval_model.pkl`

* **Prediction Results:**
  `loan_test_results.csv`

---

## ⚙️ How to Run the Project

### 🔹 Install Dependencies

```bash id="d4e5f6"
pip install pandas scikit-learn joblib
```

---

### 🔹 Train the Model

Run:

```
train_model.ipynb
```

---

### 🔹 Test using Dataset

Run:

```
test_model_data.ipynb
```

---

### 🔹 Test using User Input

Run:

```
test_model_user.ipynb
```

---

## 🏦 Applications

* Banking & Financial Institutions
* Loan Eligibility Systems
* Credit Risk Analysis
* Decision Support Systems

---

## 📌 Conclusion

This project demonstrates a complete machine learning pipeline for loan approval prediction.

By combining:

* Batch testing
* Real-time prediction

the system proves both **accuracy** and **practical usability**.

---

## 👨‍💻 Author

**Kanha Patidar**
🎓 B.Tech CSIT (5th Semester)
🏫 Chameli Devi Group of Institutions, Indore

💼 Machine Learning Intern at Technorizen Software Solutions, Indore

---

## ⭐ Support

If you like this project, please ⭐ star the repository and share it!

---

## 📜 License

This project is open-source and free for educational use.
