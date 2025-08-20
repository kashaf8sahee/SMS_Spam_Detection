# 📩 SMS Spam Detection with Machine Learning

A Python-based SMS Spam Detection system using machine learning (**SVM**, **Naive Bayes**, or **Logistic Regression**). It uses **TF-IDF** vectorization to process messages and evaluates performance using **accuracy**, **confusion matrix**, **classification report**, and **ROC-AUC curve**.

---

## 📂 Dataset

- Dataset used: **SMS Spam Collection**
- File format: CSV (`spam.csv`)

---

## ⚙️ Features

- Text preprocessing & cleaning
- TF-IDF vectorization with bi-grams
- Model options:
  - `svm` (default & recommended)
  - `nb` (Naive Bayes baseline)
  - `logreg` (Logistic Regression)
- Visual evaluation: Confusion Matrix & ROC Curve
- Predict function for custom SMS input

---

## 🚀 How to Run

1. **Set your CSV file path** in the script:
```python
file_path = r"E:\SMS_Spam_Detection\spam.csv"

MODEL_CHOICE = 'svm'  # Options: 'svm', 'nb', 'logreg'

## Run the script to:

Load and preprocess data

Train model

Evaluate results with confusion matrix and ROC curve

Predict on test data

## Sample Evaluation Output

Accuracy

Confusion Matrix (Visualized)

Classification Report

ROC Curve & AUC
