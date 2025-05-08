# Breast Cancer Diagnosis

A machine learning model to classify breast cancer tumors as benign or malignant using Random Forest.

## 🔍 Overview
This project applies a Random Forest Classifier on the Breast Cancer Wisconsin (Diagnostic) Dataset to predict if the tumor is malignant or benign.

## 📂 Files
- `model_rf_final.py`: Main Python script for data loading, preprocessing, model training, and evaluation.
- `data.csv`: Dataset (from Kaggle or UCI link provided).
- `breast_cancer_preprocessing_summary.md`: Step-by-step preprocessing summary.

## 📊 Dataset Source
[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## 📦 Requirements
```bash
- Python 3.8+
- Libraries:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
    - joblib
```
## 🚀 Run
```bash
python model_rf.py
```
## ✅ Sample Output
```bash
Accuracy: 0.96

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.98      0.97        71
           1       0.96      0.92      0.94        43

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

Confusion Matrix:
[[70  1]
 [ 3 40]]
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
