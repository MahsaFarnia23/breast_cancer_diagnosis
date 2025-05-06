# Breast Cancer Diagnosis

A machine learning model to classify breast cancer tumors as benign or malignant using Random Forest.

## Files
- `model_rf.py`: Main Python script for data loading, preprocessing, model training and evaluation.
- `data.csv`: Dataset (from Kaggle or UCI link provided).
- `breast_cancer_preprocessing_summary.md`: Step-by-step preprocessing summary.

## Dataset Source
[https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Run

```bash
python model_rf_final.py
```

## Sample Output

```
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

