
# 🧬 Breast Cancer Diagnosis Project – Data Preprocessing Summary

This document summarizes all preprocessing steps applied to the Breast Cancer Wisconsin dataset in preparation for machine learning.

---

## ✅ Step 1: Load the Data
```python
df = pd.read_csv("data.csv")
```
- Loaded the CSV file into a pandas DataFrame.
- The dataset includes features of cell nuclei derived from digitized images of fine needle aspirates of breast masses.

---

## ✅ Step 2: Initial Exploration
```python
df.head()
df.shape
df.info()
```
- Viewed the first few rows and dataset dimensions.
- Confirmed 569 samples (rows) and 32 columns (features + ID + diagnosis).
- Most columns are `float64`, except for `id` and `diagnosis`.

---

## ✅ Step 3: Missing Values & Class Distribution
```python
df.isnull().sum()
df['diagnosis'].value_counts()
df.dtypes
```
- No missing values found.
- Diagnosis values: `M` (Malignant), `B` (Benign).

---

## ✅ Step 4: Drop ID and Encode Diagnosis
```python
df = df.drop(columns=['id'])
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```
- Dropped the non-informative `id` column.
- Converted `diagnosis` from text to numeric: `1 = Malignant`, `0 = Benign`.

---

## ✅ Result So Far:
- Clean dataset with **30 numeric features** and **1 binary target column**.
- No missing values.
- Ready for feature-target separation and machine learning model training.

---

## ✅ Step 5: Feature-Target Split and Train/Test Split

Now that the data is clean and the target is encoded, we separate the features (`X`) from the target (`y`), and split the dataset for training and testing.

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the shape of each split
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)
```
- `X`: Contains the 30 numeric tumor features.
- `y`: Contains binary labels (0 = benign, 1 = malignant).
- Data is split into 80% for training and 20% for evaluation.

---

## ✅ Next Step:
We are ready to build a machine learning model (e.g., logistic regression or random forest) to predict tumor diagnosis based on these features.
