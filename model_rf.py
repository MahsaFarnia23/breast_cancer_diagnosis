# Breast Cancer Diagnosis using Random Forest
# Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set
# Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from lime import lime_tabular
import matplotlib.pyplot as plt
import joblib

def main():
    # Load the dataset
    df = pd.read_csv('data.csv')

    # Preprocessing
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Split features and labels
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Save the model (optional)
    joblib.dump(model, "rf_breast_cancer_model.pkl")

    # 🔍 Explain a prediction using LIME
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=X.columns.tolist(),
        class_names=['Benign', 'Malignant'],
        mode='classification',
        discretize_continuous=False
    )

    i = 0  # Index of the test instance to explain
    exp = explainer.explain_instance(
        X_test_scaled[i],
        model.predict_proba,
        num_features=10
    )

    # Save to HTML
    html = exp.as_html()
    with open("lime_explanation.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("LIME explanation saved as lime_explanation.html")

    # Print top features
    print("\nTop features contributing to this prediction:")
    for feature, weight in exp.as_list():
        print(f"{feature:50} {weight:+.4f}")

if __name__ == "__main__":
    main()
