import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("🔬 Breast Cancer Diagnosis (Random Forest Classifier)")
st.write("Upload a CSV file with breast cancer features for diagnosis.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("✅ Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

    st.subheader("📌 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig_cm)

    st.subheader("📈 Feature Importances")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    ax_imp.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax_imp.invert_yaxis()
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    st.pyplot(fig_imp)

    st.subheader("📃 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Optional: save model
    joblib.dump(model, "rf_breast_cancer_model.pkl")

