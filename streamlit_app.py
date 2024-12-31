# 11890090
# 魏宏恩 Gary

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import joblib

def load_dataset(uploaded_file=None):
    if uploaded_file:
        data = pd.read_csv(uploaded_file, delimiter=';')
    else:
        data = pd.read_csv('winequality-white.csv', delimiter=';')
    return data


def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'report': classification_report(y_test, y_pred, zero_division=0)
        }
    
    return results, scaler


st.title("White Wine Quality Classification")
    
uploaded_file = st.sidebar.file_uploader("Upload Wine Quality Dataset (CSV)", type="csv")
data = load_dataset(uploaded_file)

st.write("### Dataset Overview")
st.write(data.head())
    
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results, scaler = train_models(X_train, X_test, y_train, y_test)

st.write("### Model Evaluation Results")
eval_df = pd.DataFrame({
    name: {
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'F1 Score': res['f1']
    } for name, res in results.items()
}).T
st.write(eval_df)

st.write("### Classification Reports")
for name, res in results.items():
    st.write(f"#### {name}")
    st.text(res['report'])

st.write("### Predict Wine Quality")
st.write("Enter wine characteristics (comma-separated values):")
user_input = st.text_input("Format: fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol")

if st.button("Predict"):
    try:
        input_array = np.array([float(x.strip()) for x in user_input.split(',')]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        predictions = {name: model['model'].predict(input_scaled)[0] 
                     for name, model in results.items()}
        
        st.write("### Predictions")
        for model_name, prediction in predictions.items():
            st.write(f"{model_name}: {prediction}")
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
