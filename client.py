import streamlit as st
import requests
import json

st.title('Machine Learning Web App')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
delimiter = st.selectbox("Select delimiter", [",", " "])
has_header = st.selectbox("Has header?", ["yes", "no"])

if st.button('Upload'):
    if uploaded_file is not None:
        files = {'file': uploaded_file}
        data = {
            'delimiter': delimiter,
            'has_header': has_header
        }
        response = requests.post("http://127.0.0.1:5000/upload", files=files, data=data)
        if response.status_code == 200:
            st.success("File uploaded successfully")
            st.json(response.json())
        else:
            st.error("Failed to upload file")
            st.json(response.json())

model_type = st.selectbox('Select model type', ['Classification', 'Regression'])

if model_type == 'Classification':
    model_name = st.selectbox('Select model', ['LR', 'LDA', 'Decision Tree', 'KNN', 'SVM', 'Random Forest', 'AdaBoost'])
elif model_type == 'Regression':
    model_name = st.selectbox('Select model', ['LR', 'Ridge', 'Lasso', 'Elastic', 'KNN', 'Tree', 'SVR'])

if st.button('Train Model'):
    if model_name:
        response = requests.post(
            "http://127.0.0.1:5000/train",
            json={"model_type": model_type, "model_name": model_name}
        )
        if response.status_code == 200:
            st.success("Model trained successfully")
            metrics = response.json()
            st.json(metrics)
        else:
            st.error("Failed to train model")
            st.json(response.json())
    else:
        st.warning("Please select a model")
