
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Heart Disease Prediction App")
st.write("Provide the patient's details to predict the risk of heart disease.")

# User Input Form
with st.form("heart_form"):
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ("Female", "Male"))
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
    exang = st.radio("Exercise Induced Angina?", ["No", "Yes"])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

    submit = st.form_submit_button("Predict")

if submit:
    sex = 1 if sex == "Male" else 0
    cp_dict = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
    fbs = 1 if fbs == "Yes" else 0
    restecg_dict = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
    exang = 1 if exang == "Yes" else 0
    slope_dict = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}

    input_data = pd.DataFrame([
        [age, sex, cp_dict[cp], trestbps, chol, fbs, restecg_dict[restecg], thalach, exang, oldpeak, slope_dict[slope]]
    ], columns=["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar",
                "resting ecg", "max heart rate", "exercise angina", "oldpeak", "ST slope"])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease with probability: {probability:.2f}")
    else:
        st.success(f"✅ No Heart Disease Detected. Probability: {1 - probability:.2f}")
