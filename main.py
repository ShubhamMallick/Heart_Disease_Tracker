import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Heart Disease Prediction")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=63)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=145)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=233)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2])
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=2.3, format="%.1f")
st_slope = st.selectbox("ST Slope", options=[1, 2, 3])

# Prepare input for model
input_features = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                            fasting_bs, resting_ecg, max_hr, exercise_angina,
                            oldpeak, st_slope]])

# Predict button
if st.button("Predict"):
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Prediction: Heart Disease detected")
    else:
        st.success("Prediction: Normal (No Heart Disease)")

