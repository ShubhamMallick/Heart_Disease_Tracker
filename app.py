import gradio as gr
import numpy as np
import pickle

# Load model and scaler separately
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_heart_disease(age, sex, chest_pain, resting_bp, cholesterol,
                          fasting_bs, resting_ecg, max_hr, exercise_angina,
                          oldpeak, st_slope):
    input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                            fasting_bs, resting_ecg, max_hr, exercise_angina,
                            oldpeak, st_slope]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    return "Heart Disease detected ❤️‍🔥" if prediction[0] == 1 else "Normal 💚"

# Define Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=63),
        gr.Dropdown(choices=[0,1], label="Sex (0=Female,1=Male)", value=1),
        gr.Dropdown(choices=[1,2,3,4], label="Chest Pain Type", value=3),
        gr.Number(label="Resting Blood Pressure", value=145),
        gr.Number(label="Cholesterol", value=233),
        gr.Dropdown(choices=[0,1], label="Fasting Blood Sugar > 120 mg/dl", value=1),
        gr.Dropdown(choices=[0,1,2], label="Resting ECG", value=0),
        gr.Number(label="Max Heart Rate", value=150),
        gr.Dropdown(choices=[0,1], label="Exercise Induced Angina", value=0),
        gr.Number(label="Oldpeak (ST depression)", value=2.3),
        gr.Dropdown(choices=[1,2,3], label="ST Slope", value=1),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Disease Predictor",
    description="Input patient data to predict heart disease risk.",
    live=False,
)


if __name__ == "__main__":
    iface.launch()
