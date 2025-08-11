# Heart Disease Prediction Project

A machine learning project that predicts heart disease risk based on patient medical data using Logistic Regression. The project includes data analysis, model training, and two different web interfaces for making predictions.

## 🎯 Project Overview

This project uses a heart disease dataset to train a machine learning model that can predict whether a patient has heart disease based on various medical parameters. The trained model is deployed through both Streamlit and Gradio web interfaces for easy interaction.

## 📊 Dataset Features

The model uses 11 medical features to make predictions:

- **Age**: Patient's age in years
- **Sex**: Gender (0 = Female, 1 = Male)
- **Chest Pain Type**: Type of chest pain (1-4)
- **Resting BP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol level (mg/dl)
- **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes)
- **Resting ECG**: Resting electrocardiographic results (0-2)
- **Max Heart Rate**: Maximum heart rate achieved
- **Exercise Angina**: Exercise induced angina (0 = No, 1 = Yes)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST Slope**: Slope of the peak exercise ST segment (1-3)

**Target Variable**: Heart Disease (0 = Normal, 1 = Heart Disease)

## 🗂️ Project Structure

```
Heart Disease/
├── train.ipynb              # Main training notebook with EDA and model development
├── dataset.csv              # Heart disease dataset
├── model.pkl                # Trained Logistic Regression model
├── scaler.pkl               # StandardScaler for feature normalization
├── app.py                   # Gradio web interface
├── main.py                  # Streamlit web interface
├── requirements.txt         # Python dependencies
├── heart_env/               # Virtual environment
├── Detect Heart Disease.pdf # Project documentation
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd "Heart Disease"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv heart_env
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     heart_env\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source heart_env/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Model Training

The machine learning pipeline includes:

1. **Data Loading & Exploration**: Load dataset and perform exploratory data analysis
2. **Data Preprocessing**: Handle outliers and standardize features using StandardScaler
3. **Model Training**: Train Logistic Regression classifier
4. **Model Evaluation**: Assess performance using accuracy, confusion matrix, and classification report
5. **Model Persistence**: Save trained model and scaler as pickle files

To retrain the model, run all cells in `train.ipynb`.

## 🌐 Web Applications

### Option 1: Streamlit Interface

Launch the Streamlit web app:
```bash
streamlit run main.py
```

Features:
- Clean, professional interface
- Input validation with min/max values
- Color-coded results (red for disease, green for normal)
- Real-time predictions

### Option 2: Gradio Interface

Launch the Gradio web app:
```bash
python app.py
```

Features:
- Interactive web UI with dropdowns and number inputs
- Emoji-enhanced predictions
- Easy-to-use interface
- Automatic local hosting

## 🔬 Model Performance

The Logistic Regression model achieves:
- **Algorithm**: Logistic Regression with StandardScaler
- **Features**: 11 medical parameters
- **Training**: 80% of dataset (952 samples)
- **Testing**: 20% of dataset (238 samples)
- **Preprocessing**: StandardScaler for feature normalization

## 📋 Usage Example

### Using Streamlit Interface:
1. Run `streamlit run main.py`
2. Open the provided URL in your browser
3. Input patient data using the sidebar controls
4. Click "Predict" to get the result

### Using Gradio Interface:
1. Run `python app.py`
2. Open the provided URL in your browser
3. Fill in the patient information
4. Click "Submit" to get the prediction

### Sample Input:
- Age: 63
- Sex: Male (1)
- Chest Pain Type: 3
- Resting BP: 145
- Cholesterol: 233
- Fasting Blood Sugar: Yes (1)
- Resting ECG: 0
- Max Heart Rate: 150
- Exercise Angina: No (0)
- Oldpeak: 2.3
- ST Slope: 1

## 🛠️ Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tabulate
streamlit
gradio
```

## 📝 Notes

- The model uses StandardScaler for feature normalization
- Both web interfaces use the same trained model and scaler
- The dataset contains 1,190 patient records
- Missing values have been handled during preprocessing

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## ⚠️ Disclaimer

This project is for educational and research purposes only. The predictions should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical advice.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created as part of an internship project focusing on machine learning applications in healthcare.

---

**Happy Predicting! 🏥💖**
