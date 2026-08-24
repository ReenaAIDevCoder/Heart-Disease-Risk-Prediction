import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("heart_Disease_Model.pkl")
model_features = joblib.load("Model_Features.pkl")
model_scaler = joblib.load("scaler.pkl")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------

st.title("❤️ Heart Disease Risk Prediction")

st.write(
    "Enter the patient's information below to predict "
    "the heart disease risk."
)

# -----------------------------
# User Input
# -----------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=50
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=700,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["N", "Y"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=-5.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# -----------------------------
# Create Input DataFrame
# -----------------------------

if st.button("🔍 Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })


    # -----------------------------
    # Encoding
    # -----------------------------

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # Make columns exactly same as training data
    input_encoded = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(input_encoded)[0]


    # -----------------------------
    # Probability
    # -----------------------------

    probability = model.predict_proba(
        input_encoded
    )[0][1]


    # -----------------------------
    # Display Result
    # -----------------------------

    if prediction == 1:

        st.error("⚠️ Higher Heart Disease Risk")

    else:

        st.success("✅ Lower Heart Disease Risk")


    st.write(
        f"Estimated probability: **{probability * 100:.2f}%**"
    )

    st.info(
        "This prediction is for educational purposes and "
        "is not a medical diagnosis."
    )