# ❤️ Heart Disease Risk Prediction

A Machine Learning project that predicts the risk of heart disease based on patient health and clinical information.

The project includes Exploratory Data Analysis (EDA), data preprocessing, visualization, machine learning model training, model evaluation, and an interactive Streamlit web application for making predictions.

---

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Early risk assessment can help identify patients who may require further medical evaluation.

This project uses the `heart.csv` dataset to build a machine learning classification model that predicts whether a patient is likely to have heart disease.

The trained model is integrated into a Streamlit application where users can enter patient information and receive a prediction along with the estimated probability.

> ⚠️ This project is developed for educational and demonstration purposes only. It is not intended to provide medical diagnosis or replace professional medical advice.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Understand relationships between different health features
- Detect and handle missing values
- Perform data cleaning and preprocessing
- Convert categorical features into numerical features
- Train machine learning classification models
- Evaluate model performance
- Optimize the model using hyperparameter tuning
- Save the trained model using Joblib
- Build an interactive Streamlit application
- Deploy the trained ML model into a user-friendly interface

---

## 📂 Dataset

The project uses a `heart.csv` dataset containing patient health information.

### Main Features

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Sex | Gender of the patient |
| ChestPainType | Type of chest pain |
| RestingBP | Resting blood pressure |
| Cholesterol | Cholesterol level |
| FastingBS | Fasting blood sugar |
| RestingECG | Resting electrocardiogram result |
| MaxHR | Maximum heart rate |
| ExerciseAngina | Exercise-induced angina |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of the peak exercise ST segment |
| HeartDisease | Target variable |

### Target Variable

`HeartDisease`

- `0` → Lower/No heart disease indication
- `1` → Heart disease indication

---

# 🔍 Exploratory Data Analysis

The project performs several EDA operations to understand the dataset.

### EDA includes:

- Dataset shape
- Data types
- Statistical summary
- Missing value analysis
- Duplicate value analysis
- Target variable distribution
- Categorical feature analysis
- Numerical feature distributions
- Outlier visualization
- Correlation analysis

### Visualizations

The project uses:

- Histograms
- Count plots
- Box plots
- Violin plots
- Correlation heatmap

Example analyses include:

- Age distribution
- Resting Blood Pressure distribution
- Cholesterol distribution
- Maximum Heart Rate distribution
- Heart disease distribution by gender
- Heart disease distribution by chest pain type
- Heart disease distribution by fasting blood sugar
- Cholesterol vs Heart Disease
- Age vs Heart Disease

---

# 🧹 Data Preprocessing

The dataset is prepared before machine learning.

### Steps performed:

1. Check missing values
2. Check duplicate records
3. Handle missing values where required
4. Convert categorical variables into numerical variables
5. Apply one-hot encoding using:

```python
pd.get_dummies()
