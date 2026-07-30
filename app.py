import streamlit as st
import pandas as pd
import joblib


# -------------------------------
# Load Pipeline
# -------------------------------

pipeline = joblib.load(
    "models/churn_pipeline.pkl"
)


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)


st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether the customer will churn."
)


# -------------------------------
# User Inputs
# -------------------------------

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)


tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)


MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=120.0,
    value=70.0
)


TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)


Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


MultipleLines = st.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


InternetService = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


OnlineSecurity = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


OnlineBackup = st.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


DeviceProtection = st.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


TechSupport = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


StreamingTV = st.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


StreamingMovies = st.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


PaperlessBilling = st.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)


PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        {
            "SeniorCitizen": [SeniorCitizen],
            "tenure": [tenure],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges],

            "gender": [gender],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines],

            "InternetService": [InternetService],

            "OnlineSecurity": [OnlineSecurity],
            "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection],

            "TechSupport": [TechSupport],

            "StreamingTV": [StreamingTV],
            "StreamingMovies": [StreamingMovies],

            "Contract": [Contract],

            "PaperlessBilling": [PaperlessBilling],

            "PaymentMethod": [PaymentMethod]
        }
    )


    prediction = pipeline.predict(
        input_data
    )


    probability = pipeline.predict_proba(
        input_data
    )


    st.subheader("Prediction Result")


    if prediction[0] == 1:
        st.error(
            "⚠️ Customer is likely to churn"
        )

    else:
        st.success(
            "✅ Customer is likely to stay"
        )


    st.write(
        f"Churn Probability: {probability[0][1]*100:.2f}%"
    )

    st.write(
        f"Stay Probability: {probability[0][0]*100:.2f}%"
    )
