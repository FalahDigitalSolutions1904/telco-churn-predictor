import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Risk Dashboard")
st.write("Adjust the customer metrics below to calculate real-time churn probability.")

# 2. Load the Saved ML Artifacts
@st.cache_resource
def load_artifacts():
    with open('src/xgb_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/model_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

try:
    model, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model artifacts not found. Please run your training pipeline script first to save the pickle files in 'src/'.")
    st.stop()

# 3. Sidebar / UI Inputs for User Data
st.subheader("👤 Customer Profile Input")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security Service", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support Service", ["Yes", "No", "No internet service"])

# Hardcoded defaults for features we aren't tweaking in this basic UI to match model input structure
# (In production, you'd add sliders/toggles for all of them)
is_family = st.checkbox("Has Family Setup (Partner & Dependents)", value=False)

# 4. Construct the Input Dataframe matching the training structure exactly
input_data = pd.DataFrame(0.0, index=[0], columns=model_columns)

# Assign numeric inputs
input_data['tenure'] = float(tenure)
input_data['MonthlyCharges'] = float(monthly_charges)
input_data['TotalCharges'] = float(tenure * monthly_charges) # Approximation
input_data['Is_Family'] = float(is_family)

# Map our inputs to One-Hot Encoded structure columns
if f"Contract_{contract}" in input_data.columns:
    input_data[f"Contract_{contract}"] = 1.0
if f"InternetService_{internet_service}" in input_data.columns:
    input_data[f"InternetService_{internet_service}"] = 1.0
if f"OnlineSecurity_{online_security}" in input_data.columns:
    input_data[f"OnlineSecurity_{online_security}"] = 1.0
if f"TechSupport_{tech_support}" in input_data.columns:
    input_data[f"TechSupport_{tech_support}"] = 1.0

# 5. Make Predictions
st.markdown("---")
if st.button("🚀 Calculate Churn Risk Score"):
    # Extract prediction probabilities [Probability of staying, Probability of leaving]
    probabilities = model.predict_proba(input_data)[0]
    churn_probability = probabilities[1] * 100
    
    # Display results dynamically based on risk level
    st.subheader("🎯 Risk Assessment Result")
    
    if churn_probability < 30:
        st.success(f"Low Risk Customer: **{churn_probability:.1f}% Churn Probability**")
    elif 30 <= churn_probability < 70:
        st.warning(f"Medium Risk Customer: **{churn_probability:.1f}% Churn Probability**")
    else:
        st.error(f"🚨 High Risk Customer: **{churn_probability:.1f}% Churn Probability**")
        st.write("💡 *Action Plan: Suggest transitioning this customer away from Month-to-Month contracts or offering a retention discount.*")