import streamlit as st
import pandas as pd
import joblib
from utils.shap_explainer import explain_prediction_with_shap
from utils.llm_integration import generate_llm_response
import json

# Load the saved model, scaler, and feature names
scaler = joblib.load(r"C:\Users\rbham\Desktop\NeuroAid\model\scaler.joblib")
stacked_model = joblib.load(r"C:\Users\rbham\Desktop\NeuroAid\model\stacked_model.pkl")
with open(r"C:\Users\rbham\Desktop\NeuroAid\model\feature_names.json", "r") as f:
    feature_names = json.load(f)

# Load the SHAP explainer (ensure the path is correct)
explainer_path = r"C:\Users\rbham\Desktop\NeuroAid\model\explainer.pkl"

# Streamlit app interface
st.title("NeuroAid 🧠")
st.write("Enter your health details to assess Alzheimer's risk and receive personalized advice.")

# Create input fields dynamically based on feature names
patient_data = {}
for feature in feature_names:
    patient_data[feature] = st.number_input(f"Enter {feature}:", min_value=0.0, step=0.1)

# When user clicks "Submit"
if st.button("Submit"):
    try:
        # Convert patient data to DataFrame
        patient_df = pd.DataFrame([patient_data], columns=feature_names)
        
        # Scale the input data using the saved scaler
        patient_scaled = scaler.transform(patient_df)
        
        # Make prediction using the stacked model
        prediction = stacked_model.predict(patient_scaled)[0]
        prediction_proba = stacked_model.predict_proba(patient_scaled)[0]
        
        # Display prediction results to the user
        st.subheader("Prediction Results")
        if prediction == 1:
            st.write(f"The model predicts a high risk of Alzheimer's disease (Confidence: {prediction_proba[1]:.1%}).")
        else:
            st.write(f"The model predicts a low risk of Alzheimer's disease (Confidence: {prediction_proba[0]:.1%}).")
        
        # Explain prediction using SHAP and generate LLM response
        explanation_text = explain_prediction_with_shap(patient_data, stacked_model, scaler, feature_names)
        advice = generate_llm_response(explanation_text, prediction, prediction_proba)
        
        # Display explanation and advice
        st.subheader("Explanation & Advice")
        st.write(advice)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
