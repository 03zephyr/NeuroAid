import streamlit as st
import pandas as pd
import joblib
from utils.shap_explainer import explain_prediction_with_shap
from utils.llm_integration import generate_llm_response
import json

# Load the saved model, scaler, and feature names
scaler = joblib.load("model/scaler.joblib")
stacked_model = joblib.load("model/stacked_model.pkl")
with open("model/feature_names.json", "r") as f:
    feature_names = json.load(f)

# Load the SHAP explainer (ensure the path is correct)
explainer_path = "model/explainer.pkl"


# Streamlit app interface
st.title("NeuroAid 🧠")
st.write("Enter your health details to assess Alzheimer's risk and receive personalized advice.")

st.text("")
# Create input fields dynamically based on feature names
patient_data = {}

# MMSE Section


# Memory Complaints Section
memory_complaints = st.radio(
    label="Do you have any Memory Complaints?",
    options=["Yes", "No"],
    key="memory_complaints",
    help="Have you noticed any difficulties in remembering things recently?"
)
# Update patient_data based on selection
patient_data["MemoryComplaints"] = 1 if memory_complaints == "Yes" else 0

# Behavioral Problems Section
behavioral_problems = st.radio(
    label="Do you have any Behavioral Problems?",
    options=["Yes", "No"],
    key="behavioral_problems",
    help="Have you or others noticed any changes in your behavior, such as increased agitation, anxiety, or withdrawal?"
)
# Update patient_data based on selection
patient_data["BehavioralProblems"] = 1 if behavioral_problems == "Yes" else 0

patient_data["MMSE"] = st.number_input("Enter your MMSE score (0-30):", min_value=0.0, step=0.1, help="MMSE (Mini-Mental State Examination) is a brief 30-point questionnaire to screen for cognitive impairment.")

patient_data["ADL"] = st.number_input("Enter your ADL score (0-10):", min_value = 0.0,step=0.1, help="ADL (Activities of Daily Living) measures your ability to perform basic self-care tasks. Higher scores mean more independence.")

patient_data["FunctionalAssessment"] = st.number_input("Enter your Functional Assessment score (0-10):", min_value = 0.0,step=0.1,help="Functional Assessment evaluates your ability to perform daily activities.")


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
        st.markdown("#### Prediction Results")
        if prediction == 1:
            st.write(f"The model predicts a high risk of Alzheimer's disease (Confidence: {prediction_proba[1]:.1%}).")
        else:
            st.write(f"The model predicts a low risk of Alzheimer's disease (Confidence: {prediction_proba[0]:.1%}).")
        
        st.markdown("#### Explanation & Advice")
        # Explain prediction using SHAP and generate LLM response
        with st.spinner('Generating...'):
            explanation_text = explain_prediction_with_shap(patient_data, stacked_model, scaler, feature_names)
            advice = generate_llm_response(explanation_text, prediction, prediction_proba,patient_data)
            
            # Display explanation and advice
            
            st.write(advice)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")