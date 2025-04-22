from groq import Groq
import os
import streamlit as st


MODEL = "llama3-70b-8192"



# Load API key from Streamlit secrets
GROQ_API_KEY = st.secrets['GROQ_API_KEY']

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def generate_llm_response(explanation_text, prediction, prediction_proba,patient_data):
    # Define context for Alzheimer's disease
    risk_level = "high risk" if prediction == 1 else "low risk"
    confidence = prediction_proba[prediction]  # Get confidence for the predicted class
    
    # Construct the prompt for the LLM
    prompt = f"""
You are explaining predictions from a machine learning model assessing Alzheimer's disease risk.

The model predicted a {risk_level} risk of Alzheimer's disease with a confidence of {confidence:.1%}.
{explanation_text}

The patient's data includes:

    Memory Complaints (1/0): {patient_data["MemoryComplaints"]}

    Behavioral Problems (1/0): {patient_data["BehavioralProblems"]}

    ADL (0-10): {patient_data["ADL"]}

    MMSE (0-30): {patient_data["MMSE"]}

    Functional Assessment (0-10): {patient_data["FunctionalAssessment"]}

Please explain this prediction in simple terms by highlighting the SHAP contributions for each feature. For example, you might describe how a low ADL score or a low MMSE score impacts risk.

Provide actionable lifestyle advice tailored to Alzheimer's disease management based on these feature contributions. For instance, if ADL is a major contributor, suggest activities to improve daily functioning. If MMSE is low, recommend cognitive stimulation exercises.

Do it concisely, under 600 tokens.
Do not include any introductory lines like "I'm here to help explain.
"""

    try:
        # Call the Groq API to generate the response
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in Alzheimer's disease."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=600,  # Adjust token limit based on expected response length
            temperature=0.7  # Control creativity of the response
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"An error occurred while generating the LLM response: {str(e)}"
