from groq import Groq
import os
import streamlit as st


MODEL = "llama3-70b-8192"



# Load API key from Streamlit secrets
GROQ_API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def generate_llm_response(explanation_text, prediction, prediction_proba,patient_data):
    # Define context for Alzheimer's disease
    risk_level = "high risk" if prediction == 1 else "low risk"
    confidence = prediction_proba[prediction]  # Get confidence for the predicted class
    
    # Construct the prompt for the LLM
    prompt = f"""
You are an AI assistant helping explain predictions from a machine learning model designed to assess Alzheimer's disease risk.
The model predicted a {risk_level} of Alzheimer's disease with a confidence of {confidence:.1%}.
{explanation_text}

The patient's data is as follows - {patient_data} where MemoryComplaints has a score of 1 or 0 (1 meaning yes and 0 meaning no), BehavioralProblems has a score of 1 or 0 (1 meaning yes and 0 meaning no), ADL has a score of 0-10 where lower values indicate greater impairment, MMSE has a score of 0-30, and FunctionalAssessment has a score of 0-10 where lower values indicate greater impairment.

Please explain this prediction in simple terms suitable for a non-technical audience by highlighting the SHAP contributions for each feature.
Additionally, provide actionable lifestyle advice tailored to Alzheimer's disease management based on these feature contributions.
Do it consisely, under 600 tokens.
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
