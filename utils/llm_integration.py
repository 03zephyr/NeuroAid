from groq import Groq
import os
import streamlit as st


MODEL = "llama3-70b-8192"



# Load API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def generate_llm_response(explanation_text, prediction, prediction_proba):
    # Define context for Alzheimer's disease
    risk_level = "high risk" if prediction == 1 else "low risk"
    confidence = prediction_proba[prediction]  # Get confidence for the predicted class
    
    # Construct the prompt for the LLM
    prompt = f"""
You are an AI assistant helping explain predictions from a machine learning model designed to assess Alzheimer's disease risk.
The model predicted a {risk_level} of Alzheimer's disease with a confidence of {confidence:.1%}.
{explanation_text}

Please explain this prediction in simple terms suitable for a non-technical audience.
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
