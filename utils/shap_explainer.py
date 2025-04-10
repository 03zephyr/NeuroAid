import pandas as pd
import shap
import joblib

def explain_prediction_with_shap(patient_data, model, scaler, feature_names):
    # Convert patient data to DataFrame and scale it
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)

    with open("model/explainer.pkl", "rb") as f:
        explainer = joblib.load(f)

    # Compute SHAP values for the patient's data
    shap_values = explainer(pd.DataFrame(patient_scaled, columns=feature_names))

    # Extract top contributing features for this prediction
    top_features = sorted(
    zip(feature_names, shap_values.values[0]),
    key=lambda x: abs(x[1]),
    reverse=True
    )[:5]  # Top 5 features by absolute SHAP value

    # Format explanation text
    explanation_text = f"The top contributing features and their SHAP values are:\n"

    for feature, value in top_features:
        explanation_text += f"- {feature}: {value:.3f}\n"

    return explanation_text  # Return plain text explanation