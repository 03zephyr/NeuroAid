import streamlit as st

st.title("NeuroAid 🧠")

st.markdown("""
### Welcome to NeuroAid!

**NeuroAid** is an innovative AI-powered application designed to help patients and clinicians assess the risk of Alzheimer's disease using minimal, easy-to-obtain features as well as advanced MRI-based diagnostics. By combining accessible risk prediction, explainable AI, and practical advice, NeuroAid supports both early detection and informed decision-making.

---
""")

st.markdown("#### Where would you like to go?")

if st.button("🔮 Predict",help = "Estimate your Alzheimer's risk with a quick, simple questionnaire."):
    st.switch_page("pages/predict.py")

if st.button("📸 MRI Prediction", help = "Get a diagnosis of your MRI scan, with visual explanations."):
    st.switch_page("pages/MRI_Predict.py")

if st.button("ℹ️ About", help = "Learn about the features used in prediction, how the AI works, and tips for brain health."):
    st.switch_page("pages/information.py")


