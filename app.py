import streamlit as st
import pandas as pd
import joblib
from utils.shap_explainer import explain_prediction_with_shap
from utils.llm_integration import generate_llm_response
import json

st.set_page_config(page_title="NeuroAid", page_icon="🧠")

# Define navigation pages
pages = [
    st.Page(r"pages\predict.py", title="Predict", icon="🔮"),
    st.Page(r"pages\information.py", title="Information", icon="ℹ️")
]

# Create the navigation menu in the sidebar
pg = st.navigation(pages, position="sidebar", expanded=True)

# Run the selected page
pg.run()
