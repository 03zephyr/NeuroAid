import streamlit as st
import pandas as pd
import joblib
from utils.shap_explainer import explain_prediction_with_shap
from utils.llm_integration import generate_llm_response
import json
from PIL import Image

st.set_page_config(page_title="NeuroAid", page_icon="🧠")



logo_path = "images/logo-removebg-preview.png"  # Replace with your image path
logo_image = Image.open(logo_path)

icon = "images/small logo.png"
icon_img = Image.open(icon)

st.logo(icon_img, size="large",icon_image=icon_img)

st.sidebar.image(logo_image, use_container_width=False)

import streamlit as st


# Define navigation pages
pages = [
    st.Page("pages/predict.py", title="Predict", icon="🔮"),
    st.Page("pages/information.py", title="Information", icon="ℹ️")
]

# Create the navigation menu in the sidebar
pg = st.navigation(pages, position="sidebar", expanded=True)

# Run the selected page
pg.run()
