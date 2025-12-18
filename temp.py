# ============================================
# Streamlit Web App for CO2 Prediction
# Deployment: share.streamlit.io
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# ----------- App Title -----------
st.title("CO2 Prediction Web App")
st.write("Predict CO2 based on Volume using a trained Linear Regression model")

# ----------- Load Trained Model -----------
@st.cache_resource
def load_model():
    with open('25RP19523_MODEL.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# ----------- User Input -----------
st.header("Input Volume")
volume = st.number_input(
    "Enter Volume value",
    min_value=0.0,
    step=1.0
)

# ----------- Prediction -----------
if st.button("Predict CO2"):
    input_data = np.array([[volume]])
    prediction = model.predict(input_data)

    st.success(f"Predicted CO2 value: {prediction[0]:.3f}")

# ----------- Footer -----------
st.markdown("---")
st.markdown("**Developed using Streamlit & Scikit-learn**")
