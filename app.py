import streamlit as st
import pandas as pd
import pickle

st.title("Gurgaon Residential Property Price Predictor")
st.write("Enter the details below to get an instant market valuation.")

# 1. Load the pipeline assets
with open("property_pipeline.pkl", "rb") as f:
    transformer, model = pickle.load(f)

# 2. Create clean user input selectors
transaction = st.selectbox("Transaction Type", ["Resale", "New Property"])
status = st.selectbox("Property Status", ["Ready to Move", "Under Construction"])

bedroom = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])
bathroom = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6])
balcony = st.selectbox("Balconies", [0, 1, 2, 3, 4])

total_area = st.slider("Total Area (sqft)", min_value=300, max_value=6000, value=1500, step=50)

# 3. Sector Selector (We will handle mapping next)
sector = st.text_input("Enter Sector (e.g., Sector 53)", "Sector 53")