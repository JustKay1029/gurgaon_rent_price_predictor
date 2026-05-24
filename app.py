import streamlit as st
import pandas as pd
import pickle

# Set up clean web styling
st.set_page_config(page_title="Gurgaon Price Predictor", layout="centered")
st.title("🏡 Gurgaon Residential Property Valuation Engine")
st.write("Enter property attributes below to generate an instant market valuation.")
st.markdown("---")

# 1. Load your trained model pipeline
@st.cache_resource
def load_pipeline():
    with open("property_pipeline.pkl", "rb") as f:
        return pickle.load(f)

transformer, model = load_pipeline()

# 2. Hardcoded Sector Means Dictionary (From your data engineering phase)
# This maps the sector text name to the exact average price your model expects.
SECTOR_MEANS = {
    "Sector 53": 6.275000e+07,
    "DLF City Phase 5": 5.433333e+07,
    "Sector 54": 5.250000e+07,
    "Sector 26": 4.666667e+07,
    "Sector 50": 4.535000e+07,
    "Sector 28": 4.485000e+07,
    "DLF City Phase 1": 4.077143e+07,
    "NH 8": 4.000000e+07,
    "DLF Golf Course Road": 3.916667e+07,
    "Golf course Extension Road": 3.830000e+07,
    "Other": 2.500000e+07 # Default backup value for any other sector
}

# 3. Create Web UI input form components
col1, col2 = st.columns(2)

with col1:
    transaction = st.selectbox("Transaction Type", ["Resale", "New Property"])
    status = st.selectbox("Property Status", ["Ready to Move", "Under Construction"])
    sector_choice = st.selectbox("Select Location/Sector", list(SECTOR_MEANS.keys()))

with col2:
    bedroom = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5, 6], index=2)
    bathroom = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=2)
    balcony = st.selectbox("Balconies", [0, 1, 2, 3, 4], index=2)

total_area = st.slider("Total Area (Square Feet)", min_value=300, max_value=6000, value=1500, step=50)

st.markdown("---")

# 4. Trigger the prediction engine when the button is clicked
if st.button("Generate Valuation", type="primary", use_container_width=True):
    
    # Extract the matching numeric price average for the chosen sector
    mapped_sector_value = SECTOR_MEANS.get(sector_choice, SECTOR_MEANS["Other"])
    
    # Create a single-row DataFrame with exact training column names
    input_data = pd.DataFrame([{
        "status": status,
        "transaction": transaction,
        "bathroom": bathroom,
        "balcony": balcony,
        "bedroom": bedroom,
        "total_area": total_area,
        "clean_sector": mapped_sector_value
    }])
    
    # Pass the input data through the exact same processing pipeline
    input_transformed = transformer.transform(input_data)
    
    # Generate the prediction output
    predicted_price = model.predict(input_transformed)[0]
    
    # Format the final valuation into readable Indian Currency format (Crores/Lakhs)
    if predicted_price >= 10000000:
        formatted_price = f"₹{predicted_price / 10000000:.2f} Crores"
    else:
        formatted_price = f"₹{predicted_price / 100000:.2f} Lakhs"
        
    # Render the result to the user screen
    st.success(f"### Estimated Market Price: {formatted_price}")