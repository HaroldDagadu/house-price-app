
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model (change path if needed)
model = joblib.load("best_xgb_pipeline.joblib")

st.title("🏡 House Price Prediction App")

st.write("Enter house features below to predict price:")

# Example inputs (extend with more features!)
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1500)
garage_cars = st.slider("Garage Cars", 0, 4, 2)
total_bsmt_sf = st.number_input("Total Basement SF", value=800)
year_built = st.number_input("Year Built", value=2000)

# Build input dataframe
input_data = pd.DataFrame([{
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "GarageCars": garage_cars,
    "TotalBsmtSF": total_bsmt_sf,
    "YearBuilt": year_built
}])

# Predict
pred = model.predict(input_data)[0]
pred_price = np.expm1(pred)  # convert back from log
st.success(f"Predicted House Price: ${pred_price:,.0f}")
