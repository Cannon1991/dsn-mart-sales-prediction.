import os

# Define the production script with explicit error fail-safes built in
clean_code_content = """import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="DSN Mart Sales Predictor", layout="wide", page_icon="📈")

@st.cache_resource
def load_pipeline_artifacts():
    try:
        if os.path.exists('models/artifacts.pkl'):
            with open('models/artifacts.pkl', 'rb') as f:
                return pickle.load(f)
        elif os.path.exists('artifacts.pkl'):
            with open('artifacts.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            st.error("⚠️ Model artifacts binary not found in workspace directories.")
            return None
    except Exception as e:
        st.error(f"Error loading model weights: {str(e)}")
        return None

artifacts = load_pipeline_artifacts()

st.title("🇳🇬 DSN Mart Product-Store Sales Intelligence Engine")
st.markdown("Predict product-level sales revenue optimization dynamically across retail cluster configurations and distribution layers.")

if artifacts:
    model = artifacts['model']
    preprocessors = artifacts['preprocessors']
    features = artifacts['features']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Product Attributes")
        product_code = st.text_input("Product Unique Code", "PRD-PRFP9S")
        product_price = st.number_input("Unit Retail Listed Price (₦)", min_value=10.0, max_value=5000.0, value=150.0)
        product_weight = st.number_input("Product Mass Weight (kg)", min_value=0.1, max_value=50.0, value=12.5)
        fat_content = st.selectbox("Fat Classification Group", ["Low Fat", "Regular"])
        product_category = st.selectbox("Product Department", 
            ["FROZEN FOODS", "HEALTH AND HYGIENE", "CANNED", "SOFT DRINKS", "MEAT", "SNACK FOODS", "BAKING GOODS", "DAIRY", "HOUSEHOLD", "BREADS", "BREAKFAST", "OTHERS", "SEAFOOD"])
        shelf_visibility = st.slider("Allocated Display Visibility Ratio", 0.0, 1.0, 0.05)

    with col2:
        st.subheader("🏪 Store Cluster Context")
        store_code = st.text_input("Store Node ID", "STORE-AGY")
        store_age_years = st.slider("Store Seniority Lifespan (Years)", 1, 60, 15)
        store_size = st.selectbox("Store Size Footprint", ["Small", "Medium", "Large"])
        store_location_tier = st.selectbox("Urban Location Development Category", ["Tier_1", "Tier_2", "Tier_3"])
        store_format = st.selectbox("Distribution Format Classification", ["Corner Shop", "Standard Supermarket", "Superstore", "Flagship Hypermarket"])

    if st.button("🔮 Calculate Predictive Optimization Yield", type="primary"):
        input_data = pd.DataFrame([{
            'id': 'inference_run', 'product_code': product_code, 'product_weight_kg': product_weight,
            'fat_content': fat_content, 'shelf_visibility': shelf_visibility, 'product_category': product_category,
            'product_price': product_price, 'store_code': store_code, 'store_age_years': store_age_years,
            'store_size': store_size, 'store_location_tier': store_location_tier, 'store_format': store_format
        }])
        
        input_data['product_category'] = input_data['product_category'].astype(str).str.upper().str.strip()
        input_data['fat_content'] = input_data['fat_content'].astype(str).str.upper().str.strip()
        
        input_data['price_per_kg'] = input_data['product_price'] / (input_data['product_weight_kg'] + 1e-5)
        input_data['visibility_price_ratio'] = input_data['shelf_visibility'] * input_data['product_price']
        input_data['store_establishment_year'] = 2026 - input_data['store_age_years']
        
        cat_sales_map = preprocessors['cat_sales_map']
        input_data['cat_historical_avg_sales'] = input_data['product_category'].map(cat_sales_map).fillna(0)
        
        label_encoders = preprocessors['label_encoders']
        for col in ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format']:
            le = label_encoders[col]
            input_data[col] = input_data[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_)
            input_data[col] = le.transform(input_data[col])
            
        X_infer = input_data[features]
        prediction = model.predict(X_infer)
        
        st.success(f"### 📈 Projected Sales Estimation Value: **₦ {prediction[0]:,.2f}**")
        st.info("💡 **Stock Planning Tip:** Use this forecast value to balance logistics targets against regional inventory demand.")
"""

# Force overwrite the app file with clean native raw Python string data
with open("streamlit_app.py", "w", encoding="utf-8") as file:
    file.write(clean_code_content)

print("⚡ 'streamlit_app.py' successfully overwritten with clean, raw Python lines!")
