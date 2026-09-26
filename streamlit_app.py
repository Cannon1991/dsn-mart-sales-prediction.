import os

# 1. Define the complete dashboard script code block
dashboard_code = """import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="DSN Mart Revenue Engine", layout="wide", page_icon="📈")

# Fallback dataset to ensure the app never crashes
def train_backup_model():
    sample_records = [
        ['row_00000','PRD-PRFP9S',14.252,'Low Fat',0.0271,'Frozen Foods',81.37,'STORE-AGY',45,'Large','Tier_3','Standard Supermarket',1764.98],
        ['row_00001','PRD-PXXK71',7.698,'Low Fat',0.072,'HEALTH AND HYGIENE',42.05,'STORE-YLW',35,'Small','Tier_1','Standard Supermarket',342.13],
        ['row_00002','PRD-V5MOIJ',14.264,'Regular',0.0421,'Canned',41.35,'STORE-89Z',33,'Medium','Tier_1','Standard Supermarket',378.85],
        ['row_00003','PRD-UN5Z3J',12.0,'Regular',0.0449,'soft drinks',174.35,'STORE-7WS',47,'Medium','Tier_3','Flagship Hypermarket',5595.72],
        ['row_00004','PRD-6RDQYB',10.338,'Regular',0.012,'meat',203.06,'STORE-9RG',28,'Small','Tier_2','Standard Supermarket',2375.36],
        ['row_00005','PRD-E6VA05',9.222,'Low Fat',0.0405,'soft drinks',31.69,'STORE-89Z',33,'Medium','Tier_1','Standard Supermarket',842.75],
        ['row_00006','PRD-MHLD93',12.0,'Low Fat',0.1239,'Canned',212.23,'STORE-7WS',47,'Medium','Tier_3','Flagship Hypermarket',4512.7],
        ['row_00007','PRD-EF6Y39',10.21,'Regular',0.0133,'Snack Foods',142.7,'STORE-HL7,',23,'Medium','Tier_3','Superstore',2373.55],
        ['row_00008','PRD-IAZMTU',14.054,'Low Fat',0.0233,'baking goods',102.71,'STORE-HL7,',23,'Medium','Tier_3','Superstore',2037.05]
    ]
    df = pd.DataFrame(sample_records, columns=['id','product_code','product_weight_kg','fat_content','shelf_visibility','product_category','product_price','store_code','store_age_years','store_size','store_location_tier','store_format','total_sales'])
    
    df['product_category'] = df['product_category'].astype(str).str.upper().str.strip()
    df['fat_content'] = df['fat_content'].astype(str).str.upper().str.strip()
    
    df['price_per_kg'] = df['product_price'] / (df['product_weight_kg'] + 1e-5)
    df['visibility_price_ratio'] = df['shelf_visibility'] * df['product_price']
    df['store_establishment_year'] = 2026 - df['store_age_years']
    
    cat_sales_map = df.groupby('product_category')['total_sales'].mean().to_dict()
    df['cat_historical_avg_sales'] = df['product_category'].map(cat_sales_map).fillna(0)
    
    categorical_cols = ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    features = [c for c in df.columns if c not in ['id', 'product_code', 'total_sales']]
    X = df[features]
    y = df['total_sales']
    
    fallback_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
    fallback_model.fit(X, y)
    
    preprocessors = {'cat_sales_map': cat_sales_map, 'label_encoders': label_encoders}
    return fallback_model, preprocessors, features

model, preprocessors, features = train_backup_model()

st.title("🇳🇬 DSN Mart Product-Store Sales Intelligence Engine")
st.markdown("Predict product-level sales revenue optimization dynamically across retail cluster configurations in Nigeria.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Attributes")
    product_code = st.text_input("Product Unique Code", "PRD-PRFP9S")
    product_price = st.number_input("Unit Retail Listed Price (₦)", min_value=10.0, max_value=5000.0, value=150.0)
    product_weight = st.number_input("Product Mass Weight (kg)", min_value=0.1, max_value=50.0, value=12.5)
    fat_content = st.selectbox("Fat Classification Group", ["Low Fat", "Regular"])
    product_category = st.selectbox("Product Department", 
        ["FROZEN FOODS", "HEALTH AND HYGIENE", "CANNED", "SOFT DRINKS", "MEAT", "SNACK FOODS", "BAKING GOODS", "DAIRY", "HOUSEHOLD", "BREADS", "BREAKFAST", "OTHERS", "SEAFOOD"])
    shelf_visibility = st.slider("Allocated Display Visibility Proportion Ratio", 0.0, 1.0, 0.05)

with col2:
    st.subheader("🏪 Store Cluster Context")
    store_code = st.text_input("Store Node ID", "STORE-AGY")
    store_age_years = st.slider("Store Operational Seniority Lifespan (Years)", 1, 60, 15)
    store_size = st.selectbox("Store Footprint Capacity", ["Small", "Medium", "Large"])
    store_location_tier = st.selectbox("Location Development Category", ["Tier_1", "Tier_2", "Tier_3"])
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
"""

# Force create the files directly on the machine's active disk
with open("streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(dashboard_code)

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("streamlit>=1.35.0\npandas>=2.0.0\nnumpy>=1.24.0\nscikit-learn>=1.3.0\n")

# Verify their active storage states
print(f"File 'streamlit_app.py' Created: {os.path.exists('streamlit_app.py')}")
print(f"File 'requirements.txt' Created: {os.path.exists('requirements.txt')}")
