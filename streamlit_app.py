import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Ensure layout is configured before any structural processing elements render
st.set_page_config(
    page_title="DSN Mart Revenue Intelligence Core", 
    layout="wide", 
    page_icon="📈"
)

# 📊 Baseline Operational Data Ingestion Layer
def get_historical_analytics_data():
    sample_records = [
        ['row_00000','PRD-PRFP9S',14.252,'Low Fat',0.0271,'Frozen Foods',81.37,'STORE-AGY',45,'Large','Tier_3','Standard Supermarket',1764.98],
        ['row_00001','PRD-PXXK71',7.698,'Low Fat',0.0720,'HEALTH AND HYGIENE',42.05,'STORE-YLW',35,'Small','Tier_1','Standard Supermarket',342.13],
        ['row_00002','PRD-V5MOIJ',14.264,'Regular',0.0421,'Canned',41.35,'STORE-89Z',33,'Medium','Tier_1','Standard Supermarket',378.85],
        ['row_00003','PRD-UN5Z3J',12.000,'Regular',0.0449,'SOFT DRINKS',174.35,'STORE-7WS',47,'Medium','Tier_3','Flagship Hypermarket',5595.72],
        ['row_00004','PRD-6RDQYB',10.338,'Regular',0.0120,'MEAT',203.06,'STORE-9RG',28,'Small','Tier_2','Standard Supermarket',2375.36],
        ['row_00005','PRD-E6VA05',9.222,'Low Fat',0.0405,'SOFT DRINKS',31.69,'STORE-89Z',33,'Medium','Tier_1','Standard Supermarket',842.75],
        ['row_00006','PRD-MHLD93',12.000,'Low Fat',0.1239,'Canned',212.23,'STORE-7WS',47,'Medium','Tier_3','Flagship Hypermarket',4512.70],
        ['row_00007','PRD-EF6Y39',10.210,'Regular',0.0133,'Snack Foods',142.70,'STORE-HL7',23,'Medium','Tier_3','Superstore',2373.55],
        ['row_00008','PRD-IAZMTU',14.054,'Low Fat',0.0233,'Baking Goods',102.71,'STORE-HL7',23,'Medium','Tier_3','Superstore',2037.05]
    ]
    df = pd.DataFrame(sample_records, columns=['id','product_code','product_weight_kg','fat_content','shelf_visibility','product_category','product_price','store_code','store_age_years','store_size','store_location_tier','store_format','total_sales'])
    df['product_category'] = df['product_category'].astype(str).str.upper().str.strip()
    df['fat_content'] = df['fat_content'].astype(str).str.upper().str.strip()
    return df

# Fail-Safe Local Model Inbound Trainer (Generates robust structures if artifacts are missing)
def train_fallback_ensemble_model(df):
    df_train = df.copy()
    df_train['product_type_prefix'] = df_train['product_code'].astype(str).str[:3]
    df_train['price_per_kg'] = df_train['product_price'] / (df_train['product_weight_kg'] + 1e-5)
    df_train['visibility_price_ratio'] = df_train['shelf_visibility'] * df_train['product_price']
    df_train['store_establishment_year'] = 2026 - df_train['store_age_years']
    df_train['is_visibility_allocated'] = (df_train['shelf_visibility'] > 0).astype(int)
    
    cat_sales_map = df_train.groupby('product_category')['total_sales'].mean().to_dict()
    df_train['cat_historical_avg_sales'] = df_train['product_category'].map(cat_sales_map).fillna(0)
    
    mean_price_map = df_train.groupby('product_category')['product_price'].mean().to_dict()
    mean_vis_map = df_train.groupby('product_category')['shelf_visibility'].mean().to_dict()
    df_train['price_to_category_avg_ratio'] = df_train['product_price'] / df_train['product_category'].map(mean_price_map).fillna(1.0)
    df_train['relative_visibility_in_category'] = df_train['shelf_visibility'] / df_train['product_category'].map(mean_vis_map).fillna(1.0)
    df_train['sku_historical_mean_sales'] = df_train['total_sales'].mean()
    
    df_train['composite_store_density_proxy'] = df_train['store_format'].astype(str) + "_" + df_train['store_size'].astype(str)
    
    categorical_cols = ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format', 'product_type_prefix', 'composite_store_density_proxy']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        label_encoders[col] = le
        
    features = [c for c in df_train.columns if c not in ['id', 'product_code', 'total_sales']]
    X = df_train[features]
    y = df_train['total_sales']
    
    m1 = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42)
    m2 = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=24)
    m1.fit(X, y)
    m2.fit(X, y)
    
    preprocessors = {
        'cat_sales_map': cat_sales_map, 'label_encoders': label_encoders,
        'mean_price_map': mean_price_map, 'mean_vis_map': mean_vis_map,
        'sku_velocity_map': {}, 'global_sales_mean': df_train['total_sales'].mean()
    }
    return m1, m2, preprocessors, features, 0.50

@st.cache_resource
def load_pipeline_artifacts():
    raw_df = get_historical_analytics_data()
    if os.path.exists('models/artifacts.pkl'):
        try:
            with open('models/artifacts.pkl', 'rb') as f:
                artifacts = pickle.load(f)
                if 'lgb_model' in artifacts and 'cat_model' in artifacts:
                    return (
                        artifacts['lgb_model'], artifacts['cat_model'], 
                        artifacts['preprocessors'], artifacts['features'], 
                        artifacts['best_weight'], True
                    )
        except Exception:
            pass
    m1, m2, preprocessors, features, weight = train_fallback_ensemble_model(raw_df)
    return m1, m2, preprocessors, features, weight, False

# Execute Data Ingestion
raw_analytics_df = get_historical_analytics_data()
model1, model2, preprocessors, features, blend_weight, is_production_ensemble = load_pipeline_artifacts()

# --- APP LAYOUT ---
st.title("📈 DSN Mart Retail Intelligence & Revenue Engine")
st.markdown("Optimize product distribution, spatial visibility parameters, and projected store layout revenue matrix yields across Nigeria.")

if is_production_ensemble:
    st.sidebar.success("🏆 Deployed Status: Live Production LightGBM + CatBoost Ensemble Active")
else:
    st.sidebar.info("💡 Deployed Status: Fallback Baseline Engine Active")

tab1, tab2, tab3 = st.tabs(["🔮 Demand Forecasting Engine", "📊 Operational Revenue Analytics", "📂 Batch Prediction Center"])

with tab1:
    st.markdown("### Interactive Single-SKU Forecast Estimator")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Product Attributes")
        product_code = st.text_input("Product SKU Code", "PRD-PRFP9S")
        product_price = st.number_input("Unit Retail Price (₦)", min_value=10.0, max_value=5000.0, value=150.0)
        product_weight = st.number_input("Product Mass Weight (kg)", min_value=0.1, max_value=50.0, value=12.5)
        fat_content = st.selectbox("Fat Classification Group", ["LOW FAT", "REGULAR"])
        product_category = st.selectbox("Product Operational Department", 
            ["FROZEN FOODS", "HEALTH AND HYGIENE", "CANNED", "SOFT DRINKS", "MEAT", "SNACK FOODS", "BAKING GOODS", "DAIRY", "HOUSEHOLD", "BREADS", "BREAKFAST", "OTHERS", "SEAFOOD"])
        shelf_visibility = st.slider("Display Visibility Allocation Ratio", 0.0, 1.0, 0.05)

    with col2:
        st.subheader("🏪 Store Cluster Context")
        store_code = st.text_input("Store Node ID", "STORE-AGY")
        store_age_years = st.slider("Store Seniority Lifespan (Years)", 1, 60, 15)
        store_size = st.selectbox("Store Size Footprint Capacity", ["Small", "Medium", "Large"])
        store_location_tier = st.selectbox("Urban Location Tier Category", ["Tier_1", "Tier_2", "Tier_3"])
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
        input_data['product_type_prefix'] = input_data['product_code'].astype(str).str[:3]
        
        input_data['price_per_kg'] = input_data['product_price'] / (input_data['product_weight_kg'] + 1e-5)
        input_data['visibility_price_ratio'] = input_data['shelf_visibility'] * input_data['product_price']
        input_data['is_visibility_allocated'] = (input_data['shelf_visibility'] > 0).astype(int)
        input_data['store_establishment_year'] = 2026 - input_data['store_age_years']
        
        input_data['price_to_category_avg_ratio'] = input_data['product_price'] / preprocessors['mean_price_map'].get(product_category, product_price)
        input_data['relative_visibility_in_category'] = input_data['shelf_visibility'] / preprocessors['mean_vis_map'].get(product_category, 1.0)
        input_data['sku_historical_mean_sales'] = preprocessors.get('sku_velocity_map', {}).get(product_code, preprocessors['global_sales_mean'])
        input_data['composite_store_density_proxy'] = store_format + "_" + store_size
        
        label_encoders = preprocessors['label_encoders']
        for col in ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format', 'product_type_prefix', 'composite_store_density_proxy']:
            le = label_encoders[col]
            input_data[col] = input_data[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_)
