import os

import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="DSN Mart Revenue Intelligence Core", 
    layout="wide", 
    page_icon="📈"
)

# 📊 Historical Ingestion Matrix Data Mocks
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

def train_backup_model(df):
    df_train = df.copy()
    df_train['price_per_kg'] = df_train['product_price'] / (df_train['product_weight_kg'] + 1e-5)
    df_train['visibility_price_ratio'] = df_train['shelf_visibility'] * df_train['product_price']
    df_train['store_establishment_year'] = 2026 - df_train['store_age_years']
    
    cat_sales_map = df_train.groupby('product_category')['total_sales'].mean().to_dict()
    df_train['cat_historical_avg_sales'] = df_train['product_category'].map(cat_sales_map).fillna(0)
    
    categorical_cols = ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        label_encoders[col] = le
        
    features = [c for c in df_train.columns if c not in ['id', 'product_code', 'total_sales']]
    X = df_train[features]
    y = df_train['total_sales']
    
    fallback_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
    fallback_model.fit(X, y)
    
    preprocessors = {'cat_sales_map': cat_sales_map, 'label_encoders': label_encoders}
    return fallback_model, preprocessors, features

raw_analytics_df = get_historical_analytics_data()
model, preprocessors, features = train_backup_model(raw_analytics_df)

# --- APP LAYOUT ---
st.title("📈 DSN Mart Retail Intelligence & Revenue Engine")
st.markdown("Optimize product distribution, spatial visibility parameters, and projected store layout revenue matrix yields across Nigeria.")

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
        
        input_data['price_per_kg'] = input_data['product_price'] / (input_data['product_weight_kg'] + 1e-5)
        input_data['visibility_price_ratio'] = input_data['shelf_visibility'] * input_data['product_price']
        input_data['store_establishment_year'] = 2026 - input_data['store_age_years']
        
        cat_sales_map = preprocessors['cat_sales_map']
        input_data['cat_historical_avg_sales'] = input_data['product_category'].map(cat_sales_map).fillna(0)
        
        label_encoders = preprocessors['label_encoders']
        for col in ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format']:
            le = label_encoders[col]
            input_data[col] = input_data[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
            input_data[col] = le.transform(input_data[col])
            
        X_infer = input_data[features]
        prediction = model.predict(X_infer)
        st.success(f"### 📈 Projected Single-SKU Sales Estimation: **₦ {prediction[0]:,.2f}**")

with tab2:
    st.markdown("### Regional Store Performance Insights & Visual Analytics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Fleet Sample Revenue", f"₦ {raw_analytics_df['total_sales'].sum():,.2f}")
    m2.metric("Average Departmental Price Index", f"₦ {raw_analytics_df['product_price'].mean():,.2f}")
    m3.metric("Monitored Distribution Hubs Node Fleet", f"{raw_analytics_df['store_code'].nunique()} Active Nodes")
    
    st.markdown("---")
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.subheader("🏬 Revenue Matrix by Store Format Group")
        format_chart_data = raw_analytics_df.groupby('store_format')['total_sales'].sum().reset_index()
        format_chart_data = format_chart_data.set_index('store_format')
        st.bar_chart(format_chart_data, y="total_sales", color="#FF4B4B")

    with v_col2:
        st.subheader("🛍️ Revenue Matrix by Product Category Layer")
        category_chart_data = raw_analytics_df.groupby('product_category')['total_sales'].mean().reset_index()
        category_chart_data = category_chart_data.sort_values(by="total_sales", ascending=False)
        category_chart_data = category_chart_data.set_index('product_category')
        st.bar_chart(category_chart_data, y="total_sales", color="#29B5E8")

with tab3:
    st.markdown("### 📥 Bulk Processing Pipeline Module")
    st.markdown("Upload your structural test dataset file (`test.csv`) to calculate batch-inferences and export prediction sets instantly.")
    
    uploaded_file = st.file_uploader("Choose a CSV file containing inventory rows", type="csv")
    
    if uploaded_file is not None:
        try:
            # Parse inbound worksheet sheet rows
            test_batch_df = pd.read_csv(uploaded_file)
            st.info(f"📋 File mapped successfully! Detected **{len(test_batch_df)} records** awaiting feature mapping pipeline.")
            
            # Re-running preprocessing steps symmetrically to avoid downstream alignment shifts
            processed_batch = test_batch_df.copy()
            processed_batch['product_category'] = processed_batch['product_category'].astype(str).str.upper().str.strip()
            processed_batch['fat_content'] = processed_batch['fat_content'].astype(str).str.upper().str.strip()
            
            # Fill missing column parameters cleanly using baseline proxies
            processed_batch['product_weight_kg'] = processed_batch['product_weight_kg'].fillna(12.0)
            processed_batch['store_size'] = processed_batch['store_size'].fillna('Medium')
            
            # Feature Synthesis Layer
            processed_batch['price_per_kg'] = processed_batch['product_price'] / (processed_batch['product_weight_kg'] + 1e-5)
