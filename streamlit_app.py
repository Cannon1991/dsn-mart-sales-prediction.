import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def clean_and_engineer_features(df, is_train=True, pipeline_cache=None):
    """
    Advanced Data Preparation & Feature Engineering Pipeline.
    """
    df = df.copy().reset_index(drop=True)
    
    # 1. Standardize Text Strings
    df['product_category'] = df['product_category'].astype(str).str.upper().str.strip()
    df['fat_content'] = df['fat_content'].astype(str).str.upper().str.strip()
    
    # Extract structural code prefixes
    df['product_type_prefix'] = df['product_code'].astype(str).str[:3]

    if is_train:
        pipeline_cache = {}
        # Dynamic Group Imputation Maps
        pipeline_cache['cat_weight_map'] = df.groupby('product_category')['product_weight_kg'].mean().to_dict()
        pipeline_cache['global_mean_weight'] = df['product_weight_kg'].mean()
        pipeline_cache['store_size_mode'] = df.groupby(['store_location_tier', 'store_format'])['store_size'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Medium'
        ).to_dict()
        
        # High-Signal Leaderboard Real-Estate Maps
        pipeline_cache['mean_price_map'] = df.groupby('product_category')['product_price'].mean().to_dict()
        pipeline_cache['mean_vis_map'] = df.groupby('product_category')['shelf_visibility'].mean().to_dict()
        pipeline_cache['sku_velocity_map'] = df.groupby('product_code')['total_sales'].mean().to_dict()
        pipeline_cache['global_sales_mean'] = df['total_sales'].mean()
    
    # Apply Inbound Group Imputation
    cat_weight_map = pipeline_cache['cat_weight_map']
    global_mean_weight = pipeline_cache['global_mean_weight']
    store_size_mode = pipeline_cache['store_size_mode']
    
    df['product_weight_kg'] = df.apply(
        lambda r: r['product_weight_kg'] if pd.notnull(r['product_weight_kg']) 
        else cat_weight_map.get(r['product_category'], global_mean_weight), axis=1
    )
    df['store_size'] = df.apply(
        lambda r: r['store_size'] if pd.notnull(r['store_size'])
        else store_size_mode.get((r['store_location_tier'], r['store_format']), 'Medium'), axis=1
    )

    # 2. Advanced Feature Math Matrix
    df['price_per_kg'] = df['product_price'] / (df['product_weight_kg'] + 1e-5)
    df['visibility_price_ratio'] = df['shelf_visibility'] * df['product_price']
    df['is_visibility_allocated'] = (df['shelf_visibility'] > 0).astype(int)
    df['store_establishment_year'] = 2026 - df['store_age_years']
    
    # Map pricing & tracking metrics from baseline cache
    mean_price_map = pipeline_cache['mean_price_map']
    mean_vis_map = pipeline_cache['mean_vis_map']
    sku_velocity_map = pipeline_cache['sku_velocity_map']
    global_sales_mean = pipeline_cache['global_sales_mean']
    
    df['price_to_category_avg_ratio'] = df['product_price'] / df['product_category'].map(mean_price_map).fillna(1.0)
    df['relative_visibility_in_category'] = df['shelf_visibility'] / df['product_category'].map(mean_vis_map).fillna(1.0)
    df['sku_historical_mean_sales'] = df['product_code'].map(sku_velocity_map).fillna(global_sales_mean)
    
    # Composite Capacity Variable Interaction Key
    df['composite_store_density_proxy'] = df['store_format'].astype(str) + "_" + df['store_size'].astype(str)

    # 3. Encoding Layer Transforms
    categorical_cols = ['fat_content', 'product_category', 'store_code', 'store_size', 'store_location_tier', 'store_format', 'product_type_prefix', 'composite_store_density_proxy']
    
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        pipeline_cache['label_encoders'] = label_encoders
    else:
        label_encoders = pipeline_cache['label_encoders']
        for col in categorical_cols:
            le = label_encoders[col]
            df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
    if is_train:
        return df, pipeline_cache
    return df

def run_ensemble_pipeline():
    print("📈 Extracting and Engineering Feature Data Matrices...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    processed_train, cache = clean_and_engineer_features(train_df, is_train=True)
    processed_test = clean_and_engineer_features(test_df, is_train=False, pipeline_cache=cache)
    
    features = [c for c in processed_train.columns if c not in ['id', 'product_code', 'total_sales']]
    X = processed_train[features]
    y = processed_train['total_sales']
    X_test = processed_test[features]
    
    # Initialize array structures for out-of-fold blending predictions
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    
    # 5-Fold Stratification Cross-Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"🚀 Training Out-Of-Fold Ensemble Array Loop Across 5 Validation Windows...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # 🟢 Algorithm A: LightGBM Regressor Tree Array Configuration
        lgb_model = LGBMRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], 
            callbacks=[] # Add early stopping callbacks if datasets are massive
        )
        oof_lgb[val_idx] = lgb_model.predict(X_va)
        test_lgb += lgb_model.predict(X_test) / kf.n_splits
        
        # 🔵 Algorithm B: CatBoost Regressor Matrix Configuration
        cat_model = CatBoostRegressor(
            iterations=1200, learning_rate=0.03, depth=6, 
            random_seed=42, verbose=0
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        oof_cat[val_idx] = cat_model.predict(X_va)
        test_cat += cat_model.predict(X_test) / kf.n_splits
        
        # Intermediate Scoring Logging
        fold_blend = (oof_lgb[val_idx] * 0.5) + (oof_cat[val_idx] * 0.5)
        fold_rmse = root_mean_squared_error(y_va, fold_blend)
        print(f"  ↳ Fold {fold+1} Optimized Blended Validation RMSE Score: {fold_rmse:.4f}")

    # Evaluating Historical Array Blends to find the mathematical sweet spot
    best_weight = 0.5
    best_rmse = float('inf')
    
    # Search loop to determine the absolute optimal blending ratio
    for w in np.linspace(0, 1, 101):
        blended_oof = (oof_lgb * w) + (oof_cat * (1 - w))
        score = root_mean_squared_error(y, blended_oof)
        if score < best_rmse:
            best_rmse = score
            best_weight = w
            
    print(f"✅ Optimal Leaderboard Weight: {best_weight:.2f} LightGBM / {(1-best_weight):.2f} CatBoost")
    print(f"🏆 Final Out-Of-Fold Cross-Validation Ensemble RMSE: {best_rmse:.4f}")
    
    # 4. Fit Final Retrained Ecosystem weights onto the Full Dataset
    print("💾 Fitting ultimate meta-estimator on total data profiles...")
    final_lgb = LGBMRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    final_cat = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
    
    final_lgb.fit(X, y)
    final_cat.fit(X, y)
    
    # 5. Serialization and File Export Generation Layer
    os.makedirs('models', exist_ok=True)
    with open('models/artifacts.pkl', 'wb') as f:
        pickle.dump({
            'lgb_model': final_lgb, 'cat_model': final_cat,
            'preprocessors': cache, 'features': features, 'best_weight': best_weight
        }, f)
        
    final_test_preds = (final_lgb.predict(X_test) * best_weight) + (final_cat.predict(X_test) * (1 - best_weight))
    submission = pd.DataFrame({'id': test_df['id'], 'total_sales': final_test_preds})
    submission.to_csv('submission.csv', index=False)
    print("🏁 Target forecasts compiled! Final 'submission.csv' generated for Leaderboard Upload.")

if __name__ == "__main__":
    run_ensemble_pipeline()
