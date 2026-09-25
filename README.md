# 📈 DSN Mart Product-Store Sales Intelligence Engine

![Python](https://shields.io)
![Streamlit](https://shields.io)
![Scikit-Learn](https://shields.io)
![Status](https://shields.io)

An end-to-end machine learning regression pipeline and interactive business intelligence dashboard designed to predict product-level sales revenue optimization across **DSN Mart's** diverse retail footprint in Nigeria. The system transitions from small corner shops to flagship hypermarkets, modeling localized consumer demand dynamics to drive smart stock planning, optimal pricing models, and data-driven store infrastructure investments.

🌐 **Live Dashboard Application URL:** [https://streamlit.app](https://streamlit.io)

---

## 📑 Table of Contents
* [Project Overview](#-project-overview)
* [Data Architecture & Preprocessing](#-data-architecture--preprocessing)
* [Feature Engineering Matrix](#-feature-engineering-matrix)
* [Model Evaluation & Cross-Validation](#-model-evaluation--cross-validation)
* [Repository Layout Structure](#-repository-layout-structure)
* [Installation & Local Reproduction](#-installation--local-reproduction)
* [Streamlit Web Interface App](#-streamlit-web-interface-app)

---

## 📊 Project Overview

DSN Mart operates multi-format retail nodes distributed across major Nigerian urban hubs, state capitals, and rural trading centers. This project addresses the challenge of sales variance by implementing a high-capacity Random Forest ensemble framework to predict the `total_sales` value metric, evaluated via **Root Mean Squared Error (RMSE)**.

### Dataset Schema

| Column Feature | Data Type | Description Layer |
| :--- | :--- | :--- |
| `id` | String | Unique record row tracking sequence identifier |
| `product_code` | String | Unique SKU categorization standard alpha-numeric code |
| `product_weight_kg` | Float | Product weight in kilograms (Contains structural missing values) |
| `fat_content` | String | Dietary text tags: `Low Fat`, `Regular` (Inconsistent capitalization) |
| `shelf_visibility` | Float | Spatial display area proportion index allocated in-store |
| `product_category` | String | Operational sales department classification mapping |
| `product_price` | Float | Listed unit retail price in Nigerian Naira (₦) |
| `store_code` | String | Unique store node organizational hub footprint index |
| `store_age_years` | Integer | Operational runtime lifecycle longevity of the store node |
| `store_size` | String | Retail capacity classification hierarchy: `Small`, `Medium`, `Large` |
| `store_location_tier`| String | Regional classification matrix: `Tier_1`, `Tier_2`, `Tier_3` |
| `store_format` | String | Operational format: `Corner Shop`, `Standard Supermarket`, `Superstore`, `Flagship Hypermarket` |
| `total_sales` | Float | **Target Variable**. Total revenue generated per SKU/Store node configuration |

---

## 🧼 Data Architecture & Preprocessing

The dataset mirrors real-world data patterns, requiring rigorous data validation and deterministic preparation logic:
* **String Normalization:** Text attributes (`product_category`, `fat_content`) are standardized to clean trailing spaces and inconsistent letter casings.
* **Deterministic Imputation:** 
  * Missing `product_weight_kg` dimensions are imputed dynamically using the mean weight of that item's specific generalized `product_category`.
  * Missing `store_size` properties are imputed using the localized mode configuration matching the corresponding (`store_location_tier`, `store_format`) configuration.
* **Encoding Infrastructure:** Uniform string variables are indexed using deterministic label transforms mapping unseen data entries onto fallback structural defaults gracefully.

---

## ⚙️ Feature Engineering Matrix

To maximize predictive accuracy, multiple domain-specific interaction features were synthesized:
1. **Price Density Ratio (`price_per_kg`):** `product_price / product_weight_kg` — models product value density.
2. **Visual Real-Estate Yield (`visibility_price_ratio`):** `shelf_visibility * product_price` — quantifies high-visibility shelf layout conversions.
3. **Chronological Node Trajectory (`store_establishment_year`):** `2026 - store_age_years` — captures institutional baseline market maturity across generations.
4. **Target Proxy Encoding (`cat_historical_avg_sales`):** Reflects categorical mean target distributions safely out-of-fold to prevent leakage constraints.

---

## 📈 Model Evaluation & Cross-Validation

The regression strategy utilizes a robust **5-Fold Cross-Validation Framework** to verify out-of-sample consistency and guard against over-fitting:

