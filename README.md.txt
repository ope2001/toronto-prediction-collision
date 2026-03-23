# Toronto Collision Risk Forecasting & Dynamic Police Resource Allocation

## Project Overview

This capstone project builds a short-horizon traffic collision risk forecasting system for Toronto neighbourhoods.

The main goal is to predict **next-3-hour collision risk** using:
- historical collision patterns
- temporal features
- weather conditions

The project also includes:
- an interactive **Streamlit dashboard**
- choropleth **hotspot mapping**
- an optional **MILP-based patrol planning support layer**

This makes the project an end-to-end decision-support workflow:
**data -> forecasting -> visualization -> practical deployment support**

## Team Members

- Adeyi Fridaus (NF1004424)
- Jigme Jigme (NF1000447)
- Jigme Yeshi (NF1004171)
- Jambayang Singye (NF1002341)

**Supervisor:** Dr. Hany Osman  
**Course:** DAMO-699-2 - Winter 2026 Capstone Project  
**Institution:** University of Niagara Falls Canada

## Problem Type

**Spatiotemporal multiclass classification**

The model predicts neighbourhood-level collision risk for the next 3-hour block:

- **Class 0** = no collision
- **Class 1** = one collision
- **Class 2** = two or more collisions / high risk

## Project Objective

- Forecast short-term collision risk across Toronto neighbourhoods
- Identify emerging hotspots before collisions occur
- Support proactive patrol planning
- Add practical deployment value through dashboarding and optimization

## Data Sources

- **Toronto Police Service collision records** (2023-2025)
- **City of Toronto hourly weather data** (2023-2025)

## Final Analytical Dataset

- **205,227** raw collision records
- **34,097** hourly weather observations
- **158** Toronto neighbourhoods
- **16** police divisions
- **1,383,922** final analytical rows
- **50** total columns in the analytical dataset

## Data Preparation

Main preprocessing steps:
- cleaned timestamps and standardized codes
- aggregated outcomes into **3-hour neighbourhood blocks**
- built a full neighbourhood-time grid
- preserved zero-collision periods
- engineered cyclical time features
- created lag and rolling collision features
- added weather predictors
- defined next-block multiclass targets

## Class Distribution

The target is highly imbalanced:

- **Class 0:** 89.30%
- **Class 1:** 9.27%
- **Class 2:** 1.43%

Because of this imbalance, accuracy alone was not a reliable selection metric.

## Models Evaluated

The modeling workflow progressed through:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest
4. CatBoost
5. LightGBM
6. ExtraTrees
7. Leakage-safe stacking ensembles

## Validation Strategy

The final modeling notebooks used:

- **4-fold expanding-window time-series cross-validation**
- chronological train/validation splits
- a separate held-out future test period

This design helped avoid leakage and better reflect real forecasting use.

## Champion Model

The final recommended model is:

**4-Model Stacking Ensemble with Logistic Regression Meta-Learner**

Base learners:
- CatBoost
- LightGBM
- ExtraTrees
- Logistic Regression

## Final Test Performance

Held-out test metrics for the champion model:

- **Macro Recall:** 0.5511
- **Class-2 Recall:** 0.6485
- **Class-2 Precision:** 0.0683
- **AP Class 2:** 0.1270
- **Macro MAP:** 0.4044
- **Probability RMSE:** 0.3900

## Key Findings

- The 4-model stack gave the best overall balance.
- The 3-model stack was a very close runner-up.
- CatBoost remained a strong single-model alternative.
- Neighbourhood identity, cyclical time, recent collisions, and weather all carried useful signal.
- Very high recall alone was not enough because false alarms reduce operational usefulness.

## Dashboard Deployment

The forecasting layer is deployed through a **Streamlit dashboard** with:

- live weather integration
- hotspot mapping across Toronto neighbourhoods
- ranked neighbourhood risk outputs
- short-horizon forecast views
- optional MILP-based patrol planning support

## Optimization Layer

The project also includes a **Mixed Integer Linear Programming (MILP)** module.

This is an added deployment-support feature that:
- converts forecasted risks into patrol-planning priorities
- supports constrained allocation of officers and patrol assets
- improves transparency and auditability
- acts as a decision-support tool, not an automated dispatch system

## Repository Structure

```text
toronto-collision-risk/
│
├── notebooks/
│   ├── 01_cleaning_and_merging.ipynb
│   ├── 02_baseline_supervised_builder.ipynb
│   ├── 03_lable_feature_engineering.ipynb
│   ├── 04_baseline_pre_feature_engineering.ipynb
│   ├── 05_baseline_post_feature_engineering.ipynb
│   ├── 06_baseline_random_forest.ipynb
│   ├── 07_best_models_catboost_lightgbm.ipynb
│   ├── 08_feature_importance.ipynb
│   ├── 09_feature_selection_retrain_smallK.ipynb
│   ├── 10_pca_benchmark_time_series_cv.ipynb
│   ├── 11_sampling_smotenc_time_series_cv.ipynb
│   ├── 12_proper_stacking_oof_time_series_cv.ipynb
│   ├── 13_stacking_3model_meta_comp.ipynb
│   ├── 14_stacking_4model_meta_comp.ipynb
│   └── 15_final_model_comparison.ipynb
│
├── app/
│   ├── dashboard_utils.py
│   └── streamlit_dashboard_openmeteo_live_with_milp.py
│
├── models/
├── data/
├── reports/
└── README.md









