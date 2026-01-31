# Toronto Traffic Collision Risk Prediction

## Project Overview

Spatiotemporal machine learning forecasting system to predict traffic collision risk hotspots in Toronto using weather conditions and historical crash patterns.

**Team Members:**
- Adeyi Fridaus (NF1004424)
- Jigme Jigme (NF1000447)
- Jigme Yeshi (NF1004171)
- Jambayang Singye (NF1002341)

Supervisor: Dr. Hany Osman  
Course: DAMO-699-2 - Winter 2026 Capstone Project  
Institution: University of Niagara Falls, Canada

#Problem Type
Spatiotemporal Regression - Predicting the number of traffic collisions for Toronto neighbourhoods.

Models
1. Baseline: Seasonal Historical Average
2. Random Forest Regressor
3. XGBoost Regressor
4. LSTM Neural Network

## Data Sources
- Toronto Police Service Traffic Collisions (2016-2024)
- Environment Canada Weather Data (2016-2024)

## Research Questions
1. Which weather variables and temporal factors are most strongly associated with collision counts?
2. Can ML models reliably predict collision counts 1-7 days ahead?
3. Which neighbourhoods show persistent vs weather-triggered risk?

## Hypotheses

### H1: Weather Impact
- H0: Mean collision count does not differ between adverse and clear weather
- H1: Mean collision count is significantly higher during adverse weather (p < 0.05)
- Test: ANOVA

### H2: Model Performance
- H0: ML model does not achieve lower RMSE than baseline
- H1: ML model achieves >=20% lower RMSE than baseline
- Test: Diebold-Mariano test

### H3: Hotspot Dynamics
- H0: Hotspot neighbourhoods remain stationary across weather conditions
- H1: >=30% of top-10% neighborhoods change between clear and adverse weather
- Test: Jaccard similarity index

## Project Timeline
- Week 1-2: Data Collection
- Week 3: Data Preprocessing & Baseline
- Week 4-7: Model Training (RF, XGBoost, LSTM)
- Week 8-9: Evaluation
- Week 10: Interpretation
- Week 11-12: Web Application
- Week 13-14: Final Report

## Contact
Supervisor:hany.osman@unfc.ca
GitHub:https://github.com/ope2001/toronto-collision-prediction