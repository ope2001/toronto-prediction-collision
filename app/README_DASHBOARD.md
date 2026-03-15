# Toronto Traffic Collision Prediction Dashboard

This dashboard predicts future collision risk for Toronto neighborhoods using a leakage-safe stacked ensemble model. It helps identify high-risk areas, visualize hotspots, and support patrol/resource planning.

## Features
- Neighborhood-level collision risk prediction
- Hotspot map and trend visualization
- Model performance display
- Decision-support for patrol prioritization

## Prediction Classes
- **0** = No collision
- **1** = One collision
- **2** = Two or more collisions

## Model
The dashboard uses a stacked ensemble model with multiple base learners and a meta-learner selected through validation.

## Evaluation
The model is evaluated using:
- **MAP**
- **Recall**
- **RMSE**

## Tech Stack
- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- CatBoost / LightGBM / other ensemble libraries used in the final notebook

## Run
```bash
git clone <your-repo-link>
cd <your-project-folder>
pip install -r requirements.txt
streamlit run dashboard/app.py