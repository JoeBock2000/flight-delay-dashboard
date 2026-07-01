# Flight Delay: Factor Discovery and Prediction

An interactive dashboard that investigates the underlying factors driving US domestic flight delays, using a data mining approach following the CRISP-DM methodology. Prediction is included as a supporting component, but the central contribution is factor discovery: understanding what circumstances lead to disruption and why.

## Central finding

Flight disruption is multifactorial. No single factor dominates, and even the best tuned model reaches only a modest predictive ceiling (ROC-AUC around 0.71). This limited predictability is itself the key result: it motivates the shift from prediction toward discovering the contributing factors, which is where the analytical value lies.

## Overview

The application combines historical flight performance data with weather observations across ten US hub airports. Four discovery views (factors, patterns, mechanisms, and synthesis) present the drivers of disruption identified through three complementary methods, while a single prediction view estimates delay likelihood for a given flight, shown alongside an honest account of the model's limits.

## Methods

- Association rule mining to surface conditions that co-occur with disruption
- SHAP factor analysis to quantify which factors drive delays versus cancellations
- K-means clustering to identify distinct flight profiles
- Delay propagation, severity, and cross-airport efficiency analyses
- A six-model prediction comparison (logistic regression, decision tree, k-nearest neighbours, random forest, XGBoost, CatBoost), with SMOTE evaluated for its effect on class imbalance

## Data

- Flight performance: US Bureau of Transportation Statistics (BTS) on-time performance records
- Weather: NOAA observational data, joined to flights by station and date

The cleaned, merged dataset (929,899 flights) is included as flight_delay_clean.parquet.

## Model

The deployed prediction model is a gradient-boosted tree (XGBoost), serialised as xgboost_model.json. Its modest performance is expected and consistent with the project's central finding that delays are not sharply predictable from the available features.

## Requirements

- Python 3.10 or later
- Dependencies listed in requirements.txt

## Setup

    git clone https://github.com/JoeBock2000/flight-delay-dashboard.git
    cd flight-delay-dashboard
    python -m venv .venv
    source .venv/bin/activate   # on Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Running the app

    streamlit run app.py

The dashboard opens at http://localhost:8501.

## Files

| File | Description |
| --- | --- |
| app.py | Dashboard application with six views |
| requirements.txt | Python dependencies |
| flight_delay_clean.parquet | Cleaned, merged BTS and NOAA dataset |
| xgboost_model.json | Trained XGBoost prediction model |
| .gitignore | Ignored files and directories |

## Author

Joseph Bockarie
