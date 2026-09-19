# Dynamic Electricity Load Forecasting — MM26ML02

**Team:** MM2611  
**Track:** Machine Learning  
**Hackathon:** Modelling Minds 2.0 — VITMAS

## Problem

Forecast electricity demand for multiple nodes of a regional transmission system operator for the next **168 hours (7 days)**.

The required forecasts are:

- `nat_demand`
- `load_tocumen_mwh`
- `load_santiago_mwh`
- `load_david_mwh`

The solution focuses on accurate multi horizon forecasting while also exploring grid level operational optimization.

## Approach

The project follows a time aware machine learning pipeline:

1. Dataset understanding and temporal validation
2. Exploratory data analysis
3. Feature engineering
4. Comparison of XGBoost, LightGBM and CatBoost
5. CatBoost hyperparameter optimization
6. Peak load aware training
7. 50/50 ensemble of regular and peak weighted CatBoost
8. Grid dispatch optimization using MILP for the bonus objective

### Feature Engineering

The final model uses **110 features**, including:

- Cyclical time features
- Historical demand trends
- Moving average deviations and ratios
- Weather interactions
- Weekly demand stability features
- Net load and renewable generation proxies
- Peak period indicators

Temporal leakage was avoided by removing identifiers and inappropriate horizon information from the model inputs.

## Model

The final forecasting system uses:

**CatBoostRegressor**

with target specific tuned configurations.

Two model variants are trained:

- Regular CatBoost
- Peak weighted CatBoost

The final forecast is the **50/50 blend** of both predictions.

## Validation Results

Evaluated on 34 complete validation windows, each covering a 168-hour forecasting horizon.

| Target | MAE (MWh) | RMSE (MWh) | MAPE | R² |
|---|---:|---:|---:|---:|
| National Demand | 44.0213 | 62.2679 | 3.9216% | 0.8921 |
| Tocumen | 35.9187 | 50.9814 | 3.9094% | 0.8921 |
| Santiago | 2.8609 | 4.0285 | 3.9694% | 0.8913 |
| David | 5.1924 | 7.2996 | 3.9594% | 0.8924 |
| **Average** | **22.00** | **31.14** | **3.94%** | **0.8920** |

The final forecasting system uses a 50/50 blend of a regular CatBoost model and a peak-weighted CatBoost model. Peak weighting was introduced because peak-load periods showed substantially higher forecasting errors than non-peak periods. :contentReference[oaicite:1]{index=1}  

## Bonus: Grid Optimization

A MILP based dispatch optimization layer was built using the supplied generator and transmission constraints.

Across all **34 complete validation windows**:

- Feasible windows: **34/34**
- Minimum load served: **100%**
- Total unserved energy: **0 MWh**
- Thermal generation: effectively **0 MWh**
- Renewable curtailment: **~9.14 MWh**
- Thermal startups: **0**
- Total operating cost: **~$72.19M**

The tiny numerical negative thermal value observed in solver output is treated as zero within floating point solver tolerance.

The optimization respects generator capacity, ramping, transmission and minimum load service constraints.

## Project Structure

```text
Electricity-Forecasting/
│
├── data/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_model_optimization.ipynb
│   ├── 06_bonus_optimization.ipynb
│   └── 07_final_submission.ipynb
│
├── src/
│   └── features.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│       └── MM2611_MM26ML02.csv
│
├── README.md
├── requirements.txt
└── .gitignore
