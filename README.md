# Housing Affordability Stress in Greater Boston

## Overview
This project predicts housing affordability stress across Greater Boston ZIP
codes using Zillow home value and rent data combined with FRED macroeconomic
indicators. A ZIP code is considered high affordability stress in a given month if the
income needed to buy a home there puts it in the top 25% most expensive
ZIP-months across the dataset.

## Models
- **SARIMAX** (Python) — forecasts ZIP-level home values using mortgage rates
  as an exogenous variable. Fitted on six representative ZIP codes spanning all
  location tiers and price ranges.
- **Bayesian Regression** (R) — predicts income needed to afford housing using
  mortgage rates, unemployment, CPI, inventory, and location tier as features.
- **MLP Classifier** (Python) — manually implemented neural network that
  classifies ZIP-months as high or low affordability stress.

## Data Sources
- Zillow ZHVI and ZORI at ZIP code level
- Zillow metro-level inventory, days-to-pending, and income needed
- FRED: 30-year mortgage rate, national and Boston unemployment, CPI

## Structure
```
outputs/       preprocessed datasets and model outputs
notebooks/
  preprocessing/   shared and model-specific preprocessing notebooks
  modeling/        model training notebooks
  analysis/        diagnostic plots
```

## Run Order
```
notebooks/preprocessing/preprocess.ipynb
notebooks/preprocessing/preprocess_sarima.ipynb
notebooks/modeling/train_sarima.ipynb
```

## Dependencies
```
pip install pandas numpy statsmodels scikit-learn matplotlib
```
For Bayesian model: brms or rstanarm in R
