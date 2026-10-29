<p align="center">
  <img src="app/assets/logo.png" alt="SSGC logo" width="110">
</p>

# Gas Consumption Forecast - Next-Month Prediction

Predicts how much gas (**`U_SCM`**, in millions of SCM) will be consumed **next month** (July 2026), either for the whole company (billing month only) or for a chosen **Unit**, **Region** or **Zone**. Four models are compared (**Linear Regression, Random Forest, XGBoost, SVM**) and the results are shown in a Streamlit dashboard.

> **Data note:** the dataset covers 24 billing months (Jul 2024 - Jun 2026). Only the first 12 months are original. The last 12 are **synthetic** (copies of year 1, each value randomly changed by up to ±10%). The dataset's own README says it is for practice only, so results here are for learning and demonstration, not real-world planning.

## Quick start

Requires **Python 3.11 or newer**. Open a terminal inside this folder.

**1. Create and activate a virtual environment (once)**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
Mac / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install the libraries (once)**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app/streamlit_app.py
```
The trained models are already included, so the app works immediately. (If Streamlit asks for an email, just press Enter.)

If the saved models cannot be loaded on your computer (for example `XGBoostError: input stream corrupted`, which happens when library versions differ), the app **rebuilds them automatically** on first start - this takes a few seconds. You can also do it yourself at any time:
```bash
python src/train_models.py
```

**Re-running the full workflow (optional):** open the notebooks in VS Code (or Jupyter), select the `venv` kernel, and run them top to bottom in order: `01` then `02`. This recreates `data/processed/`, `models/` and `reports/`.

## Project structure

```
gas-consumption-forecast/
├── data/
│   ├── raw/                 original Excel file (never edited)
│   └── processed/           monthly totals: overall, by unit, by region, by zone
├── notebooks/
│   ├── 01_data_exploration_and_cleaning.ipynb
│   └── 02_model_development.ipynb
├── src/
│   ├── forecast.py          loads saved models and predicts next month (used by the app)
│   └── train_models.py      script version of notebook 02: trains and saves all models
├── models/                  trained models, one file per level (.pkl)
├── app/
│   ├── streamlit_app.py     the dashboard
│   └── assets/              logo and browser-tab icon
├── reports/
│   ├── model_comparison.csv test errors of every model at every level
│   └── figures/             charts created by the notebooks
├── .streamlit/config.toml   app theme
├── requirements.txt
└── README.md
```

## Workflow

| Step | Notebook | What happens |
|---|---|---|
| 1 | 01 | Load the data, read the dataset README, check quality (missing values, duplicates, constant columns, zeros, negatives) |
| 2 | 01 | Clean: remove 100 duplicate rows, drop `CUST_CL_CD` (always "IND"), convert `BILLING_MONTH` to a date |
| 3 | 01 | Explore: seasonality, year 1 vs year 2, shares by unit/region/zone |
| 4 | 01 | Aggregate `U_SCM` per month at four levels and save the tables |
| 5 | 02 | Features: month number (season), month counter (trend), one-hot Unit/Region/Zone |
| 6 | 02 | Split by time: train Jul 2024 - Dec 2025, test Jan - Jun 2026 |
| 7 | 02 | Train and compare 4 models per level, plus a "same month last year" reference |
| 8 | 02 | Retrain on all 24 months, save models, forecast next month |
| 9 | app | Dashboard: pick a level, group and model to see the forecast |

The four forecasting levels and their inputs:

| Level | Inputs |
|---|---|
| Overall | billing month only |
| Unit | billing month + Unit |
| Region | billing month + Region |
| Zone | billing month + Zone |

## Results (test = last 6 months)

| Level | Best model | Avg. error (M SCM) | WAPE | Reference "same month last year" |
|---|---|---|---|---|
| Overall | XGBoost | 3.11 | 1.1% | 0.5% |
| Unit | XGBoost | 1.69 | 1.7% | 0.9% |
| Region | XGBoost | 1.23 | 3.8% | 1.0% |
| Zone | Random Forest | 0.50 | 8.4% | 1.4% |

**Forecast for July 2026 (overall): about 397.6 million SCM.** Full results for all models: `reports/model_comparison.csv`.

## Limitations

- **Synthetic data:** year 2 is a noisy copy of year 1, so the simple reference "same month last year" beats every model. This is expected here. Do not present these scores as proof of real-world accuracy.
- **Only 24 months:** the models learn the seasonal pattern, but not long-term trends, weather effects or business changes.
- **Zone level is the weakest:** 34 of the 49 zones each hold less than 0.5% of total gas and are noisy.
- Only the requested inputs (month, unit, region, zone) are used.
