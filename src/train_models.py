"""
train_models.py
---------------
Script version of notebook 02 (Steps 4 - 10): builds the features, trains and compares
the four models at every level, retrains them on all 24 months and saves them.

The app calls this automatically if the saved models in /models cannot be loaded
(for example because they were saved with a different version of XGBoost / scikit-learn).
You can also run it yourself:

    python src/train_models.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# ------------------------------------------------------------------ settings (same as notebook 02)
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LEVELS = {"overall": None, "unit": "UNIT", "region": "REGION", "zone": "ZONE"}
FILE_NAMES = {
    "overall": "monthly_overall.csv",
    "unit": "monthly_by_unit.csv",
    "region": "monthly_by_region.csv",
    "zone": "monthly_by_zone.csv",
}
TARGET = "U_SCM"
TEST_MONTHS = 6
SEED = 42
START_DATE = pd.Timestamp("2024-07-01")
REFERENCE = "Reference: same month last year"


# ------------------------------------------------------------------ Step 4: features
def build_features(table, level, feature_columns=None):
    """Turn a monthly table into the numbers (X) the models learn from."""
    X = pd.DataFrame({
        "month_num": table["DATE"].dt.month,
        "time_idx": (table["DATE"].dt.year - START_DATE.year) * 12
                    + (table["DATE"].dt.month - START_DATE.month) + 1,
    })
    group_col = LEVELS[level]
    if group_col is not None:
        dummies = pd.get_dummies(table[group_col], prefix=group_col, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X


# ------------------------------------------------------------------ Step 5: time split
def time_split(table, test_months=TEST_MONTHS):
    dates = sorted(table["DATE"].unique())
    cutoff = dates[-test_months]
    return table[table["DATE"] < cutoff], table[table["DATE"] >= cutoff]


# ------------------------------------------------------------------ Step 6: models
def make_models(feature_columns):
    """Fresh, untrained models. The SVM gets no time_idx, scaled inputs and a scaled target."""
    columns_for_svm = [c for c in feature_columns if c != "time_idx"]
    svm = TransformedTargetRegressor(
        regressor=Pipeline([
            ("select_columns", ColumnTransformer([("keep", "passthrough", columns_for_svm)])),
            ("scale_inputs", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100, epsilon=0.01)),
        ]),
        transformer=StandardScaler(),
    )
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED),
        "SVM (SVR)": svm,
    }


# ------------------------------------------------------------------ Step 7: error measures
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wape = float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum() * 100)
    return {"MAE": mae, "RMSE": rmse, "WAPE_%": wape}


def same_month_last_year(table, test, level):
    keys = ["DATE"] + ([LEVELS[level]] if LEVELS[level] else [])
    last_year = table[keys + [TARGET]].copy()
    last_year["DATE"] = last_year["DATE"] + pd.DateOffset(years=1)
    merged = test[keys].merge(last_year, on=keys, how="left")
    return merged[TARGET].fillna(0).to_numpy()


# ------------------------------------------------------------------ Steps 8 - 10
def train_all(verbose=False):
    """Evaluate every model at every level, then retrain on all months and save everything."""
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    monthly = {level: pd.read_csv(PROCESSED_DIR / name, parse_dates=["DATE"])
               for level, name in FILE_NAMES.items()}

    # Step 8: train on the early months, score on the last 6 months
    results = []
    for level in LEVELS:
        table = monthly[level]
        train, test = time_split(table)
        X_train = build_features(train, level)
        feature_columns = list(X_train.columns)
        X_test = build_features(test, level, feature_columns)
        y_test = test[TARGET].to_numpy()

        for name, model in make_models(feature_columns).items():
            model.fit(X_train, train[TARGET])
            results.append({"level": level, "model": name, **evaluate(y_test, model.predict(X_test))})
        results.append({"level": level, "model": REFERENCE,
                        **evaluate(y_test, same_month_last_year(table, test, level))})

    comparison = pd.DataFrame(results)
    real = comparison[comparison["model"] != REFERENCE]
    best_models = real.sort_values("MAE").groupby("level").first()["model"].to_dict()

    # Step 10: retrain on all 24 months and save one file per level
    for level, group_col in LEVELS.items():
        table = monthly[level]
        X_all = build_features(table, level)
        feature_columns = list(X_all.columns)

        fitted = {}
        for name, model in make_models(feature_columns).items():
            model.fit(X_all, table[TARGET])
            fitted[name] = model

        bundle = {
            "level": level,
            "models": fitted,
            "best_model": best_models[level],
            "feature_columns": feature_columns,
            "groups": sorted(table[group_col].unique()) if group_col else [],
            "last_date": table["DATE"].max(),
            "start_date": START_DATE,
        }
        joblib.dump(bundle, MODELS_DIR / f"{level}_models.pkl", compress=3)
        if verbose:
            print(f"saved {level}_models.pkl (best model: {best_models[level]})")

    comparison.round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    if verbose:
        print("saved model_comparison.csv")


if __name__ == "__main__":
    train_all(verbose=True)
