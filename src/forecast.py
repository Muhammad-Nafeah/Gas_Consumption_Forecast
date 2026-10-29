"""
forecast.py
-----------
Helper functions used by the Streamlit app.

It loads the models that were trained and saved in notebooks/02_model_development.ipynb
and predicts the gas consumption (U_SCM, in millions of SCM) for the month right after
the last month of data (July 2026).

The feature logic here is the same as "Step 4" and "Step 11" of notebook 02.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# ------------------------------------------------------------------ constants
# level -> the column that defines the group (None = whole company, month only)
LEVELS = {"overall": None, "unit": "UNIT", "region": "REGION", "zone": "ZONE"}
LEVEL_LABELS = {"overall": "Overall", "unit": "Unit", "region": "Region", "zone": "Zone"}
HISTORY_FILES = {
    "overall": "monthly_overall.csv",
    "unit": "monthly_by_unit.csv",
    "region": "monthly_by_region.csv",
    "zone": "monthly_by_zone.csv",
}
REFERENCE_MODEL = "Reference: same month last year"


# ------------------------------------------------------------------ loading
@lru_cache(maxsize=None)
def load_bundle(level: str) -> dict:
    """Load the saved models (and their settings) for one level."""
    return joblib.load(MODELS_DIR / f"{level}_models.pkl")


def ensure_models() -> bool:
    """
    Make sure the saved models can be loaded on THIS computer.

    Saved models only load reliably with the same library versions that created them
    (XGBoost is strict about this). If any file cannot be loaded, we simply retrain all
    models from the data in /data/processed (takes a few seconds) and save them again.
    Returns True if models had to be rebuilt.
    """
    try:
        for level in LEVELS:
            load_bundle(level)
        return False
    except Exception:
        from . import train_models          # imported here so normal runs stay fast

        load_bundle.cache_clear()
        train_models.train_all()
        for level in LEVELS:                # check that the fresh files load
            load_bundle(level)
        return True


@lru_cache(maxsize=None)
def load_history_table(level: str) -> pd.DataFrame:
    """Monthly totals of U_SCM (millions of SCM) for one level."""
    return pd.read_csv(PROCESSED_DIR / HISTORY_FILES[level], parse_dates=["DATE"])


def load_comparison() -> pd.DataFrame:
    """Test-set errors of every model (created in notebook 02)."""
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


def model_names(level: str) -> list:
    return list(load_bundle(level)["models"].keys())


def best_model_name(level: str) -> str:
    return load_bundle(level)["best_model"]


def get_groups(level: str) -> list:
    """Groups of a level, biggest gas consumers first (so the default choice is a big one)."""
    group_col = LEVELS[level]
    if group_col is None:
        return []
    totals = load_history_table(level).groupby(group_col)["U_SCM"].sum()
    return totals.sort_values(ascending=False).index.tolist()


def get_history(level: str, group: str | None = None) -> pd.DataFrame:
    """Past monthly gas consumption for the whole company or one group."""
    table = load_history_table(level)
    group_col = LEVELS[level]
    if group_col is not None and group is not None:
        table = table[table[group_col] == group]
    return table[["DATE", "U_SCM"]].sort_values("DATE").reset_index(drop=True)


def get_next_month(level: str) -> pd.Timestamp:
    """The month we forecast: one month after the last month of data."""
    return load_bundle(level)["last_date"] + pd.DateOffset(months=1)


# ------------------------------------------------------------------ prediction
def _make_row(bundle: dict, level: str, group: str | None, date: pd.Timestamp) -> pd.DataFrame:
    """One row with exactly the columns the models were trained on."""
    start = bundle["start_date"]
    row = {col: 0.0 for col in bundle["feature_columns"]}
    row["month_num"] = date.month
    row["time_idx"] = (date.year - start.year) * 12 + (date.month - start.month) + 1
    if group is not None:
        row[f"{LEVELS[level]}_{group}"] = 1.0     # switch on the chosen unit / region / zone
    return pd.DataFrame([row])[bundle["feature_columns"]]


def predict_next_month(level: str, group: str | None = None, model_name: str | None = None) -> dict:
    """
    Forecast next month for one level/group.
    Returns the prediction of the chosen model AND of every other model (in millions of SCM).
    """
    bundle = load_bundle(level)
    date = get_next_month(level)
    X = _make_row(bundle, level, group, date)

    all_predictions = {
        name: max(float(model.predict(X)[0]), 0.0)      # gas use cannot be negative
        for name, model in bundle["models"].items()
    }
    chosen = model_name or bundle["best_model"]
    return {
        "month": date,
        "level": level,
        "group": group or "Whole company",
        "model_name": chosen,
        "best_model": bundle["best_model"],
        "prediction": all_predictions[chosen],
        "all_predictions": all_predictions,
    }


def forecast_all_groups(level: str, model_name: str | None = None) -> pd.DataFrame:
    """Next-month forecast for every unit / region / zone, using ONE model."""
    bundle = load_bundle(level)
    chosen = model_name or bundle["best_model"]
    model = bundle["models"][chosen]
    date = get_next_month(level)

    rows = []
    for group in bundle["groups"]:
        X = _make_row(bundle, level, group, date)
        rows.append({"Group": group, "Forecast": max(float(model.predict(X)[0]), 0.0)})
    out = pd.DataFrame(rows).sort_values("Forecast", ascending=False).reset_index(drop=True)
    total = out["Forecast"].sum()
    out["Share_%"] = out["Forecast"] / total * 100 if total > 0 else 0.0
    return out
