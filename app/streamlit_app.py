"""
streamlit_app.py  -  the user interface

Run from the project folder:
    streamlit run app/streamlit_app.py

The app loads the models trained in notebooks/02_model_development.ipynb and shows the
forecast of next month's gas consumption (U_SCM) overall, or by Unit, Region or Zone.
"""
import sys
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

# make the project root importable so that "from src import forecast" works
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "assets"       # logo and browser-tab icon
sys.path.insert(0, str(ROOT))
from src import forecast as fc  # noqa: E402

# ------------------------------------------------------------------ look & feel
ORANGE = "#E8590C"      # forecast / accent colour
BLUE = "#2563EB"        # actual (past) data
GREY = "#9CA3AF"
LIGHT_BLUE = "#93C5FD"
SYNTHETIC_START = pd.Timestamp("2025-07-01")   # first synthetic month in the data
AUTO_MODEL = "Best model (auto)"

LEVEL_OPTIONS = {
    "Overall (billing month only)": "overall",
    "By Unit": "unit",
    "By Region": "region",
    "By Zone": "zone",
}

st.set_page_config(page_title="Gas Consumption Forecast", page_icon=Image.open(ASSETS / "favicon.png"), layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1250px;}

    .hero-title {font-size: 2.1rem; font-weight: 800; color: #111827; line-height: 1.15;}
    .hero-sub   {color: #6B7280; font-size: 1.02rem; margin: 6px 0 22px 0;}

    .kpi-card {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
               padding: 18px 22px; box-shadow: 0 1px 3px rgba(16,24,40,.06); min-height: 138px;}
    .kpi-card.accent {background: linear-gradient(135deg, #E8590C 0%, #F59F00 100%);
                      border: none; box-shadow: 0 6px 16px rgba(232,89,12,.28);}
    .kpi-label {font-size: .74rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: #6B7280;}
    .kpi-value {font-size: 2.1rem; font-weight: 800; color: #111827; line-height: 1.2; margin-top: 6px;}
    .kpi-unit  {font-size: .95rem; font-weight: 600; opacity: .75; margin-left: 4px;}
    .kpi-sub   {font-size: .84rem; color: #6B7280; margin-top: 4px;}
    .accent .kpi-label, .accent .kpi-value, .accent .kpi-sub {color: #FFFFFF;}
    .accent .kpi-label, .accent .kpi-sub {opacity: .92;}

    .side-title {text-align: center; font-weight: 800; font-size: 1.02rem; color: #111827; margin-top: 8px;}
    .side-sub   {text-align: center; color: #6B7280; font-size: .82rem; margin-bottom: 2px;}

    .section-title {font-size: 1.05rem; font-weight: 700; color: #111827; margin: 18px 0 4px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ small helpers
def fmt(value: float) -> str:
    """Show big numbers with 1 decimal and small ones with more, so tiny zones are readable."""
    if value >= 100:
        return f"{value:,.1f}"
    if value >= 1:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def pct_change(new: float, old: float | None):
    if old is None or abs(old) < 1e-9:
        return None
    return (new - old) / abs(old) * 100


def delta_text(change) -> str:
    if change is None:
        return "n/a"
    return f"{'▲' if change >= 0 else '▼'} {abs(change):.1f}%"


def kpi_card(label: str, value: str, sub: str, unit: str = "", accent: bool = False) -> None:
    css = "kpi-card accent" if accent else "kpi-card"
    unit_html = f'<span class="kpi-unit">{escape(unit)}</span>' if unit else ""
    st.markdown(
        f'<div class="{css}"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}{unit_html}</div>'
        f'<div class="kpi-sub">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Preparing the models for this computer (first start only, a few seconds)...")
def prepare_models() -> bool:
    """Loads the saved models; if they do not match this computer's libraries they are rebuilt."""
    return fc.ensure_models()


prepare_models()


# cached wrappers, so the app stays fast when the user clicks around
@st.cache_data(show_spinner=False)
def get_prediction(level, group, model_name):
    return fc.predict_next_month(level, group, model_name)


@st.cache_data(show_spinner=False)
def get_history(level, group):
    return fc.get_history(level, group)


@st.cache_data(show_spinner="Forecasting every group...")
def get_all_groups(level, model_name):
    return fc.forecast_all_groups(level, model_name)


@st.cache_data(show_spinner=False)
def get_comparison():
    return fc.load_comparison()


def forecast_chart(history: pd.DataFrame, next_month: pd.Timestamp, value: float) -> alt.Chart:
    """Past months (blue) + the forecast for next month (orange)."""
    hist = history.rename(columns={"DATE": "Month", "U_SCM": "Million SCM"})
    last = hist.iloc[-1]
    link = pd.DataFrame({"Month": [last["Month"], next_month], "Million SCM": [last["Million SCM"], value]})
    point = pd.DataFrame({"Month": [next_month], "Million SCM": [value], "What": ["Forecast"]})
    marker = pd.DataFrame({"Month": [SYNTHETIC_START], "Note": ["synthetic data starts →"]})

    x = alt.X("Month:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=-40))
    y = alt.Y("Million SCM:Q", title="Million SCM", scale=alt.Scale(zero=False))

    actual = alt.Chart(hist).mark_line(
        color=BLUE, strokeWidth=2.5, point=alt.OverlayMarkDef(color=BLUE, size=55)
    ).encode(
        x=x, y=y,
        tooltip=[alt.Tooltip("Month:T", format="%B %Y"), alt.Tooltip("Million SCM:Q", format=",.3f")],
    )
    connector = alt.Chart(link).mark_line(color=ORANGE, strokeDash=[6, 4], strokeWidth=2.5).encode(x=x, y=y)
    forecast = alt.Chart(point).mark_point(color=ORANGE, filled=True, size=240).encode(
        x=x, y=y,
        tooltip=[alt.Tooltip("What:N", title=""), alt.Tooltip("Month:T", format="%B %Y"),
                 alt.Tooltip("Million SCM:Q", format=",.3f")],
    )
    rule = alt.Chart(marker).mark_rule(color=GREY, strokeDash=[4, 4]).encode(x=x)
    rule_text = alt.Chart(marker).mark_text(align="left", dx=6, color=GREY, fontSize=11).encode(
        x=x, y=alt.value(10), text="Note:N"
    )
    return alt.layer(actual, connector, forecast, rule, rule_text).properties(height=390)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    # logo, centred (the two empty side columns act as margins)
    _, logo_col, _ = st.columns([1, 2, 1])
    logo_col.image(str(ASSETS / "logo.png"), width="stretch")
    st.markdown('<div class="side-title">Gas Consumption Forecast</div>'
                '<div class="side-sub">Next-month gas volume (U_SCM)</div>', unsafe_allow_html=True)
    st.divider()

    level_label = st.radio(
        "Forecast level",
        list(LEVEL_OPTIONS),
        help="Overall uses only the billing month. The other levels also use the Unit, Region or Zone.",
    )
    level = LEVEL_OPTIONS[level_label]

    group = None
    if level != "overall":
        group = st.selectbox(
            fc.LEVEL_LABELS[level],
            fc.get_groups(level),
            key=f"group_{level}",
            help="Listed from the biggest gas consumer to the smallest.",
        )

    model_choice = st.selectbox(
        "Model",
        [AUTO_MODEL] + fc.model_names(level),
        help="'Best model' is the one with the lowest error on the held-out test months.",
    )
    model_name = None if model_choice == AUTO_MODEL else model_choice


# ------------------------------------------------------------------ prediction
result = get_prediction(level, group, model_name)
history = get_history(level, group)
next_month = result["month"]
prediction = result["prediction"]
group_name = result["group"]

last_row = history.iloc[-1]
last_year_date = next_month - pd.DateOffset(years=1)
last_year_rows = history[history["DATE"] == last_year_date]
last_year_value = float(last_year_rows["U_SCM"].iloc[0]) if len(last_year_rows) else None

comparison = get_comparison()
this_level = comparison[comparison["level"] == level]
used_model = this_level[this_level["model"] == result["model_name"]]

# ------------------------------------------------------------------ header + KPI cards
st.markdown(
    f'<div class="hero-title">Next-Month Gas Consumption Forecast</div>'
    f'<div class="hero-sub">{escape(group_name)} &nbsp;·&nbsp; {fc.LEVEL_LABELS[level]} level '
    f'&nbsp;·&nbsp; forecast for <b>{next_month:%B %Y}</b></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card(f"Predicted gas · {next_month:%b %Y}", fmt(prediction),
             f"Model: {result['model_name']}" + (" (best)" if result["model_name"] == result["best_model"] else ""),
             unit="M SCM", accent=True)
with c2:
    kpi_card(f"vs last month ({last_row['DATE']:%b %Y})",
             delta_text(pct_change(prediction, float(last_row["U_SCM"]))),
             f"Last month: {fmt(float(last_row['U_SCM']))} M SCM")
with c3:
    kpi_card(f"vs same month last year ({last_year_date:%b %Y})",
             delta_text(pct_change(prediction, last_year_value)),
             f"{last_year_date:%b %Y}: {fmt(last_year_value)} M SCM" if last_year_value is not None else "no data")
with c4:
    if len(used_model):
        row = used_model.iloc[0]
        kpi_card("Model test error (WAPE)", f"{row['WAPE_%']:.1f}%",
                 f"Avg. error {row['MAE']:.2f} M SCM · {fc.LEVEL_LABELS[level]} level · Jan–Jun 2026")
    else:
        kpi_card("Model test error (WAPE)", "n/a", "no test result")

st.write("")

# ------------------------------------------------------------------ tabs
tab_forecast, tab_groups, tab_models, tab_about = st.tabs(
    ["Forecast", "Compare groups", "Model performance", "About"]
)

# ---- Tab 1: forecast chart + what every model predicts
with tab_forecast:
    st.markdown(f'<div class="section-title">{escape(group_name)}: actual gas consumption and next-month forecast</div>',
                unsafe_allow_html=True)
    st.altair_chart(forecast_chart(history, next_month, prediction), width="stretch")
    st.caption("Blue = actual past months · Orange = forecast. Hover over the points for exact values (million SCM).")

    st.markdown('<div class="section-title">What each model predicts for this selection</div>', unsafe_allow_html=True)
    rows = []
    for name, value in result["all_predictions"].items():
        tags = []
        if name == result["best_model"]:
            tags.append("Best on test data")
        if name == result["model_name"]:
            tags.append("shown above")
        rows.append({"Model": name, "Forecast (M SCM)": value, "": " · ".join(tags)})
    model_table = pd.DataFrame(rows).sort_values("Forecast (M SCM)", ascending=False)
    st.dataframe(
        model_table, hide_index=True, width="stretch",
        column_config={"Forecast (M SCM)": st.column_config.NumberColumn(format="%.3f")},
    )

# ---- Tab 2: compare all groups of the chosen level
with tab_groups:
    if level == "overall":
        st.info("The overall forecast is a single number for the whole company. "
                "Choose **By Unit**, **By Region** or **By Zone** in the sidebar to compare groups.")
    else:
        groups_df = get_all_groups(level, model_name).copy()
        groups_df["Selected"] = groups_df["Group"] == group
        st.markdown(
            f'<div class="section-title">Forecast for {next_month:%B %Y} - every {fc.LEVEL_LABELS[level].lower()} '
            f'({result["model_name"]})</div>', unsafe_allow_html=True)

        bars = alt.Chart(groups_df).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Group:N", sort="-x", title=None),
            x=alt.X("Forecast:Q", title="Forecast (million SCM)"),
            color=alt.Color("Selected:N", legend=None,
                            scale=alt.Scale(domain=[True, False], range=[ORANGE, "#CBD5E1"])),
            tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Forecast:Q", format=",.3f"),
                     alt.Tooltip("Share_%:Q", format=".1f", title="Share of total (%)")],
        ).properties(height=max(220, 28 * len(groups_df)))
        st.altair_chart(bars, width="stretch")
        st.caption(f"Orange = the {fc.LEVEL_LABELS[level].lower()} selected in the sidebar.")

        st.dataframe(
            groups_df.drop(columns="Selected").rename(
                columns={"Forecast": "Forecast (M SCM)", "Share_%": "Share of total (%)"}),
            hide_index=True, width="stretch",
            column_config={
                "Forecast (M SCM)": st.column_config.NumberColumn(format="%.3f"),
                "Share of total (%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

# ---- Tab 3: how good were the models?
with tab_models:
    st.markdown(f'<div class="section-title">{fc.LEVEL_LABELS[level]} level: model errors on the held-out test months '
                f'(Jan - Jun 2026)</div>', unsafe_allow_html=True)

    perf = this_level.copy()
    perf["Kind"] = perf["model"].apply(
        lambda m: "Reference" if m == fc.REFERENCE_MODEL else ("Best model" if m == result["best_model"] else "Other models")
    )
    perf_chart = alt.Chart(perf).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("model:N", sort="x", title=None),
        x=alt.X("WAPE_%:Q", title="WAPE (%)  -  lower is better"),
        color=alt.Color("Kind:N", legend=alt.Legend(orient="bottom", title=None),
                        scale=alt.Scale(domain=["Best model", "Other models", "Reference"],
                                        range=[ORANGE, LIGHT_BLUE, GREY])),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("MAE:Q", format=".3f"),
                 alt.Tooltip("RMSE:Q", format=".3f"), alt.Tooltip("WAPE_%:Q", format=".2f", title="WAPE (%)")],
    ).properties(height=260)
    st.altair_chart(perf_chart, width="stretch")

    table = perf.sort_values("MAE")[["model", "MAE", "RMSE", "WAPE_%"]].rename(
        columns={"model": "Model", "MAE": "MAE (M SCM)", "RMSE": "RMSE (M SCM)", "WAPE_%": "WAPE (%)"})
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c != "Model"},
    )

    reference_row = perf[perf["model"] == fc.REFERENCE_MODEL]
    best_row = perf[perf["model"] == result["best_model"]]
    if len(reference_row) and len(best_row) and reference_row["MAE"].iloc[0] < best_row["MAE"].iloc[0]:
        st.warning(
            "The grey **reference** (\"same month last year\") scores better than every model. "
            "That is expected here: the second year of the data is a noisy copy of the first, so repeating "
            "last year is almost always right. It would not be true for real data."
        )

    with st.expander("What do MAE, RMSE and WAPE mean?"):
        st.markdown(
            "- **MAE**: the average size of the mistake, in million SCM.\n"
            "- **RMSE**: like MAE, but big mistakes count more.\n"
            "- **WAPE**: total mistake as a percentage of the total gas used. Easy to compare across levels.\n\n"
            "Lower is better for all three."
        )

# ---- Tab 4: about
with tab_about:
    st.markdown(
        """
### What this app does
It predicts **how much gas (U_SCM, in million SCM) will be consumed next month**, either for the whole
company (using only the billing month) or for a chosen **Unit**, **Region** or **Zone**.

### The data
* 24 billing months, **Jul 2024 - Jun 2026**, from the *Sales Data Sample* workbook.
* The first 12 months are original. The last 12 months are **synthetic**: copies of the first year with every
  value randomly changed by up to ±10%. The dataset's own README says it should be used for practice only.

### How the forecasts were built
1. **Cleaning:** removed exact duplicate rows and a column that never changes; converted the billing month to a date.
2. **Aggregation:** added up U_SCM for each month, and for each Unit, Region and Zone.
3. **Features:** month number (season), a month counter (trend), and the Unit / Region / Zone as 0/1 columns.
4. **Models:** Linear Regression, Random Forest, XGBoost and SVM, compared on the last 6 months, which were hidden during training.
5. **Final models:** retrained on all 24 months. The forecast is for the month after the last one in the data.

### Limitations
* Because half the data is synthetic, accuracy scores look better than they would on real data.
* With only 24 months, the models learn the **seasonal pattern** but not long-term trends, weather or business changes.
* Zone-level forecasts are the least reliable: most zones are small and noisy.
        """
    )
