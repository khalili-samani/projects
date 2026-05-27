"""
Streamlit dashboard for NEM price forecasting.

Run from the project root:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

from src.config import DB_CONFIG
from src.db_utils import get_engine

MODEL_PATH = "models/xgboost_nem_price_model.joblib"
SPIKE_THRESHOLD_MWH = 300.0

st.set_page_config(
    page_title="NEM Price Forecasting",
    layout="wide",
)

st.title("🔌 Australian NEM Price Forecasting Dashboard")


@st.cache_data(ttl=300)
def fetch_feature_data() -> pd.DataFrame:
    query = """
        SELECT *
        FROM engineered_features
        ORDER BY settlement_date DESC
        LIMIT 3000;
    """
    df = pd.read_sql(query, get_engine())
    if not df.empty:
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        df.sort_values("settlement_date", inplace=True)
    return df


@st.cache_resource
def load_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


# ── Data loading ──────────────────────────────────────────────────────────────
data = fetch_feature_data()

if data.empty:
    st.warning(
        "No engineered feature data found in the database. "
        "Run `python run_pipeline.py` first."
    )
    st.stop()

bundle = load_model_bundle()
model = bundle["model"]
model_features = bundle["features"]
metrics = bundle["metrics"]

# ── Sidebar controls ──────────────────────────────────────────────────────────
regions = sorted(data["region_id"].dropna().unique())
selected_region = st.sidebar.selectbox("NEM Region", regions)

region_data = data[data["region_id"] == selected_region].copy()
latest_row = region_data.iloc[-1].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario Inputs")

temperature = st.sidebar.slider(
    "Temperature (°C)", 0.0, 50.0, float(latest_row["temperature_c"])
)
demand_lag_1h = st.sidebar.number_input(
    "Demand lag 1 h (MW)", value=float(latest_row["demand_lag_1h"])
)
price_lag_1h = st.sidebar.number_input(
    "Price lag 1 h ($/MWh)", value=float(latest_row["price_lag_1h"])
)

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader(f"{selected_region} — Market History")
    st.line_chart(
        region_data.set_index("settlement_date")[["demand_mw", "dispatch_price_mwh"]]
    )
    st.subheader("Recent Feature Rows")
    st.dataframe(region_data.tail(25), use_container_width=True)

with right_col:
    st.subheader("Forecast")
    st.caption(
        f"Hold-out MAE: **${metrics['mae']:.2f}/MWh** | "
        f"RMSE: **${metrics['rmse']:.2f}/MWh** | "
        f"R²: **{metrics['r2']:.4f}**"
    )

    scenario = latest_row.copy()
    scenario["temperature_c"] = temperature
    scenario["demand_lag_1h"] = demand_lag_1h
    scenario["price_lag_1h"] = price_lag_1h

    payload = pd.DataFrame([scenario[model_features].to_dict()])
    prediction = model.predict(payload)[0]

    st.metric("Forecast Price", f"${prediction:,.2f} /MWh")

    spike_status = "🔴 High" if prediction >= SPIKE_THRESHOLD_MWH else "🟢 Normal"
    st.metric(
        "Spike Risk",
        spike_status,
        help=f"Threshold: ${SPIKE_THRESHOLD_MWH:.0f}/MWh",
    )