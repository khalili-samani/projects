"""
Project-wide configuration.
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "nem_forecasting"),
}

FORECAST_TARGET = "target_price_mwh"

FORECAST_HORIZON_STEPS = 6   # 6 × 5-min intervals = 30 min ahead

RANDOM_SEED = 42

MIN_TRAINING_ROWS = 500

# ── AEMO data sources ─────────────────────────────────────────────────────────

AEMO_DISPATCHIS_URL = (
    "https://nemweb.com.au/Reports/CURRENT/DispatchIS_Reports/"
)

AEMO_ARCHIVE_BASE_URL = (
    "https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/"
)


def _default_archive_months(n=3):
    """Return the last n completed months as YYYYMM strings."""
    months = []
    cursor = datetime.today().replace(day=1)
    for _ in range(n):
        cursor = cursor - timedelta(days=1)
        months.append(cursor.strftime("%Y%m"))
        cursor = cursor.replace(day=1)
    return list(reversed(months))


# Override in .env: AEMO_ARCHIVE_MONTHS=202602,202603,202604
_env_months = os.getenv("AEMO_ARCHIVE_MONTHS")
AEMO_ARCHIVE_MONTHS = (
    [m.strip() for m in _env_months.split(",")]
    if _env_months
    else _default_archive_months(n=3)
)

# Days to sample per month from the archive.
# Each daily zip yields ~1,440 rows (288 intervals × 5 regions).
# 30 days × 3 months = ~130,000 rows — enough for solid training.
# Increase if you want more data; decrease to speed up the download.
AEMO_ARCHIVE_DAYS_PER_MONTH = int(os.getenv("AEMO_ARCHIVE_DAYS_PER_MONTH", 30))

# ── BOM weather stations ──────────────────────────────────────────────────────

BOM_STATIONS = {
    "VIC1": {
        "station_name": "Melbourne Olympic Park",
        "url": "https://www.bom.gov.au/fwo/IDV60901/IDV60901.95936.json",
    }
}

# How far apart a NEM interval and a weather observation can be and still be
# matched. BOM observations are hourly; NEM data is every 5 minutes.
# For historical ingestion the BOM feed only covers ~72 hours, so we use a
# generous tolerance and forward-fill the last known temperature.
WEATHER_MERGE_TOLERANCE = os.getenv("WEATHER_MERGE_TOLERANCE", "24h")

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_FEATURES = [
    "temperature_c",
    # Calendar features
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    # Cyclical encodings
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    # Demand lags
    "demand_lag_1h",
    "demand_lag_24h",
    # Price lags
    "price_lag_1h",
    "price_lag_24h",
    # Rolling means
    "demand_rolling_mean_4h",
    "price_rolling_mean_4h",
]

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
}