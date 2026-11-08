"""
Feature engineering for leakage-safe NEM price forecasting.

All lags are shifted by at least FORECAST_HORIZON_STEPS to ensure no
future information leaks into the training features.
"""

import numpy as np
import pandas as pd
import mysql.connector

from src.config import DB_CONFIG, FORECAST_HORIZON_STEPS
from src.db_utils import get_engine


def run_feature_pipeline():
    print("[Features] Building forecasting feature matrix...")

    query = """
        SELECT settlement_date, region_id, demand_mw, dispatch_price_mwh, temperature_c
        FROM cleaned_nem_weather_data
        ORDER BY region_id, settlement_date;
    """

    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(query, get_engine())

    if df.empty:
        print("[Warning] cleaned_nem_weather_data is empty — skipping features.")
        conn.close()
        return

    df["settlement_date"] = pd.to_datetime(df["settlement_date"])

    processed_groups = []

    for _region_id, group in df.groupby("region_id"):
        g = group.copy().set_index("settlement_date").sort_index()

        # Calendar features
        g["hour"] = g.index.hour
        g["day_of_week"] = g.index.dayofweek
        g["month"] = g.index.month
        g["is_weekend"] = g["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encodings — prevent the model treating hour 23 as far from hour 0
        g["hour_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
        g["hour_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
        g["day_sin"] = np.sin(2 * np.pi * g["day_of_week"] / 7)
        g["day_cos"] = np.cos(2 * np.pi * g["day_of_week"] / 7)

        # Lag features (shift=2 → 1 h ago at 30-min resolution; leakage-safe)
        lag_1h = FORECAST_HORIZON_STEPS
        lag_24h = 48  # 24 h at 30-min resolution

        g["demand_lag_1h"] = g["demand_mw"].shift(lag_1h)
        g["demand_lag_24h"] = g["demand_mw"].shift(lag_24h)
        g["price_lag_1h"] = g["dispatch_price_mwh"].shift(lag_1h)
        g["price_lag_24h"] = g["dispatch_price_mwh"].shift(lag_24h)

        # Rolling means (also shifted to avoid leakage)
        g["demand_rolling_mean_4h"] = (
            g["demand_mw"].shift(lag_1h).rolling(window=8, min_periods=8).mean()
        )
        g["price_rolling_mean_4h"] = (
            g["dispatch_price_mwh"].shift(lag_1h).rolling(window=8, min_periods=8).mean()
        )

        # Target: price FORECAST_HORIZON_STEPS intervals ahead
        g["target_price_mwh"] = g["dispatch_price_mwh"].shift(-FORECAST_HORIZON_STEPS)

        processed_groups.append(g.reset_index())

    final_df = pd.concat(processed_groups, ignore_index=True)
    final_df.dropna(inplace=True)

    insert_query = """
        INSERT INTO engineered_features (
            settlement_date, region_id,
            demand_mw, dispatch_price_mwh, target_price_mwh,
            temperature_c,
            hour, day_of_week, month, is_weekend,
            hour_sin, hour_cos, day_sin, day_cos,
            demand_lag_1h, demand_lag_24h,
            price_lag_1h, price_lag_24h,
            demand_rolling_mean_4h, price_rolling_mean_4h
        )
        VALUES (
            %s, %s,
            %s, %s, %s,
            %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s
        )
        ON DUPLICATE KEY UPDATE
            target_price_mwh = VALUES(target_price_mwh);
    """

    cols = [
        "settlement_date", "region_id",
        "demand_mw", "dispatch_price_mwh", "target_price_mwh",
        "temperature_c",
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "demand_lag_1h", "demand_lag_24h",
        "price_lag_1h", "price_lag_24h",
        "demand_rolling_mean_4h", "price_rolling_mean_4h",
    ]

    rows = list(final_df[cols].itertuples(index=False, name=None))

    cursor = conn.cursor()
    cursor.executemany(insert_query, rows)
    conn.commit()
    print(f"[Features] Upserted {cursor.rowcount} feature rows.")
    cursor.close()
    conn.close()