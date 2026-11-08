"""
Cleaning and synchronisation of raw NEM and BOM datasets.

The BOM live feed only covers ~72 hours of observations. For historical NEM
data that falls outside this window, we fill missing temperature using
Melbourne monthly climate averages so no rows are dropped.
"""

import math
import pandas as pd
import mysql.connector

from src.config import DB_CONFIG, WEATHER_MERGE_TOLERANCE
from src.db_utils import get_engine

PRICE_FLOOR_MWH = -1000.0
PRICE_CAP_MWH   = 20_000.0

# Melbourne long-term monthly mean temperatures (°C) — Bureau of Meteorology.
# Used as fallback when no BOM observation is available for a given row.
MELBOURNE_MONTHLY_MEAN_TEMP = {
    1: 25.9, 2: 25.8, 3: 23.2, 4: 19.8,
    5: 16.0, 6: 13.1, 7: 12.4, 8: 13.7,
    9: 15.8, 10: 18.5, 11: 21.3, 12: 23.8,
}


def _safe(value):
    """Convert float nan to None so MySQL writes NULL instead of 'nan'."""
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def clean_raw_observations():
    print("[Cleaning] Aligning raw NEM and weather observations...")

    nem_query = """
        SELECT settlement_date, region_id, demand_mw, dispatch_price_mwh
        FROM raw_nem_data
        ORDER BY region_id, settlement_date;
    """

    weather_query = """
        SELECT
            w.observation_date,
            w.station_name,
            m.region_id,
            w.temperature_c
        FROM raw_weather_data w
        JOIN weather_station_region_map m ON w.station_name = m.station_name
        ORDER BY m.region_id, w.observation_date;
    """

    insert_query = """
        INSERT INTO cleaned_nem_weather_data (
            settlement_date, region_id,
            demand_mw, dispatch_price_mwh,
            temperature_c, weather_station_name
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            demand_mw            = VALUES(demand_mw),
            dispatch_price_mwh   = VALUES(dispatch_price_mwh),
            temperature_c        = VALUES(temperature_c),
            weather_station_name = VALUES(weather_station_name);
    """

    conn = mysql.connector.connect(**DB_CONFIG)
    nem_df     = pd.read_sql(nem_query, get_engine())
    weather_df = pd.read_sql(weather_query, get_engine())

    if nem_df.empty:
        print("[Warning] raw_nem_data is empty — skipping cleaning.")
        conn.close()
        return

    nem_df["settlement_date"]      = pd.to_datetime(nem_df["settlement_date"])
    weather_df["observation_date"] = pd.to_datetime(weather_df["observation_date"])

    nem_df = nem_df[
        nem_df["dispatch_price_mwh"].between(PRICE_FLOOR_MWH, PRICE_CAP_MWH)
    ].copy()

    tolerance = pd.Timedelta(WEATHER_MERGE_TOLERANCE)
    print(f"[Cleaning] Using weather merge tolerance: {WEATHER_MERGE_TOLERANCE}")
    print(f"[Cleaning] NEM rows to clean: {len(nem_df):,}")

    aligned_groups = []

    for region_id, nem_group in nem_df.groupby("region_id"):
        weather_group = weather_df[weather_df["region_id"] == region_id].copy()
        nem_group = nem_group.sort_values("settlement_date").copy()

        if not weather_group.empty:
            weather_group = weather_group.sort_values("observation_date")
            nem_group = pd.merge_asof(
                nem_group,
                weather_group,
                left_on="settlement_date",
                right_on="observation_date",
                by="region_id",
                direction="backward",
                tolerance=tolerance,
            )
            nem_group["temperature_c"] = nem_group["temperature_c"].ffill(limit=48)
        else:
            nem_group["temperature_c"] = float("nan")
            nem_group["station_name"]  = None

        # Fill remaining NaN temperature with Melbourne monthly climate average
        still_missing = nem_group["temperature_c"].isna().sum()
        if still_missing:
            nem_group["temperature_c"] = nem_group.apply(
                lambda row: (
                    MELBOURNE_MONTHLY_MEAN_TEMP.get(row["settlement_date"].month)
                    if pd.isna(row["temperature_c"])
                    else row["temperature_c"]
                ),
                axis=1,
            )
            print(
                f"  [{region_id}] Filled {still_missing:,} rows with "
                "monthly climate average temperature."
            )

        aligned_groups.append(nem_group)

    if not aligned_groups:
        print("[Warning] No aligned rows produced.")
        conn.close()
        return

    clean_df = pd.concat(aligned_groups, ignore_index=True)

    # Ensure station_name column exists (may be absent if no weather matched)
    if "station_name" not in clean_df.columns:
        clean_df["station_name"] = None

    print(f"[Cleaning] {len(clean_df):,} rows ready to upsert.")

    # Use .values to get numpy array rows — avoids named-tuple/attribute issues
    cols = [
        "settlement_date", "region_id",
        "demand_mw", "dispatch_price_mwh",
        "temperature_c", "station_name",
    ]
    rows = [
        (row[0], row[1], _safe(row[2]), _safe(row[3]), _safe(row[4]), _safe(row[5]))
        for row in clean_df[cols].values.tolist()
    ]

    cursor = conn.cursor()
    cursor.executemany(insert_query, rows)
    conn.commit()
    print(f"[Cleaning] Upserted {cursor.rowcount:,} cleaned rows.")
    cursor.close()
    conn.close()