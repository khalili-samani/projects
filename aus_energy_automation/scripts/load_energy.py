import sys
import pandas as pd
from sqlalchemy import create_engine, text

from config import (
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_DATABASE,
    PROCESSED_DIR,
    TABLE_NAME,
)


if not MYSQL_PASSWORD:
    raise ValueError("MYSQL_PASSWORD environment variable is not set.")


server_engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}"
)

db_engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
)


def create_database() -> None:
    create_db_sql = f"""
    CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci
    """

    with server_engine.connect() as conn:
        conn.execute(text(create_db_sql))
        conn.commit()

    print(f"Database ready: {MYSQL_DATABASE}")


def create_table() -> None:
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        record_id BIGINT PRIMARY KEY AUTO_INCREMENT,
        region_code VARCHAR(10) NOT NULL,
        settlement_datetime DATETIME NOT NULL,
        trading_date DATE NOT NULL,
        rrp_aud_mwh DECIMAL(12,4),
        total_demand_mw DECIMAL(12,4),
        period_type VARCHAR(20)
    )
    """

    with db_engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()

    print(f"Table ready: {TABLE_NAME}")


def create_daily_summary_view() -> None:
    create_view_sql = f"""
    CREATE OR REPLACE VIEW vw_nem_daily_summary AS
    SELECT
        trading_date,
        region_code,
        ROUND(AVG(rrp_aud_mwh), 2) AS avg_price_aud_mwh,
        ROUND(MAX(rrp_aud_mwh), 2) AS max_price_aud_mwh,
        ROUND(MIN(rrp_aud_mwh), 2) AS min_price_aud_mwh,
        ROUND(STDDEV(rrp_aud_mwh), 2) AS price_volatility,
        ROUND(AVG(total_demand_mw), 2) AS avg_demand_mw
    FROM {TABLE_NAME}
    GROUP BY trading_date, region_code
    """

    with db_engine.connect() as conn:
        conn.execute(text(create_view_sql))
        conn.commit()

    print("View ready: vw_nem_daily_summary")


def load_csv(year_month: str) -> None:
    csv_path = PROCESSED_DIR / f"nem_price_demand_{year_month}_combined.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "region_code",
        "settlement_datetime",
        "trading_date",
        "rrp_aud_mwh",
        "total_demand_mw",
        "period_type",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["settlement_datetime"] = pd.to_datetime(df["settlement_datetime"], errors="coerce")
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.date
    df["rrp_aud_mwh"] = pd.to_numeric(df["rrp_aud_mwh"], errors="coerce")
    df["total_demand_mw"] = pd.to_numeric(df["total_demand_mw"], errors="coerce")

    df = df.dropna(
        subset=[
            "region_code",
            "settlement_datetime",
            "trading_date",
            "rrp_aud_mwh",
            "total_demand_mw",
        ]
    )

    df = df[required_cols]

    with db_engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLE_NAME}"))
        conn.commit()

    print(f"Cleared old data from: {TABLE_NAME}")

    df.to_sql(
        TABLE_NAME,
        con=db_engine,
        if_exists="append",
        index=False,
        chunksize=1000,
    )

    print(f"Loaded {len(df)} rows into {TABLE_NAME}")


def main(year_month: str) -> None:
    create_database()
    create_table()
    load_csv(year_month)
    create_daily_summary_view()

    print("Load step completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/load_energy.py YYYYMM")

    main(sys.argv[1])
