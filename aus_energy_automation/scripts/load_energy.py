"""
load_energy.py
--------------
Loads cleaned NEM price/demand data into MySQL for a given month.

Usage:
    python scripts/load_energy.py YYYYMM
"""

import logging
import sys

import pandas as pd
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine

from config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_USER,
    PROCESSED_DIR,
    TABLE_NAME,
    VIEW_NAME,
    REGIONAL_VIEW_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "region_code",
    "settlement_datetime",
    "trading_date",
    "rrp_aud_mwh",
    "total_demand_mw",
    "period_type",
]


def _check_password() -> None:
    """Validate database password configuration."""
    if MYSQL_PASSWORD is None or MYSQL_PASSWORD == "":
        raise EnvironmentError(
            "MYSQL_PASSWORD is empty. Set it as an environment variable before running the pipeline."
        )


def _server_engine() -> Engine:
    """Return a SQLAlchemy engine connected to the MySQL server."""
    url = URL.create(
        drivername="mysql+pymysql",
        username=MYSQL_USER,
        password=MYSQL_PASSWORD,
        host=MYSQL_HOST,
    )
    return create_engine(url, pool_pre_ping=True)


def _db_engine() -> Engine:
    """Return a SQLAlchemy engine connected to the target database."""
    url = URL.create(
        drivername="mysql+pymysql",
        username=MYSQL_USER,
        password=MYSQL_PASSWORD,
        host=MYSQL_HOST,
        database=MYSQL_DATABASE,
    )
    return create_engine(url, pool_pre_ping=True)


def create_database() -> None:
    """Create the target MySQL database if it does not already exist."""
    sql = (
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )

    with _server_engine().connect() as conn:
        conn.execute(text(sql))
        conn.commit()

    logger.info("Database ready: %s", MYSQL_DATABASE)


def create_table() -> None:
    """Create the fact table if it does not already exist."""
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
        record_id           BIGINT       NOT NULL AUTO_INCREMENT,
        region_code         VARCHAR(10)  NOT NULL,
        settlement_datetime DATETIME     NOT NULL,
        trading_date        DATE         NOT NULL,
        rrp_aud_mwh         DECIMAL(12,4),
        total_demand_mw     DECIMAL(12,4),
        period_type         VARCHAR(20),
        PRIMARY KEY (record_id),
        INDEX idx_region_date (region_code, trading_date),
        INDEX idx_settlement  (settlement_datetime)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """

    with _db_engine().connect() as conn:
        conn.execute(text(sql))
        conn.commit()

    logger.info("Table ready: %s", TABLE_NAME)


def create_daily_summary_view() -> None:
    """Create or replace the daily summary analytical view."""
    sql = f"""
    CREATE OR REPLACE VIEW `{VIEW_NAME}` AS
    SELECT
        trading_date,
        region_code,
        ROUND(AVG(rrp_aud_mwh), 2)     AS avg_price_aud_mwh,
        ROUND(MAX(rrp_aud_mwh), 2)     AS max_price_aud_mwh,
        ROUND(MIN(rrp_aud_mwh), 2)     AS min_price_aud_mwh,
        ROUND(STDDEV(rrp_aud_mwh), 2)  AS price_volatility,
        ROUND(AVG(total_demand_mw), 2) AS avg_demand_mw,
        COUNT(*)                       AS interval_count
    FROM `{TABLE_NAME}`
    GROUP BY trading_date, region_code
    """

    with _db_engine().connect() as conn:
        conn.execute(text(sql))
        conn.commit()

    logger.info("View ready: %s", VIEW_NAME)


def create_regional_summary_view() -> None:
    """Create or replace the regional summary analytical view."""
    sql = f"""
    CREATE OR REPLACE VIEW `{REGIONAL_VIEW_NAME}` AS
    SELECT
        region_code,
        MIN(trading_date) AS start_date,
        MAX(trading_date) AS end_date,
        COUNT(*)          AS interval_count,

        ROUND(AVG(rrp_aud_mwh), 2)    AS avg_price_aud_mwh,
        ROUND(MAX(rrp_aud_mwh), 2)    AS max_price_aud_mwh,
        ROUND(MIN(rrp_aud_mwh), 2)    AS min_price_aud_mwh,
        ROUND(STDDEV(rrp_aud_mwh), 2) AS price_volatility,

        ROUND(AVG(total_demand_mw), 2) AS avg_demand_mw,
        ROUND(MAX(total_demand_mw), 2) AS max_demand_mw,
        ROUND(MIN(total_demand_mw), 2) AS min_demand_mw,

        SUM(CASE WHEN rrp_aud_mwh < 0 THEN 1 ELSE 0 END) AS negative_price_intervals,
        ROUND(
            100 * SUM(CASE WHEN rrp_aud_mwh < 0 THEN 1 ELSE 0 END) / COUNT(*),
            2
        ) AS negative_price_pct
    FROM `{TABLE_NAME}`
    GROUP BY region_code
    """

    with _db_engine().connect() as conn:
        conn.execute(text(sql))
        conn.commit()

    logger.info("View ready: %s", REGIONAL_VIEW_NAME)


def load_csv(year_month: str) -> None:
    """Load the processed monthly CSV into MySQL."""
    csv_path = PROCESSED_DIR / f"nem_price_demand_{year_month}_combined.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["settlement_datetime"] = pd.to_datetime(
        df["settlement_datetime"],
        errors="coerce",
    )

    df["trading_date"] = pd.to_datetime(
        df["trading_date"],
        errors="coerce",
    ).dt.date

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

    df = df[REQUIRED_COLUMNS]

    engine = _db_engine()

    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE `{TABLE_NAME}`"))
        conn.commit()

    logger.info("Cleared existing data from: %s", TABLE_NAME)

    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi",
    )

    logger.info("Loaded %d rows into %s.", len(df), TABLE_NAME)


def main(year_month: str) -> None:
    """Run the full load step for a given period."""
    _check_password()

    logger.info("Starting load step for period: %s", year_month)

    create_database()
    create_table()
    load_csv(year_month)
    create_daily_summary_view()
    create_regional_summary_view()

    logger.info("Load step completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python scripts/load_energy.py YYYYMM")
        sys.exit(1)

    main(sys.argv[1])