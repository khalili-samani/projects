"""
load_data.py
------------
Creates the market_analysis database schema and loads all cleaned stock price
CSVs into the fact_stock_prices table.

Steps performed:
    1. Create the database if it does not exist.
    2. Create dimension tables and insert reference data (sectors, companies).
    3. Create the fact table if it does not exist.
    4. Load each cleaned CSV into fact_stock_prices.
    5. Create analytical views.

Running this script more than once is safe — all CREATE statements use
IF NOT EXISTS, and each company's data is deleted and reloaded cleanly
to avoid duplicate rows.

Usage:
    python scripts/load_data.py
"""

import logging
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import (
    CLEANED_DIR,
    COMPANIES,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_USER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _check_password() -> None:
    """Raise immediately if the database password is not configured."""
    if not MYSQL_PASSWORD:
        raise EnvironmentError(
            "MYSQL_PASSWORD environment variable is not set. "
            "Export it before running this script."
        )


def _server_engine() -> Engine:
    """Return a SQLAlchemy engine at server level (no database selected)."""
    return create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}",
        pool_pre_ping=True,
    )


def _db_engine() -> Engine:
    """Return a SQLAlchemy engine connected to the target database."""
    return create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}",
        pool_pre_ping=True,
    )


def create_database() -> None:
    """Create the target database if it does not already exist."""
    sql = (
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    with _server_engine().connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("Database ready: %s", MYSQL_DATABASE)


def create_schema() -> None:
    """Create dimension tables, fact table, and insert reference data."""
    engine = _db_engine()

    statements = [

        # Dimension tables
        """
        CREATE TABLE IF NOT EXISTS dim_sector (
            sector_id   INT         NOT NULL,
            sector_name VARCHAR(50) NOT NULL,
            PRIMARY KEY (sector_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,

        """
        CREATE TABLE IF NOT EXISTS dim_company (
            company_id   INT          NOT NULL,
            company_name VARCHAR(100) NOT NULL,
            ticker       VARCHAR(10)  NOT NULL,
            sector_id    INT          NOT NULL,
            PRIMARY KEY (company_id),
            UNIQUE KEY uq_ticker (ticker),
            FOREIGN KEY (sector_id) REFERENCES dim_sector (sector_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,

        # Fact table
        """
        CREATE TABLE IF NOT EXISTS fact_stock_prices (
            price_id    INT            NOT NULL AUTO_INCREMENT,
            company_id  INT            NOT NULL,
            date        DATE           NOT NULL,
            open_price  DECIMAL(10, 4) NOT NULL,
            high_price  DECIMAL(10, 4) NOT NULL,
            low_price   DECIMAL(10, 4) NOT NULL,
            close_price DECIMAL(10, 4) NOT NULL,
            volume      BIGINT         NOT NULL,
            PRIMARY KEY (price_id),
            UNIQUE KEY uq_company_date (company_id, date),
            INDEX idx_date (date),
            FOREIGN KEY (company_id) REFERENCES dim_company (company_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
    ]

    with engine.connect() as conn:
        for sql in statements:
            conn.execute(text(sql))
        conn.commit()

    logger.info("Tables ready: dim_sector, dim_company, fact_stock_prices")

    # Insert reference data — skip if already present.
    sectors = [
        (1, "Technology"),
        (2, "Finance"),
        (3, "Energy"),
        (4, "Healthcare"),
    ]

    companies = [
        (1,  "Apple",             "AAPL", 1),
        (2,  "Microsoft",         "MSFT", 1),
        (3,  "Nvidia",            "NVDA", 1),
        (4,  "JPMorgan Chase",    "JPM",  2),
        (5,  "Goldman Sachs",     "GS",   2),
        (6,  "Bank of America",   "BAC",  2),
        (7,  "ExxonMobil",        "XOM",  3),
        (8,  "Chevron",           "CVX",  3),
        (9,  "ConocoPhillips",    "COP",  3),
        (10, "Johnson & Johnson", "JNJ",  4),
        (11, "Pfizer",            "PFE",  4),
        (12, "Merck",             "MRK",  4),
    ]

    with engine.connect() as conn:
        for sector_id, sector_name in sectors:
            conn.execute(
                text(
                    "INSERT IGNORE INTO dim_sector (sector_id, sector_name) "
                    "VALUES (:id, :name)"
                ),
                {"id": sector_id, "name": sector_name},
            )

        for company_id, company_name, ticker, sector_id in companies:
            conn.execute(
                text(
                    "INSERT IGNORE INTO dim_company "
                    "(company_id, company_name, ticker, sector_id) "
                    "VALUES (:cid, :name, :ticker, :sid)"
                ),
                {
                    "cid": company_id,
                    "name": company_name,
                    "ticker": ticker,
                    "sid": sector_id,
                },
            )

        conn.commit()

    logger.info("Reference data ready: %d sectors, %d companies", len(sectors), len(companies))


def load_company(ticker: str, company_id: int, company_name: str) -> None:
    """Load one company's cleaned CSV into fact_stock_prices.

    Existing rows for the company are deleted before loading to ensure
    the script is safe to re-run without creating duplicates.

    Args:
        ticker: File stem, e.g. "aapl_us".
        company_id: Integer FK matching dim_company.
        company_name: Human-readable name used for logging.

    Raises:
        FileNotFoundError: If the cleaned CSV does not exist.
    """
    csv_path = CLEANED_DIR / f"{ticker}_clean.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned file not found: {csv_path} — "
            "run clean_data.py first."
        )

    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open_price"] = pd.to_numeric(df["open_price"], errors="coerce")
    df["high_price"] = pd.to_numeric(df["high_price"], errors="coerce")
    df["low_price"] = pd.to_numeric(df["low_price"], errors="coerce")
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["date", "close_price"])

    load_cols = [
        "company_id",
        "date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
    ]
    df = df[load_cols]

    engine = _db_engine()

    # Delete existing rows for this company before reloading.
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM fact_stock_prices WHERE company_id = :cid"),
            {"cid": company_id},
        )
        conn.commit()

    df.to_sql(
        "fact_stock_prices",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=500,
        method="multi",
    )

    logger.info(
        "%s (%s): %d rows loaded.", company_name, ticker, len(df)
    )


def create_views() -> None:
    """Create or replace all analytical views."""
    engine = _db_engine()

    views = {
        "vw_daily_returns": """
            CREATE OR REPLACE VIEW vw_daily_returns AS
            WITH price_changes AS (
                SELECT
                    company_id,
                    date,
                    close_price,
                    LAG(close_price) OVER (
                        PARTITION BY company_id ORDER BY date
                    ) AS previous_close
                FROM fact_stock_prices
            )
            SELECT
                company_id,
                date,
                close_price,
                previous_close,
                ROUND(
                    (close_price - previous_close) / previous_close * 100,
                    4
                ) AS daily_return_pct
            FROM price_changes
            WHERE previous_close IS NOT NULL
        """,

        "vw_company_summary": """
            CREATE OR REPLACE VIEW vw_company_summary AS
            SELECT
                c.company_id,
                c.company_name,
                c.ticker,
                s.sector_name,
                ROUND(AVG(r.daily_return_pct),    4) AS avg_daily_return_pct,
                ROUND(STDDEV(r.daily_return_pct), 4) AS volatility_pct,
                ROUND(
                    ((MAX(f.close_price) - MIN(f.close_price))
                        / MIN(f.close_price)) * 100, 2
                )                                    AS total_return_pct,
                COUNT(DISTINCT r.date)               AS trading_days
            FROM vw_daily_returns r
            JOIN dim_company c ON r.company_id = c.company_id
            JOIN dim_sector s  ON c.sector_id  = s.sector_id
            JOIN fact_stock_prices f ON r.company_id = f.company_id
            GROUP BY
                c.company_id, c.company_name, c.ticker, s.sector_name
        """,

        "vw_sector_summary": """
            CREATE OR REPLACE VIEW vw_sector_summary AS
            SELECT
                s.sector_name,
                COUNT(DISTINCT c.company_id)          AS company_count,
                ROUND(AVG(r.daily_return_pct),    4)  AS avg_daily_return_pct,
                ROUND(STDDEV(r.daily_return_pct), 4)  AS volatility_pct
            FROM vw_daily_returns r
            JOIN dim_company c ON r.company_id = c.company_id
            JOIN dim_sector s  ON c.sector_id  = s.sector_id
            GROUP BY s.sector_name
            ORDER BY avg_daily_return_pct DESC
        """,
    }

    with engine.connect() as conn:
        for view_name, sql in views.items():
            conn.execute(text(sql))
            conn.commit()
            logger.info("View ready: %s", view_name)


def main() -> None:
    """Run the full load pipeline for all companies."""
    _check_password()

    logger.info("Starting load step for %d companies.", len(COMPANIES))

    create_database()
    create_schema()

    success = 0
    failures: list[str] = []

    for ticker, meta in COMPANIES.items():
        try:
            load_company(ticker, meta["company_id"], meta["name"])
            success += 1
        except (FileNotFoundError, Exception) as exc:
            logger.error("%s: %s", ticker, exc)
            failures.append(ticker)

    create_views()

    logger.info(
        "Load step completed — %d succeeded, %d failed.",
        success,
        len(failures),
    )

    if failures:
        logger.error("Failed tickers: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
