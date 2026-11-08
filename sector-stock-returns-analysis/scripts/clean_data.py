"""
clean_data.py
-------------
Cleans and standardises raw stock price CSV files for all configured companies.

For each company in the registry, the raw Stooq CSV is loaded, columns are
renamed to a consistent snake_case schema, a company_id is attached, types
are coerced, and rows with missing values are dropped. Each cleaned file is
saved individually to data_cleaned/ for downstream loading into MySQL.

Data source: Stooq (https://stooq.com)
Expected raw filename format: {ticker}_us.csv (e.g. aapl_us.csv)

Usage:
    python scripts/clean_data.py
"""

import logging
import sys

import pandas as pd

from config import CLEANED_DIR, COMPANIES, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Columns expected in every raw Stooq CSV after renaming.
REQUIRED_COLUMNS: list[str] = [
    "date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
]

COLUMN_RENAME_MAP: dict[str, str] = {
    "Date":   "date",
    "Open":   "open_price",
    "High":   "high_price",
    "Low":    "low_price",
    "Close":  "close_price",
    "Volume": "volume",
}

NUMERIC_COLUMNS: list[str] = [
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
]


def clean_company(ticker: str, company_id: int, company_name: str) -> pd.DataFrame:
    """Load, clean, and standardise the raw CSV for one company.

    Args:
        ticker: Raw file stem, e.g. "aapl_us".
        company_id: Integer FK matching dim_company in the database.
        company_name: Human-readable name used for logging.

    Returns:
        Cleaned DataFrame ready to be saved as CSV.

    Raises:
        FileNotFoundError: If the raw CSV does not exist.
        ValueError: If required columns are missing after renaming.
    """
    file_path = RAW_DIR / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    df = pd.read_csv(file_path)
    raw_row_count = len(df)

    # Rename to snake_case schema.
    df = df.rename(columns=COLUMN_RENAME_MAP)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{company_name}: missing column(s) after rename: {missing}"
        )

    # Attach company identifier.
    df["company_id"] = company_id

    # Parse and standardise types.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    # Drop rows where any field could not be parsed.
    before_drop = len(df)
    df = df.dropna()
    dropped = before_drop - len(df)

    if dropped > 0:
        logger.warning(
            "%s: dropped %d row(s) with unparseable values.", company_name, dropped
        )

    # Sort chronologically and reorder columns.
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["company_id", "date"] + REQUIRED_COLUMNS]

    logger.info(
        "%s (%s): %d rows | %s to %s",
        company_name,
        ticker,
        len(df),
        df["date"].iloc[0],
        df["date"].iloc[-1],
    )

    return df


def main() -> None:
    """Clean raw CSVs for all companies defined in config.COMPANIES."""
    logger.info(
        "Starting clean step for %d companies.", len(COMPANIES)
    )

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failures: list[str] = []

    for ticker, meta in COMPANIES.items():
        try:
            df = clean_company(ticker, meta["company_id"], meta["name"])
            output_path = CLEANED_DIR / f"{ticker}_clean.csv"
            df.to_csv(output_path, index=False)
            success += 1
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s: %s", ticker, exc)
            failures.append(ticker)

    logger.info(
        "Clean step completed — %d succeeded, %d failed.",
        success,
        len(failures),
    )

    if failures:
        logger.error("Failed tickers: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
