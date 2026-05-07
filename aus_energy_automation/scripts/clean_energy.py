"""
clean_energy.py
---------------
Cleans and standardises raw NEM price/demand CSV files for a given month.

For each configured region the raw file is loaded, columns are renamed to a
consistent snake_case schema, types are coerced, and rows with missing core
values are dropped. All regions are then concatenated into a single processed
CSV for downstream loading.

Note on negative prices: negative RRP values are a known feature of the NEM,
typically caused by oversupply from non-dispatchable generation (e.g. rooftop
solar). They are intentionally retained rather than treated as errors.

Usage:
    python scripts/clean_energy.py YYYYMM
"""

import logging
import sys

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, REGIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Columns expected directly from the AEMO CSV after renaming.
# region_code and trading_date are derived in code and excluded from this check.
RAW_REQUIRED_COLUMNS: list[str] = [
    "settlement_datetime",
    "rrp_aud_mwh",
    "total_demand_mw",
    "period_type",
]

# Columns required to be non-null before a row is kept.
REQUIRED_COLUMNS: list[str] = [
    "region_code",
    "settlement_datetime",
    "trading_date",
    "rrp_aud_mwh",
    "total_demand_mw",
]

OUTPUT_COLUMNS: list[str] = [
    "region_code",
    "settlement_datetime",
    "trading_date",
    "rrp_aud_mwh",
    "total_demand_mw",
    "period_type",
]


def load_region_file(year_month: str, region_code: str) -> pd.DataFrame:
    """Load, clean, and standardise the raw CSV for one region.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".
        region_code: NEM region code, e.g. "NSW1".

    Returns:
        Cleaned DataFrame with standardised column names and types.

    Raises:
        FileNotFoundError: If the expected raw file does not exist.
        ValueError: If required columns are absent after renaming.
    """
    file_path = RAW_DIR / f"PRICE_AND_DEMAND_{year_month}_{region_code}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    df = pd.read_csv(file_path)
    raw_row_count = len(df)

    # Normalise column names to lowercase snake_case.
    df.columns = [col.strip().lower() for col in df.columns]

    df = df.rename(
        columns={
            "settlementdate": "settlement_datetime",
            "rrp": "rrp_aud_mwh",
            "totaldemand": "total_demand_mw",
            "periodtype": "period_type",
        }
    )

    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{region_code}: missing required column(s) after rename: {missing}"
        )

    # Attach region identifier and parse types.
    df["region_code"] = region_code
    df["settlement_datetime"] = pd.to_datetime(
        df["settlement_datetime"], errors="coerce"
    )
    df["trading_date"] = df["settlement_datetime"].dt.date
    df["rrp_aud_mwh"] = pd.to_numeric(df["rrp_aud_mwh"], errors="coerce")
    df["total_demand_mw"] = pd.to_numeric(df["total_demand_mw"], errors="coerce")

    df = df[OUTPUT_COLUMNS]

    # Drop rows where any core field could not be parsed.
    df = df.dropna(subset=REQUIRED_COLUMNS)

    dropped = raw_row_count - len(df)
    if dropped > 0:
        logger.warning(
            "%s: dropped %d row(s) with unparseable core values.", region_code, dropped
        )

    df = df.sort_values(["region_code", "settlement_datetime"]).reset_index(drop=True)

    # Log price range to surface any unexpected extreme values for review.
    logger.info(
        "%s: %d rows | price range %.2f to %.2f AUD/MWh",
        region_code,
        len(df),
        df["rrp_aud_mwh"].min(),
        df["rrp_aud_mwh"].max(),
    )

    return df


def main(year_month: str) -> None:
    """Clean all configured regions and write a combined processed CSV.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".
    """
    logger.info("Starting clean step for period: %s", year_month)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for region in REGIONS:
        df_region = load_region_file(year_month, region)
        frames.append(df_region)

    combined = pd.concat(frames, ignore_index=True)

    output_path = PROCESSED_DIR / f"nem_price_demand_{year_month}_combined.csv"
    combined.to_csv(output_path, index=False)

    logger.info(
        "Clean step completed — %d total rows saved to %s.",
        len(combined),
        output_path,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python scripts/clean_energy.py YYYYMM")
        sys.exit(1)

    main(sys.argv[1])
