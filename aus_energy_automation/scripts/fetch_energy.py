"""
fetch_energy.py
---------------
Fetches raw NEM price and demand CSV files from AEMO for a given month.

Each region defined in config.REGIONS is downloaded separately and saved
to the raw data directory. Raises an exception immediately on any HTTP error
so the pipeline fails fast rather than loading incomplete data.

Usage:
    python scripts/fetch_energy.py YYYYMM
"""

import logging
import sys

import certifi
import requests

from config import BASE_URL, RAW_DIR, REGIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_url(year_month: str, region: str) -> str:
    """Construct the AEMO download URL for a given month and region.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".
        region: NEM region code, e.g. "NSW1".

    Returns:
        Full URL string pointing to the AEMO CSV file.
    """
    return f"{BASE_URL}/PRICE_AND_DEMAND_{year_month}_{region}.csv"


def fetch_region(year_month: str, region: str) -> None:
    """Download and save the AEMO price/demand CSV for one region.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".
        region: NEM region code, e.g. "NSW1".

    Raises:
        requests.HTTPError: If the AEMO server returns a non-2xx response.
        requests.RequestException: On network-level failures.
    """
    url = build_url(year_month, region)
    logger.info("Fetching: %s", url)

    response = requests.get(url, timeout=30, verify=certifi.where())
    response.raise_for_status()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / f"PRICE_AND_DEMAND_{year_month}_{region}.csv"
    output_path.write_text(response.text, encoding="utf-8")

    logger.info("Saved: %s", output_path)


def main(year_month: str) -> None:
    """Fetch raw data for all configured regions.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".
    """
    logger.info("Starting fetch step for period: %s", year_month)

    for region in REGIONS:
        fetch_region(year_month, region)

    logger.info("Fetch step completed — %d region(s) downloaded.", len(REGIONS))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python scripts/fetch_energy.py YYYYMM")
        sys.exit(1)

    main(sys.argv[1])
