from pathlib import Path
import sys
import requests
import certifi

from config import BASE_URL, REGIONS, RAW_DIR


def build_url(year_month: str, region: str) -> str:
    return f"{BASE_URL}/PRICE_AND_DEMAND_{year_month}_{region}.csv"


def fetch_region(year_month: str, region: str) -> None:
    url = build_url(year_month, region)
    print(f"Fetching {url}")

    response = requests.get(url, timeout=30, verify=certifi.where())
    response.raise_for_status()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / f"PRICE_AND_DEMAND_{year_month}_{region}.csv"
    output_path.write_text(response.text, encoding="utf-8")

    print(f"Saved: {output_path}")


def main(year_month: str) -> None:
    for region in REGIONS:
        fetch_region(year_month, region)

    print("Fetch step completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/fetch_energy.py YYYYMM")

    main(sys.argv[1])
