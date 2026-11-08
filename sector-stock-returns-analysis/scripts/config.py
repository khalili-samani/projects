"""
config.py
---------
Centralised configuration for the Sector-Based Stock Market Analysis project.

All sensitive values (database credentials) are sourced from environment
variables. Modify COMPANIES to add or remove tickers from the pipeline.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent.parent

RAW_DIR: Path = BASE_DIR / "data_raw"
CLEANED_DIR: Path = BASE_DIR / "data_cleaned"
CHARTS_DIR: Path = BASE_DIR / "outputs" / "charts"

# ---------------------------------------------------------------------------
# MySQL connection
# ---------------------------------------------------------------------------

MYSQL_USER: str = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD: str | None = os.environ.get("MYSQL_PASSWORD")
MYSQL_HOST: str = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_DATABASE: str = os.environ.get("MYSQL_DATABASE", "market_analysis")

# ---------------------------------------------------------------------------
# Company registry
# Maps ticker → (company_id, output filename stem)
# company_id must match dim_company in the database schema.
# ---------------------------------------------------------------------------

COMPANIES: dict[str, dict] = {
    "aapl_us": {"company_id": 1,  "name": "Apple"},
    "msft_us": {"company_id": 2,  "name": "Microsoft"},
    "nvda_us": {"company_id": 3,  "name": "Nvidia"},
    "jpm_us":  {"company_id": 4,  "name": "JPMorgan Chase"},
    "gs_us":   {"company_id": 5,  "name": "Goldman Sachs"},
    "bac_us":  {"company_id": 6,  "name": "Bank of America"},
    "xom_us":  {"company_id": 7,  "name": "ExxonMobil"},
    "cvx_us":  {"company_id": 8,  "name": "Chevron"},
    "cop_us":  {"company_id": 9,  "name": "ConocoPhillips"},
    "jnj_us":  {"company_id": 10, "name": "Johnson & Johnson"},
    "pfe_us":  {"company_id": 11, "name": "Pfizer"},
    "mrk_us":  {"company_id": 12, "name": "Merck"},
}

# ---------------------------------------------------------------------------
# Chart settings
# ---------------------------------------------------------------------------

CHART_DPI: int = 300

SECTOR_COLOURS: dict[str, str] = {
    "Technology":  "#1f4e79",
    "Finance":     "#7f6000",
    "Energy":      "#38761d",
    "Healthcare":  "#741b47",
}
