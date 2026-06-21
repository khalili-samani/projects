"""
config.py
---------
Centralised configuration for the Australian Energy Market pipeline.

All runtime settings are sourced from environment variables where sensitive,
with safe defaults for non-sensitive values. Modify this file to extend
regions, paths, or database targets.
"""

import os
from pathlib import Path

BASE_URL: str = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"

REGIONS: list[str] = ["NSW1", "QLD1", "VIC1"]

BASE_DIR: Path = Path(__file__).resolve().parent.parent

RAW_DIR: Path = BASE_DIR / "data" / "raw"
PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
CHARTS_DIR: Path = BASE_DIR / "outputs" / "charts"

MYSQL_USER: str = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD: str | None = os.environ.get("MYSQL_PASSWORD")
MYSQL_HOST: str = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_DATABASE: str = os.environ.get("MYSQL_DATABASE", "aus_energy_automation")

TABLE_NAME: str = "fact_nem_price_demand"
VIEW_NAME: str = "vw_nem_daily_summary"
REGIONAL_VIEW_NAME: str = "vw_nem_regional_summary"

CHART_DPI: int = 300
HISTOGRAM_BINS: int = 40
PRICE_SPIKE_THRESHOLD_AUD: float = 300.0