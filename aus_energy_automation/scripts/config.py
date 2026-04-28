from pathlib import Path
import os

BASE_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"
REGIONS = ["NSW1", "QLD1", "VIC1"]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHARTS_DIR = Path("outputs/charts")

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "your_password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "aus_energy_automation")

TABLE_NAME = "fact_nem_price_demand"
