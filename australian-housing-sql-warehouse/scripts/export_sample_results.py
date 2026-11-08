#!/usr/bin/env python3
"""Export selected MySQL portfolio result sets to CSV."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv

QUERIES = {
    "monthly_market_summary.csv": "SELECT * FROM mart.monthly_market_summary ORDER BY state_code, property_type_code, sale_month",
    "data_quality_summary.csv": "SELECT * FROM mart.data_quality_summary ORDER BY batch_id, record_disposition, duplicate_class",
    "executive_monthly_kpis.csv": "SELECT * FROM mart.v_executive_monthly_kpis ORDER BY state_code, sale_month",
}


def connection_config() -> dict[str, object]:
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "database": os.getenv("MYSQL_DATABASE", "housing_analytics"),
        "user": os.getenv("MYSQL_USER", "housing_user"),
        "password": os.getenv("MYSQL_PASSWORD", "housing_password"),
    }


def main() -> None:
    load_dotenv()
    output_dir = Path("outputs/sample_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with mysql.connector.connect(**connection_config()) as connection:
        for file_name, query in QUERIES.items():
            cursor = connection.cursor()
            try:
                cursor.execute(query)
                output_path = output_dir / file_name
                with output_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([column[0] for column in cursor.description])
                    writer.writerows(cursor.fetchall())
                print(f"Exported {output_path}")
            finally:
                cursor.close()


if __name__ == "__main__":
    main()
