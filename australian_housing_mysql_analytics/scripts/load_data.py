#!/usr/bin/env python3
"""Load the simulator CSV into MySQL without coercing source fields."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Iterable

import mysql.connector
from dotenv import load_dotenv

EXPECTED_COLUMNS = ['listing_id', 'source', 'address', 'suburb', 'state', 'postcode', 'council_area', 'region', 'distance_to_cbd_km', 'lat', 'lon', 'property_type', 'bedrooms', 'bathrooms', 'car_spaces', 'toilets', 'land_size', 'building_area', 'year_built', 'has_pool', 'has_garage', 'sale_price', 'price_raw_aud', 'sale_date', 'sale_method', 'days_on_market', 'inspection_note', 'agent_name', 'agency_name', 'agent_phone', 'rba_cash_rate_pct', 'market_sentiment', 'market_context', 'suburb_median_price', 'auction_clearance_rate_pct', 'weekly_rent_aud', 'property_count_suburb']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_hash(values: Iterable[str | None]) -> str:
    canonical = "|".join("" if value is None else value for value in values)
    return hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()


def connection_config() -> dict[str, object]:
    return {"host": os.getenv("MYSQL_HOST", "localhost"), "port": int(os.getenv("MYSQL_PORT", "3306")), "database": os.getenv("MYSQL_DATABASE", "housing_analytics"), "user": os.getenv("MYSQL_USER", "housing_user"), "password": os.getenv("MYSQL_PASSWORD", "housing_password"), "autocommit": False}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.getenv("SOURCE_CSV", "data/raw/aus_housing_messy_2021-2023_jan-dec_nsw-qld-vic.csv"))
    parser.add_argument("--source-system", default=os.getenv("SOURCE_SYSTEM", "australian_housing_data_quality_simulator"))
    parser.add_argument("--force", action="store_true", help="Reload a checksum already marked loaded")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    checksum = sha256_file(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            missing = sorted(set(EXPECTED_COLUMNS) - set(reader.fieldnames or []))
            unexpected = sorted(set(reader.fieldnames or []) - set(EXPECTED_COLUMNS))
            raise ValueError(f"Schema mismatch. Missing={missing} unexpected={unexpected} order={reader.fieldnames}")
        rows = list(reader)

    insert_columns = ["batch_id", "source_row_number", "source_row_hash", *EXPECTED_COLUMNS]
    placeholders = ",".join(["%s"] * len(insert_columns))
    insert_sql = f"INSERT INTO raw.housing_listing ({','.join(insert_columns)}) VALUES ({placeholders})"

    with mysql.connector.connect(**connection_config()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT batch_id FROM raw.ingestion_batch WHERE source_file_sha256=%s AND status='loaded'", (checksum,))
            existing = cur.fetchone()
            if existing and not args.force:
                print(f"Skipped: checksum already loaded as batch {existing[0]}")
                return
            cur.execute(
                "INSERT INTO raw.ingestion_batch(source_file_name,source_file_sha256,source_system,status) VALUES(%s,%s,%s,'started')",
                (csv_path.name, checksum, args.source_system),
            )
            batch_id = cur.lastrowid
            try:
                payload = []
                for number, row in enumerate(rows, start=2):
                    values = [row.get(column) for column in EXPECTED_COLUMNS]
                    payload.append((batch_id, number, row_hash(values), *values))
                cur.executemany(insert_sql, payload)
                cur.execute("UPDATE raw.ingestion_batch SET row_count=%s,status='loaded' WHERE batch_id=%s", (len(rows), batch_id))
            except Exception as exc:
                cur.execute("UPDATE raw.ingestion_batch SET status='failed',error_message=%s WHERE batch_id=%s", (str(exc), batch_id))
                conn.rollback()
                raise
            else:
                conn.commit()
    print(f"Loaded {len(rows):,} rows into batch {batch_id} from {csv_path}")


if __name__ == "__main__":
    main()
