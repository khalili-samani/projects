"""
Data ingestion pipeline.

Downloads:
- Historical AEMO DispatchIS data from NEMWeb archive.
- Current live DispatchIS files from NEMWeb CURRENT.
- Configured BOM weather observations.

AEMO archive structure (confirmed May 2026):
  The archive index lists one zip per day:
    PUBLIC_DISPATCHIS_YYYYMMDD.zip

  Each daily zip contains 288 individual 5-min interval zips:
    PUBLIC_DISPATCHIS_YYYYMMDDHHmm_<id>.zip

  Each inner zip contains one CSV in AEMO MMS format.

AEMO MMS CSV format:
  C  — file header
  I  — column header row for following D rows
  D  — data row
  Column layout: parts[0]=type, parts[1]=report, parts[2]=table,
                 parts[3]=version, parts[4+]=data columns
  Tables used:
    DISPATCH / PRICE     — RRP per region
    DISPATCH / REGIONSUM — TOTALDEMAND per region
"""

import csv
import io
import zipfile
from datetime import datetime
from urllib.parse import urljoin

import mysql.connector
import requests
from bs4 import BeautifulSoup

from src.config import (
    AEMO_DISPATCHIS_URL,
    AEMO_ARCHIVE_BASE_URL,
    AEMO_ARCHIVE_MONTHS,
    AEMO_ARCHIVE_DAYS_PER_MONTH,
    BOM_STATIONS,
    DB_CONFIG,
)

REQUEST_HEADERS = {"User-Agent": "nem-forecasting-ml/1.0"}
REQUEST_TIMEOUT_SECONDS = 60
DOMAIN = "https://nemweb.com.au"


def _to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _extract_dispatch_region_rows(csv_text):
    """
    Parse one AEMO MMS CSV and return
    (settlement_date, region_id, demand_mw, dispatch_price_mwh) tuples.
    """
    reader = csv.reader(io.StringIO(csv_text))

    headers = {}
    price_rows = {}
    demand_rows = {}

    for parts in reader:
        if len(parts) < 5:
            continue

        row_type = parts[0].strip().upper()
        report   = parts[1].strip().upper()
        table    = parts[2].strip().upper()
        key      = (report, table)

        if row_type == "I":
            headers[key] = [c.strip().upper() for c in parts[4:]]
            continue

        if row_type != "D":
            continue

        col_headers = headers.get(key)
        if not col_headers:
            continue

        row = dict(zip(col_headers, parts[4:]))
        settlement_date = row.get("SETTLEMENTDATE", "").strip().strip('"')
        region_id       = row.get("REGIONID", "").strip()

        if not settlement_date or not region_id:
            continue

        if table == "PRICE":
            rrp = row.get("RRP")
            if rrp is not None:
                try:
                    price_rows[(settlement_date, region_id)] = _to_float(rrp)
                except ValueError:
                    pass

        elif table == "REGIONSUM":
            demand = row.get("TOTALDEMAND")
            if demand is not None:
                try:
                    demand_rows[(settlement_date, region_id)] = _to_float(demand)
                except ValueError:
                    pass

    result = []
    for (settlement_date, region_id), price in price_rows.items():
        demand = demand_rows.get((settlement_date, region_id))
        if demand is None or price is None:
            continue
        result.append((settlement_date, region_id, demand, price))

    return result


def _parse_csv_from_zip_bytes(zip_bytes):
    """
    Parse rows from zip bytes. Handles two structures:
      - Direct: zip contains CSV files.
      - Nested: zip contains inner zips which contain CSV files.
    """
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer_zf:
        for name in outer_zf.namelist():
            name_lower = name.lower()

            if name_lower.endswith(".csv"):
                with outer_zf.open(name) as f:
                    csv_text = f.read().decode("utf-8", errors="ignore")
                rows.extend(_extract_dispatch_region_rows(csv_text))

            elif name_lower.endswith(".zip"):
                inner_bytes = outer_zf.read(name)
                try:
                    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
                        for inner_name in inner_zf.namelist():
                            if inner_name.lower().endswith(".csv"):
                                with inner_zf.open(inner_name) as f:
                                    csv_text = f.read().decode("utf-8", errors="ignore")
                                rows.extend(_extract_dispatch_region_rows(csv_text))
                except zipfile.BadZipFile:
                    pass

    return rows


def _fetch_and_parse_zip(zip_url):
    """Download one zip URL and return parsed dispatch rows. Skips on error."""
    try:
        resp = requests.get(
            zip_url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return _parse_csv_from_zip_bytes(resp.content)
    except Exception as exc:
        print(f"  [Warning] Skipping {zip_url}: {exc}")
        return []


def _href_to_url(href, base_url):
    """Convert an AEMO page href to an absolute URL."""
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return DOMAIN + href
    return base_url + href


def _get_zip_urls(index_url):
    """Return (absolute_url, filename) pairs for all .zip links on an AEMO index page."""
    resp = requests.get(index_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.lower().endswith(".zip"):
            url = _href_to_url(href, index_url)
            filename = url.split("/")[-1]
            results.append((url, filename))
    return results


def _month_from_filename(filename):
    """
    Extract YYYYMM from AEMO filename.
    e.g. PUBLIC_DISPATCHIS_20260415.zip       → 202604
         PUBLIC_DISPATCHIS_202604150005_...zip → 202604
    """
    try:
        for part in filename.upper().replace(".ZIP", "").split("_"):
            if len(part) >= 8 and part.isdigit():
                return part[:6]
    except Exception:
        pass
    return None


def _upsert_nem_rows(rows, source=""):
    """Insert or update NEM dispatch rows into raw_nem_data."""
    if not rows:
        raise RuntimeError(
            f"No usable dispatch rows found from {source}. "
            "Check your internet connection or run inspect_archive_zip.py."
        )

    insert_query = """
        INSERT INTO raw_nem_data (
            settlement_date, region_id, demand_mw, dispatch_price_mwh
        )
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            demand_mw          = VALUES(demand_mw),
            dispatch_price_mwh = VALUES(dispatch_price_mwh);
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.executemany(insert_query, rows)
    conn.commit()
    print(f"[Ingestion] Upserted {cursor.rowcount} AEMO rows ({source}).")
    cursor.close()
    conn.close()


def ingest_aemo_current_month():
    """Ingest the most recent live DispatchIS files from NEMWeb CURRENT."""
    print("[Ingestion] Downloading latest AEMO DispatchIS files (CURRENT)...")

    zip_urls = _get_zip_urls(AEMO_DISPATCHIS_URL)
    if not zip_urls:
        raise RuntimeError("No zip files found on NEMWeb CURRENT page.")

    rows = []
    for zip_url, filename in zip_urls[-100:]:
        print(f"[Ingestion] Reading {filename}")
        rows.extend(_fetch_and_parse_zip(zip_url))

    _upsert_nem_rows(rows, source="CURRENT")


def ingest_aemo_historical(months=None):
    """
    Ingest historical DispatchIS zips from the NEMWeb archive.

    Each daily zip contains 288 inner zips (one per 5-min interval).
    We sample AEMO_ARCHIVE_DAYS_PER_MONTH days per month evenly spaced.

    Parameters
    ----------
    months : list[str] | None
        YYYYMM strings, e.g. ['202602', '202603', '202604'].
        Defaults to AEMO_ARCHIVE_MONTHS from config.
    """
    from collections import defaultdict

    if months is None:
        months = AEMO_ARCHIVE_MONTHS

    days_per_month = AEMO_ARCHIVE_DAYS_PER_MONTH

    print(f"[Ingestion] Fetching archive index: {AEMO_ARCHIVE_BASE_URL}")
    print(f"[Ingestion] Target months: {months} ({days_per_month} days each)")

    all_urls = _get_zip_urls(AEMO_ARCHIVE_BASE_URL)
    print(f"[Ingestion] Found {len(all_urls)} total zip files in archive.")

    if not all_urls:
        raise RuntimeError(
            f"No zip files found at {AEMO_ARCHIVE_BASE_URL}. "
            "Open that URL in your browser to confirm access."
        )

    available_months = sorted({
        _month_from_filename(fname) for _, fname in all_urls
        if _month_from_filename(fname)
    })
    print(f"[Ingestion] Available months: {available_months}")

    filtered = [
        (url, fname) for url, fname in all_urls
        if _month_from_filename(fname) in months
    ]

    if not filtered:
        raise RuntimeError(
            f"No archive files found for months {months}. "
            f"Available months are: {available_months}. "
            "Update AEMO_ARCHIVE_MONTHS in your .env to match."
        )

    # Sample evenly within each month
    by_month = defaultdict(list)
    for url, fname in filtered:
        by_month[_month_from_filename(fname)].append((url, fname))

    sampled = []
    for month, entries in sorted(by_month.items()):
        step = max(1, len(entries) // days_per_month)
        picked = entries[::step][:days_per_month]
        print(f"  {month}: {len(entries)} files → sampling {len(picked)} days")
        sampled.extend(picked)

    print(f"\n[Ingestion] Downloading {len(sampled)} daily zips...")

    all_rows = []
    total = len(sampled)
    for i, (zip_url, filename) in enumerate(sampled, 1):
        print(f"[Ingestion] [{i}/{total}] {filename} ...")
        rows = _fetch_and_parse_zip(zip_url)
        all_rows.extend(rows)
        print(f"  → {len(rows)} rows parsed, running total: {len(all_rows)}")

    print(f"\n[Ingestion] Total rows parsed: {len(all_rows)}")
    _upsert_nem_rows(all_rows, source="ARCHIVE")


def ingest_bom_weather():
    print("[Ingestion] Downloading BOM weather observations...")

    insert_query = """
        INSERT INTO raw_weather_data (observation_date, station_name, temperature_c)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            temperature_c = VALUES(temperature_c);
    """

    rows = []
    for _region_id, station_config in BOM_STATIONS.items():
        response = requests.get(
            station_config["url"],
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        for obs in response.json()["observations"]["data"]:
            if obs.get("air_temp") is None:
                continue
            dt_obj = datetime.strptime(
                str(obs["local_date_time_full"]), "%Y%m%d%H%M%S"
            )
            rows.append((
                dt_obj.strftime("%Y-%m-%d %H:%M:%S"),
                station_config["station_name"],
                float(obs["air_temp"]),
            ))

    if not rows:
        raise RuntimeError("No valid BOM weather observations found.")

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.executemany(insert_query, rows)
    conn.commit()
    print(f"[Ingestion] Upserted {cursor.rowcount} BOM weather rows.")
    cursor.close()
    conn.close()