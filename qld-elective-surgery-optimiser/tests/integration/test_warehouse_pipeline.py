"""Integration test for canonical warehouse construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb
import pandas as pd

from qld_surgery_optimiser.config import (
    AppSettings,
)
from qld_surgery_optimiser.processing.warehouse import (
    build_warehouse,
)


def test_build_warehouse_from_validated_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Validated source data should produce a reconciled DuckDB model."""
    monkeypatch.chdir(
        tmp_path
    )

    sql_directory = (
        tmp_path / "sql"
    )

    sql_directory.mkdir(
        parents=True
    )

    project_sql = Path(
        __file__
    ).parents[2] / "sql/create_warehouse.sql"

    # When running from the real repository this path exists.
    # For an isolated test, provide equivalent DDL directly.
    ddl = """
    CREATE TABLE dim_facility (
        facility_key VARCHAR,
        facility_code VARCHAR,
        facility_name VARCHAR,
        hhs VARCHAR,
        region VARCHAR,
        resolution_status VARCHAR
    );

    CREATE TABLE dim_specialty (
        specialty_key VARCHAR,
        specialty_code VARCHAR,
        specialty_name VARCHAR
    );

    CREATE TABLE dim_urgency_category (
        urgency_category_key VARCHAR,
        urgency_category_name VARCHAR
    );

    CREATE TABLE dim_reporting_period (
        reporting_period_key DATE,
        calendar_year INTEGER,
        calendar_quarter INTEGER,
        month INTEGER,
        quarter_label VARCHAR
    );

    CREATE TABLE dim_source_resource (
        source_resource_key VARCHAR,
        resource_id VARCHAR,
        source_sha256 VARCHAR,
        source_url VARCHAR,
        source_file VARCHAR,
        retrieved_at TIMESTAMPTZ
    );

    CREATE TABLE fact_elective_surgery_performance (
        record_id VARCHAR,
        facility_key VARCHAR,
        reporting_period_key DATE,
        resource_kind VARCHAR,
        specialty_key VARCHAR,
        urgency_category_key VARCHAR,
        vol_treated DOUBLE,
        pct_treated_in_time DOUBLE,
        pct_variation_treated_prior_year DOUBLE,
        vol_waiting DOUBLE,
        vol_long_waits DOUBLE,
        pct_waiting_in_time_total DOUBLE,
        vol_long_waits_rfs DOUBLE,
        vol_long_waits_nrfs DOUBLE,
        pct_waiting_in_time_rfs DOUBLE,
        previous_vol_waiting DOUBLE,
        backlog_change DOUBLE,
        previous_vol_long_waits DOUBLE,
        long_wait_change DOUBLE,
        long_wait_share DOUBLE,
        treatment_to_waiting_ratio DOUBLE,
        data_last_update TIMESTAMP,
        source_resource_key VARCHAR
    );

    CREATE TABLE fact_data_quality_event (
        event_id VARCHAR,
        source_path VARCHAR,
        resource_kind VARCHAR,
        rule_id VARCHAR,
        severity VARCHAR,
        row_index BIGINT,
        column_name VARCHAR,
        observed_value VARCHAR,
        message VARCHAR
    );
    """

    (
        sql_directory
        / "create_warehouse.sql"
    ).write_text(
        ddl,
        encoding="utf-8",
    )

    raw_directory = (
        tmp_path
        / "data/raw"
    )

    interim_directory = (
        tmp_path
        / "data/interim/specialty"
    )

    reference_directory = (
        tmp_path
        / "data/reference"
    )

    report_directory = (
        tmp_path
        / "reports/outputs"
    )

    raw_directory.mkdir(
        parents=True
    )

    interim_directory.mkdir(
        parents=True
    )

    reference_directory.mkdir(
        parents=True
    )

    report_directory.mkdir(
        parents=True
    )

    raw_stem = (
        "abcdef1234567890_source"
    )

    validated_path = (
        interim_directory
        / f"{raw_stem}_validated.parquet"
    )

    validated = pd.DataFrame(
        {
            "Facility_Code": ["101"],
            "Facility_Name": [
                "Example Hospital"
            ],
            "Report_Month": [
                pd.Timestamp("2025-09-01")
            ],
            "Specialty_Code": ["GS"],
            "Specialty_Desc": [
                "General Surgery"
            ],
            "Vol_Treated": [50],
            "Vol_Waiting": [100],
            "Vol_LongWaits": [10],
            "Percent_Treated_InTime": [90],
            "data_last_update": [
                pd.Timestamp(
                    "2025-10-01"
                )
            ],
        }
    )

    validated.to_parquet(
        validated_path,
        index=False,
    )

    manifest_path = (
        raw_directory
        / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "resource_id",
                "source_url",
                "retrieved_at",
                "local_path",
                "sha256",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "resource_id": "resource-1",
                "source_url": (
                    "https://example.test/source.csv"
                ),
                "retrieved_at": (
                    "2025-10-02T00:00:00Z"
                ),
                "local_path": (
                    "data/raw/specialty/"
                    "resource-1/"
                    f"{raw_stem}.csv"
                ),
                "sha256": "a" * 64,
            }
        )

    (
        reference_directory
        / "facility_aliases.csv"
    ).write_text(
        (
            "alias_name,canonical_name,"
            "canonical_code,hhs,region,active\n"
        ),
        encoding="utf-8",
    )

    (
        report_directory
        / "data_quality_summary.json"
    ).write_text(
        json.dumps(
            {
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    settings = AppSettings(
        data_dir=tmp_path / "data",
        raw_data_dir=raw_directory,
        interim_data_dir=(
            tmp_path / "data/interim"
        ),
        processed_data_dir=(
            tmp_path / "data/processed"
        ),
        quarantine_data_dir=(
            tmp_path / "data/quarantine"
        ),
        reports_dir=(
            tmp_path / "reports"
        ),
        duckdb_path=(
            tmp_path
            / "data/processed/test.duckdb"
        ),
        facility_aliases_path=(
            reference_directory
            / "facility_aliases.csv"
        ),
    )

    summary = build_warehouse(
        settings=settings
    )

    assert summary.canonical_rows == 1
    assert summary.longitudinal_rows == 1
    assert summary.facility_count == 1
    assert summary.specialty_count == 1

    with duckdb.connect(
        str(settings.duckdb_path),
        read_only=True,
    ) as connection:
        fact_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM fact_elective_surgery_performance
            """
        ).fetchone()[0]

        facility_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM dim_facility
            """
        ).fetchone()[0]

    assert fact_count == 1
    assert facility_count == 1

    reconciliation = json.loads(
        summary.reconciliation_report_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        reconciliation[
            "fact_matches_longitudinal"
        ]
        is True
    )