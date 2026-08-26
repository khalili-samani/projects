"""Integration tests for raw-to-validated processing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qld_surgery_optimiser.config import (
    AppSettings,
    BaseConfig,
)
from qld_surgery_optimiser.validation.pipeline import (
    run_validation,
)


def _config() -> BaseConfig:
    return BaseConfig.model_validate(
        {
            "project": {
                "name": "qld-elective-surgery-optimiser",
                "version": "0.1.0",
                "geography": "Queensland, Australia",
                "decision_scope": (
                    "aggregate-health-service-planning"
                ),
                "patient_level_use_permitted": False,
            },
            "sources": {
                "queensland_open_data": {
                    "organisation": "Queensland Government",
                    "api_base_url": (
                        "https://example.test/api/3/action"
                    ),
                    "dataset_id": "elective-surgery",
                    "allowed_formats": [
                        "CSV"
                    ],
                    "category_patterns": [
                        "summary 1",
                        "by category",
                        "bycat",
                    ],
                    "specialty_patterns": [
                        "summary 2",
                        "by speciality",
                        "by specialty",
                        "byspeciality",
                        "byspecialty",
                    ],
                    "include_historical_resources": True,
                }
            },
            "storage": {
                "raw_format": "csv",
                "processed_format": "parquet",
                "preserve_raw_files": True,
                "calculate_sha256": True,
                "overwrite_raw_files": False,
            },
            "validation": {
                "fail_on_missing_required_columns": True,
                "fail_on_duplicate_business_keys": True,
                "quarantine_invalid_rows": True,
                "allow_unexpected_columns": True,
                "maximum_unresolved_entity_rate": 0.01,
                "percentage_minimum": 0,
                "percentage_maximum": 100,
                "null_tokens": [
                    "",
                    "NA",
                    "N/A",
                    "NULL",
                    "null",
                    "-",
                    "--",
                ],
                "required_common_columns": [
                    "Facility_Code",
                    "Facility_Name",
                    "Report_Month",
                    "Vol_Treated",
                    "Vol_Waiting",
                    "Vol_LongWaits",
                ],
                "category_identity_columns": [
                    "Category"
                ],
                "specialty_identity_columns": [
                    "Specialty_Code",
                    "Specialty_Desc",
                ],
                "numeric_volume_columns": [
                    "Vol_Treated",
                    "Vol_Waiting",
                    "Vol_LongWaits",
                    "Vol_LongWaits_RFS",
                    "Vol_LongWaits_NRFS",
                ],
                "percentage_columns": [
                    "Percent_Treated_InTime",
                    "Percent_Variation_Treated_Prior_Year",
                    "Percent_Waiting_InTime_Total",
                    "Percent_Waiting_InTime_RFS",
                ],
                "date_columns": [
                    "Report_Month",
                    "data_last_update",
                ],
            },
            "warehouse": {
                "database_schema": "main",
                "replace_derived_tables": True,
                "preserve_run_metadata": True,
            },
            "reporting": {
                "include_data_freshness": True,
                "include_scenario_provenance": True,
                "include_solver_status": True,
                "include_limitations": True,
                "include_responsible_use_notice": True,
            },
        }
    )


def test_validation_separates_valid_and_invalid_rows(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path
        / "data/raw/specialty/resource-1"
    )

    raw_directory.mkdir(
        parents=True
    )

    raw_path = (
        raw_directory
        / "abc_source.csv"
    )

    source = pd.DataFrame(
        {
            "Facility_Code": [
                "101",
                "102",
                "103",
            ],
            "Facility_Name": [
                "Hospital A",
                "Hospital B",
                "Hospital C",
            ],
            "Report_Month": [
                "2025-06-01",
                "2025-06-01",
                "2025-06-01",
            ],
            "Specialty_Code": [
                "GS",
                "OR",
                "ENT",
            ],
            "Specialty_Desc": [
                "General Surgery",
                "Orthopaedics",
                "ENT",
            ],
            "Vol_Treated": [
                "20",
                "15",
                "10",
            ],
            "Vol_Waiting": [
                "100",
                "10",
                "40",
            ],
            "Vol_LongWaits": [
                "10",
                "15",
                "5",
            ],
            "Percent_Treated_InTime": [
                "95%",
                "90%",
                "110%",
            ],
        }
    )

    source.to_csv(
        raw_path,
        index=False,
    )

    settings = AppSettings(
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data/raw",
        interim_data_dir=tmp_path / "data/interim",
        processed_data_dir=tmp_path / "data/processed",
        quarantine_data_dir=tmp_path / "data/quarantine",
        reports_dir=tmp_path / "reports",
        duckdb_path=(
            tmp_path
            / "data/processed/test.duckdb"
        ),
    )

    summary = run_validation(
        settings=settings,
        base_config=_config(),
    )

    assert summary.files_processed == 1
    assert summary.rows_read == 3

    assert summary.rows_valid == 1
    assert summary.rows_quarantined == 2

    validated_files = list(
        (
            settings.interim_data_dir
            / "specialty"
        ).glob("*.parquet")
    )

    quarantined_files = list(
        (
            settings.quarantine_data_dir
            / "specialty"
        ).glob("*.parquet")
    )

    assert len(validated_files) == 1
    assert len(quarantined_files) == 1

    valid = pd.read_parquet(
        validated_files[0]
    )

    quarantine = pd.read_parquet(
        quarantined_files[0]
    )

    assert len(valid) == 1
    assert len(quarantine) == 2

    report = json.loads(
        summary.report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report["rows_quarantined"] == 2

    assert (
        report["rule_counts"][
            "LONG_WAITS_EXCEED_WAITING"
        ]
        == 1
    )

    assert (
        report["rule_counts"][
            "PERCENTAGE_OUT_OF_RANGE"
        ]
        == 1
    )