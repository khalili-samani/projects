"""Integration test for discovery, download and manifest persistence."""

from __future__ import annotations

from pathlib import Path

import httpx

from qld_surgery_optimiser.config import (
    AppSettings,
    BaseConfig,
)
from qld_surgery_optimiser.ingestion.ckan_client import CkanClient
from qld_surgery_optimiser.ingestion.downloader import ResourceDownloader
from qld_surgery_optimiser.ingestion.manifest import RawManifest


def _base_config() -> BaseConfig:
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
                    "allowed_formats": ["CSV"],
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
                "percentage_minimum": 0.0,
                "percentage_maximum": 100.0,
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


def _csv_payload(
    *,
    service_column: str,
    service_value: str,
) -> bytes:
    return (
        "Facility_Code,Facility_Name,Report_Month,"
        f"{service_column},Vol_Treated,Vol_Waiting,"
        "Vol_LongWaits\n"
        "101,Example Hospital,2025-06,"
        f"{service_value},20,40,5\n"
    ).encode("utf-8")


def test_discovery_download_and_manifest(
    tmp_path: Path,
) -> None:
    """Two source families should be persisted with lineage."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        if request.url.path.endswith("/package_show"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "id": "dataset-id",
                        "name": "elective-surgery",
                        "title": "Elective surgery",
                        "license_title": (
                            "Creative Commons Attribution 4.0"
                        ),
                        "organization": {
                            "title": "Queensland Health"
                        },
                        "resources": [
                            {
                                "id": "category-1",
                                "package_id": "dataset-id",
                                "name": (
                                    "June 2025 – Elective Surgery "
                                    "by Category – Summary 1"
                                ),
                                "format": "CSV",
                                "url": (
                                    "https://example.test/"
                                    "category.csv"
                                ),
                            },
                            {
                                "id": "specialty-1",
                                "package_id": "dataset-id",
                                "name": (
                                    "June 2025 – Elective Surgery "
                                    "by Speciality – Summary 2"
                                ),
                                "format": "CSV",
                                "url": (
                                    "https://example.test/"
                                    "specialty.csv"
                                ),
                            },
                        ],
                    },
                },
            )

        if request.url.path.endswith("/category.csv"):
            return httpx.Response(
                200,
                content=_csv_payload(
                    service_column="Category",
                    service_value="1",
                ),
                headers={"content-type": "text/csv"},
            )

        if request.url.path.endswith("/specialty.csv"):
            return httpx.Response(
                200,
                content=_csv_payload(
                    service_column="Specialty_Desc",
                    service_value="General Surgery",
                ),
                headers={"content-type": "text/csv"},
            )

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    config = _base_config()
    settings = AppSettings(
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data/raw",
        interim_data_dir=tmp_path / "data/interim",
        processed_data_dir=tmp_path / "data/processed",
        quarantine_data_dir=tmp_path / "data/quarantine",
        reports_dir=tmp_path / "reports",
        duckdb_path=(
            tmp_path / "data/processed/test.duckdb"
        ),
    )

    with CkanClient(
        config.sources.queensland_open_data,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="integration-test",
        transport=transport,
    ) as ckan:
        dataset, resources = ckan.get_dataset()

    assert len(resources) == 2

    manifest = RawManifest(
        settings.raw_data_dir / "manifest.csv"
    )

    with ResourceDownloader(
        raw_data_dir=settings.raw_data_dir,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="integration-test",
        transport=transport,
    ) as downloader:
        for resource in resources:
            result = downloader.download(resource)

            manifest.add(
                dataset=dataset,
                result=result,
            )

    category_files = list(
        (
            settings.raw_data_dir
            / "category"
            / "category-1"
        ).glob("*.csv")
    )

    specialty_files = list(
        (
            settings.raw_data_dir
            / "specialty"
            / "specialty-1"
        ).glob("*.csv")
    )

    assert len(category_files) == 1
    assert len(specialty_files) == 1

    manifest_rows = manifest.read()

    assert len(manifest_rows) == 2
    assert {
        row["resource_kind"]
        for row in manifest_rows
    } == {"category", "specialty"}