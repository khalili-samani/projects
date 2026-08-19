"""Integration test for the discovery-to-manifest workflow."""

from __future__ import annotations

from pathlib import Path

import httpx

from qld_surgery_optimiser.config import (
    AppSettings,
    BaseConfig,
)
from qld_surgery_optimiser.ingestion import pipeline
from qld_surgery_optimiser.ingestion.ckan_client import CkanClient
from qld_surgery_optimiser.ingestion.downloader import ResourceDownloader


def _base_config() -> BaseConfig:
    return BaseConfig.model_validate(
        {
            "project": {
                "name": (
                    "qld-elective-surgery-optimiser"
                ),
                "version": "0.1.0",
                "geography": (
                    "Queensland, Australia"
                ),
                "decision_scope": (
                    "aggregate-health-service-planning"
                ),
                "patient_level_use_permitted": False,
            },
            "sources": {
                "queensland_open_data": {
                    "organisation": (
                        "Queensland Government"
                    ),
                    "api_base_url": (
                        "https://example.test/"
                        "api/3/action"
                    ),
                    "dataset_id": (
                        "elective-surgery"
                    ),
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
    ).encode()


def test_ingestion_pipeline_downloads_and_records_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The pipeline should persist both source families and a manifest."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        if request.url.path.endswith(
            "/package_show"
        ):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "id": "dataset-id",
                        "name": "elective-surgery",
                        "title": "Elective surgery",
                        "license_title": (
                            "Creative Commons "
                            "Attribution 4.0"
                        ),
                        "organization": {
                            "title": (
                                "Queensland Health"
                            )
                        },
                        "resources": [
                            {
                                "id": "category-1",
                                "package_id": "dataset-id",
                                "name": (
                                    "June 2025 "
                                    "by Category Summary 1"
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
                                    "June 2025 "
                                    "by Speciality Summary 2"
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

        if request.url.path.endswith(
            "/category.csv"
        ):
            return httpx.Response(
                200,
                content=_csv_payload(
                    service_column="Category",
                    service_value="1",
                ),
                headers={
                    "content-type": "text/csv"
                },
            )

        if request.url.path.endswith(
            "/specialty.csv"
        ):
            return httpx.Response(
                200,
                content=_csv_payload(
                    service_column="Specialty_Desc",
                    service_value="General Surgery",
                ),
                headers={
                    "content-type": "text/csv"
                },
            )

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    original_ckan_init = CkanClient.__init__
    original_downloader_init = ResourceDownloader.__init__

    def patched_ckan_init(
        self,
        config,
        *,
        timeout_seconds,
        max_retries,
        retry_backoff_seconds,
        user_agent,
        transport_override=None,
    ):
        original_ckan_init(
            self,
            config,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=(
                retry_backoff_seconds
            ),
            user_agent=user_agent,
            transport=transport,
        )

    def patched_downloader_init(
        self,
        *,
        raw_data_dir,
        timeout_seconds,
        max_retries,
        retry_backoff_seconds,
        user_agent,
        transport_override=None,
    ):
        original_downloader_init(
            self,
            raw_data_dir=raw_data_dir,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=(
                retry_backoff_seconds
            ),
            user_agent=user_agent,
            transport=transport,
        )

    monkeypatch.setattr(
        pipeline.CkanClient,
        "__init__",
        patched_ckan_init,
    )

    monkeypatch.setattr(
        pipeline.ResourceDownloader,
        "__init__",
        patched_downloader_init,
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

    summary = pipeline.run_ingestion(
        settings=settings,
        base_config=_base_config(),
    )

    assert summary.resources_selected == 2
    assert summary.resources_downloaded == 2
    assert summary.manifest_records_added == 2

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
    assert summary.manifest_path.exists()