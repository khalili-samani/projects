"""Tests for deterministic CKAN resource discovery."""

from __future__ import annotations

import httpx

from qld_surgery_optimiser.config import QueenslandOpenDataConfig
from qld_surgery_optimiser.ingestion.ckan_client import CkanClient


def _source_config() -> QueenslandOpenDataConfig:
    return QueenslandOpenDataConfig(
        organisation="Queensland Government",
        api_base_url="https://example.test/api/3/action",
        dataset_id="elective-surgery",
        allowed_formats=["CSV"],
        category_patterns=[
            "summary 1",
            "by category",
            "bycat",
        ],
        specialty_patterns=[
            "summary 2",
            "by speciality",
            "by specialty",
            "byspeciality",
            "byspecialty",
        ],
        include_historical_resources=True,
    )


def test_discovers_only_matching_csv_resources() -> None:
    """Description files and unrelated resources should be excluded."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.path.endswith(
            "/package_show"
        )

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
                                "bycat_jun25.csv"
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
                                "byspeciality_jun25.csv"
                            ),
                        },
                        {
                            "id": "description-1",
                            "package_id": "dataset-id",
                            "name": (
                                "Elective surgery summary "
                                "description"
                            ),
                            "format": "XLS",
                            "url": (
                                "https://example.test/"
                                "description.xls"
                            ),
                        },
                    ],
                },
            },
        )

    transport = httpx.MockTransport(handler)

    with CkanClient(
        _source_config(),
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="test",
        transport=transport,
    ) as client:
        dataset, resources = client.get_dataset()

    assert dataset.title == "Elective surgery"
    assert len(resources) == 2

    kinds = {
        resource.resource_kind
        for resource in resources
    }

    assert kinds == {"category", "specialty"}


def test_package_show_uses_configured_dataset_id() -> None:
    """Discovery should target the configured dataset directly."""

    captured_id: list[str] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_id.append(
            request.url.params["id"]
        )

        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {
                    "id": "elective-surgery-id",
                    "name": "elective-surgery",
                    "title": "Elective surgery",
                    "resources": [
                        {
                            "id": "resource-1",
                            "package_id": (
                                "elective-surgery-id"
                            ),
                            "name": (
                                "March 2025 "
                                "by Category Summary 1"
                            ),
                            "format": "CSV",
                            "url": (
                                "https://example.test/"
                                "bycat.csv"
                            ),
                        }
                    ],
                },
            },
        )

    with CkanClient(
        _source_config(),
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="test",
        transport=httpx.MockTransport(handler),
    ) as client:
        client.get_dataset()

    assert captured_id == ["elective-surgery"]