"""Tests for raw-source lineage manifest persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from qld_surgery_optimiser.ingestion.manifest import RawManifest
from qld_surgery_optimiser.ingestion.models import (
    DatasetMetadata,
    DownloadResult,
    ResourceRef,
)


def _dataset() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="dataset-1",
        dataset_name="elective-surgery",
        title="Elective surgery",
        organisation="Queensland Health",
        licence_title=(
            "Creative Commons Attribution 4.0"
        ),
    )


def _result(
    local_path: Path,
) -> DownloadResult:
    resource = ResourceRef(
        resource_id="resource-1",
        package_id="dataset-1",
        name="June 2025 by Category Summary 1",
        resource_kind="category",
        format="CSV",
        download_url=(
            "https://example.test/bycat_jun25.csv"
        ),
        source_hash="source-md5",
    )

    return DownloadResult(
        resource=resource,
        local_path=local_path,
        sha256="a" * 64,
        byte_count=123,
        retrieved_at=datetime(
            2026,
            8,
            17,
            tzinfo=UTC,
        ),
        content_type="text/csv",
        downloaded=True,
    )


def test_manifest_records_retrieval(
    tmp_path: Path,
) -> None:
    manifest = RawManifest(
        tmp_path / "manifest.csv"
    )

    result = _result(
        tmp_path / "resource.csv"
    )

    added = manifest.add(
        dataset=_dataset(),
        result=result,
    )

    rows = manifest.read()

    assert added is True
    assert len(rows) == 1
    assert rows[0]["resource_id"] == "resource-1"
    assert rows[0]["sha256"] == "a" * 64
    assert (
        rows[0]["source_licence"]
        == "Creative Commons Attribution 4.0"
    )


def test_manifest_deduplicates_resource_checksum(
    tmp_path: Path,
) -> None:
    manifest = RawManifest(
        tmp_path / "manifest.csv"
    )

    result = _result(
        tmp_path / "resource.csv"
    )

    first = manifest.add(
        dataset=_dataset(),
        result=result,
    )

    second = manifest.add(
        dataset=_dataset(),
        result=result,
    )

    assert first is True
    assert second is False
    assert len(manifest.read()) == 1