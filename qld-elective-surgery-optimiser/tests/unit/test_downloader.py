"""Tests for verified raw-resource downloading."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from qld_surgery_optimiser.exceptions import DownloadError
from qld_surgery_optimiser.ingestion.downloader import (
    ResourceDownloader,
)
from qld_surgery_optimiser.ingestion.models import ResourceRef


def _resource() -> ResourceRef:
    return ResourceRef(
        resource_id="resource-123",
        package_id="dataset-123",
        name="June 2025 by Specialty Summary 2",
        resource_kind="specialty",
        format="CSV",
        download_url=(
            "https://example.test/"
            "es_quarterly_byspecialty_jun25.csv"
        ),
    )


def _valid_csv() -> bytes:
    return (
        "Facility_Code,Facility_Name,Report_Month,"
        "Specialty_Code,Specialty_Desc,Vol_Treated,"
        "Vol_Waiting,Vol_LongWaits\n"
        "101,Example Hospital,2025-06,"
        "01,General Surgery,50,100,10\n"
    ).encode("utf-8")


def test_downloads_and_versions_csv_by_checksum(
    tmp_path: Path,
) -> None:
    """Raw filenames should include a content-derived version."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            content=_valid_csv(),
            headers={
                "content-type": "text/csv"
            },
        )

    downloader = ResourceDownloader(
        raw_data_dir=tmp_path,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="test",
        transport=httpx.MockTransport(handler),
    )

    with downloader:
        result = downloader.download(_resource())

    assert result.downloaded is True
    assert result.local_path.exists()
    assert result.local_path.parent.name == "resource-123"
    assert (
        result.local_path.parent.parent.name
        == "specialty"
    )
    assert result.sha256[:16] in result.local_path.name


def test_repeated_identical_content_is_not_rewritten(
    tmp_path: Path,
) -> None:
    """A checksum-identical source should resolve to the same raw file."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            content=_valid_csv(),
            headers={
                "content-type": "application/csv"
            },
        )

    with ResourceDownloader(
        raw_data_dir=tmp_path,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="test",
        transport=httpx.MockTransport(handler),
    ) as downloader:
        first = downloader.download(_resource())
        second = downloader.download(_resource())

    assert first.downloaded is True
    assert second.downloaded is False
    assert first.local_path == second.local_path
    assert first.sha256 == second.sha256


def test_rejects_html_error_page(
    tmp_path: Path,
) -> None:
    """An HTML page returned with HTTP 200 must not be stored as CSV."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html><body>Error</body></html>",
            headers={"content-type": "text/html"},
        )

    with ResourceDownloader(
        raw_data_dir=tmp_path,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="test",
        transport=httpx.MockTransport(handler),
    ) as downloader:
        with pytest.raises(
            DownloadError,
            match="HTML",
        ):
            downloader.download(_resource())


def test_rejects_csv_missing_identity_columns(
    tmp_path: Path,
) -> None:
    """Unrelated CSV files must fail the source identity check."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"name,value\none,1\n",
            headers={"content-type": "text/csv"},
        )

    with ResourceDownloader(
        raw_data_dir=tmp_path,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="test",
        transport=httpx.MockTransport(handler),
    ) as downloader:
        with pytest.raises(
            DownloadError,
            match="mandatory identity columns",
        ):
            downloader.download(_resource())