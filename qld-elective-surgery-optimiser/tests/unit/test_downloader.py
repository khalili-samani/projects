"""Tests for immutable raw-resource downloading and source layouts."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from qld_surgery_optimiser.exceptions import DownloadError
from qld_surgery_optimiser.ingestion.downloader import ResourceDownloader
from qld_surgery_optimiser.ingestion.models import ResourceRef
from qld_surgery_optimiser.ingestion.source_layout import (
    CATEGORY_COLUMNS,
    LEGACY_CATEGORY_COLUMNS,
    SPECIALTY_COLUMNS,
)


def _resource(
    *,
    resource_kind: str = "specialty",
) -> ResourceRef:
    return ResourceRef(
        resource_id="resource-123",
        package_id="dataset-123",
        name="Elective Surgery Source",
        resource_kind=resource_kind,
        format="CSV",
        download_url=(
            "https://example.test/elective_surgery.csv"
        ),
    )


def _valid_csv() -> bytes:
    return (
        ",".join(SPECIALTY_COLUMNS)
        + "\n"
        + (
            "101,Example Hospital,Jun-25,1,"
            "General Surgery,50,100,0,100,10,90,"
            "2025-07-01,5,5,95\n"
        )
    ).encode("utf-8")


def _category_row() -> bytes:
    return (
        b"211,Atherton Hospital,Dec-23,1,16,100,6.7,4,1,75,"
        b" 16-Jan-2024,1,0,66.7\n"
    )


def _specialty_row() -> bytes:
    return (
        b"211,Atherton Hospital,Dec-23,1,cardio-thoracic surgery,"
        b"0,-,0,0,0,-,16-Jan-2024,0,0,-\n"
    )


def _padded_category_row() -> bytes:
    return (
        b"211,Atherton Hospital,Jun-22,1,16,81.3,-27.3,6,0,100,"
        b" 07-Jul-2022,0,0,100,,,\n"
    )


def _downloader(
    tmp_path: Path,
    transport: httpx.MockTransport,
) -> ResourceDownloader:
    return ResourceDownloader(
        raw_data_dir=tmp_path,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0,
        user_agent="test",
        transport=transport,
    )


def _transport_for(
    payload: bytes,
    *,
    content_type: str = "text/csv",
) -> httpx.MockTransport:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={
                "content-type": content_type,
            },
        )

    return httpx.MockTransport(handler)


def test_downloads_and_versions_csv_by_checksum(
    tmp_path: Path,
) -> None:
    """Raw filenames include a checksum-derived version."""
    with _downloader(
        tmp_path,
        _transport_for(_valid_csv()),
    ) as downloader:
        result = downloader.download(_resource())

    assert result.downloaded is True
    assert result.local_path.exists()
    assert result.local_path.parent.name == "resource-123"
    assert result.local_path.parent.parent.name == "specialty"
    assert result.sha256[:16] in result.local_path.name


def test_repeated_identical_content_is_not_rewritten(
    tmp_path: Path,
) -> None:
    """Identical source bytes reuse the existing immutable file."""
    with _downloader(
        tmp_path,
        _transport_for(
            _valid_csv(),
            content_type="application/csv",
        ),
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
    """HTML masquerading as a CSV must not be persisted."""
    with _downloader(
        tmp_path,
        _transport_for(
            b"<html><body>Error</body></html>",
            content_type="text/html",
        ),
    ) as downloader, pytest.raises(
        DownloadError,
        match="HTML",
    ):
        downloader.download(_resource())


def test_rejects_csv_missing_identity_columns(
    tmp_path: Path,
) -> None:
    """Unrelated CSV files must fail source-layout recognition."""
    with _downloader(
        tmp_path,
        _transport_for(
            b"name,value\none,1\n",
        ),
    ) as downloader, pytest.raises(
        DownloadError,
        match="mandatory identity columns",
    ):
        downloader.download(_resource())


def test_accepts_headerless_category_without_rewriting(
    tmp_path: Path,
) -> None:
    """Preserve the original 14-column category source bytes."""
    payload = _category_row()

    with _downloader(
        tmp_path,
        _transport_for(payload),
    ) as downloader:
        result = downloader.download(
            _resource(resource_kind="category")
        )

    assert result.downloaded
    assert result.local_path.read_bytes() == payload


def test_accepts_17_column_category_with_empty_padding(
    tmp_path: Path,
) -> None:
    """Accept the verified 17-column legacy category variant."""
    payload = _padded_category_row() * 216

    with _downloader(
        tmp_path,
        _transport_for(payload),
    ) as downloader:
        result = downloader.download(
            _resource(resource_kind="category")
        )

    assert result.downloaded
    assert result.local_path.read_bytes() == payload


def test_rejects_populated_category_padding(
    tmp_path: Path,
) -> None:
    """Never silently discard meaningful trailing fields."""
    payload = (
        _padded_category_row()
        + (
            b"211,Atherton Hospital,Jun-22,2,19,100,-42.4,"
            b"15,0,100,07-Jul-2022,0,0,100,important,,\n"
        )
    )

    with _downloader(
        tmp_path,
        _transport_for(payload),
    ) as downloader, pytest.raises(
        DownloadError,
        match="non-empty trailing fields",
    ):
        downloader.download(
            _resource(resource_kind="category")
        )


def test_accepts_legacy_headered_category(
    tmp_path: Path,
) -> None:
    """Accept June 2023 category names without changing raw bytes."""
    payload = (
        ",".join(LEGACY_CATEGORY_COLUMNS)
        + "\n"
        + _category_row().decode("utf-8")
    ).encode("utf-8")

    with _downloader(
        tmp_path,
        _transport_for(payload),
    ) as downloader:
        result = downloader.download(
            _resource(resource_kind="category")
        )

    assert result.downloaded
    assert result.local_path.read_bytes() == payload


def test_accepts_modern_headered_category(
    tmp_path: Path,
) -> None:
    """Retain support for current canonical category headers."""
    payload = (
        ",".join(CATEGORY_COLUMNS)
        + "\n"
        + _category_row().decode("utf-8")
    ).encode("utf-8")

    with _downloader(
        tmp_path,
        _transport_for(payload),
    ) as downloader:
        result = downloader.download(
            _resource(resource_kind="category")
        )

    assert result.downloaded
    assert result.local_path.read_bytes() == payload


def test_accepts_headerless_specialty(
    tmp_path: Path,
) -> None:
    """Accept the verified 15-column headerless specialty format."""
    payload = _specialty_row()

    with _downloader(
        tmp_path,
        _transport_for(payload),
    ) as downloader:
        result = downloader.download(
            _resource(resource_kind="specialty")
        )

    assert result.downloaded
    assert result.local_path.read_bytes() == payload


def test_rejects_wrong_width_specialty_layout(
    tmp_path: Path,
) -> None:
    """Do not interpret a category record as specialty data."""
    with _downloader(
        tmp_path,
        _transport_for(_category_row()),
    ) as downloader, pytest.raises(
        DownloadError,
        match="unsupported source layout",
    ):
        downloader.download(
            _resource(resource_kind="specialty")
        )