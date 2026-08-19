"""Verified and immutable raw-resource downloading."""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from qld_surgery_optimiser.exceptions import DownloadError
from qld_surgery_optimiser.ingestion.models import (
    DownloadResult,
    ResourceRef,
)

logger = logging.getLogger(__name__)


_REQUIRED_HEADER_COLUMNS = {
    "Facility_Code",
    "Facility_Name",
    "Report_Month",
}


class ResourceDownloader:
    """Download CKAN resources with basic transport and content validation."""

    def __init__(
        self,
        *,
        raw_data_dir: Path,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        user_agent: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._raw_data_dir = raw_data_dir
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "Accept": "text/csv,application/csv,text/plain,*/*",
                "User-Agent": user_agent,
            },
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> ResourceDownloader:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def download(
        self,
        resource: ResourceRef,
    ) -> DownloadResult:
        """Retrieve, validate and persist one immutable resource."""
        response = self._request(resource)
        payload = response.content

        content_type = response.headers.get("content-type")

        self._validate_payload(
            payload=payload,
            content_type=content_type,
            resource=resource,
        )

        sha256 = hashlib.sha256(payload).hexdigest()

        target_path = self._target_path(
            resource=resource,
            sha256=sha256,
        )

        retrieved_at = datetime.now(UTC)

        downloaded = self._write_if_new(
            path=target_path,
            payload=payload,
        )

        logger.info(
            "Raw resource ready",
            extra={
                "resource_id": resource.resource_id,
                "resource_kind": resource.resource_kind,
                "local_path": str(target_path),
                "sha256": sha256,
                "byte_count": len(payload),
                "downloaded": downloaded,
            },
        )

        return DownloadResult(
            resource=resource,
            local_path=target_path,
            sha256=sha256,
            byte_count=len(payload),
            retrieved_at=retrieved_at,
            content_type=content_type,
            downloaded=downloaded,
        )

    def _request(
        self,
        resource: ResourceRef,
    ) -> httpx.Response:
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(resource.download_url)
                response.raise_for_status()
                return response

            except httpx.HTTPError as exc:
                last_error = exc

                if attempt >= self._max_retries:
                    break

                delay = self._retry_backoff_seconds * (2**attempt)

                logger.warning(
                    "Resource download failed; retrying",
                    extra={
                        "resource_id": resource.resource_id,
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )

                time.sleep(delay)

        raise DownloadError(
            f"Failed to download resource "
            f"{resource.resource_id} after "
            f"{self._max_retries + 1} attempt(s)."
        ) from last_error

    def _validate_payload(
        self,
        *,
        payload: bytes,
        content_type: str | None,
        resource: ResourceRef,
    ) -> None:
        if not payload:
            raise DownloadError(
                f"Resource {resource.resource_id} returned an empty body."
            )

        normalised_content_type = (
            content_type.casefold()
            if content_type is not None
            else ""
        )

        if "text/html" in normalised_content_type:
            raise DownloadError(
                f"Resource {resource.resource_id} returned HTML "
                "instead of CSV."
            )

        prefix = payload[:512].lstrip().casefold()

        html_markers = (
            b"<!doctype html",
            b"<html",
            b"<head",
            b"<body",
        )

        if any(prefix.startswith(marker) for marker in html_markers):
            raise DownloadError(
                f"Resource {resource.resource_id} appears to contain HTML "
                "instead of CSV."
            )

        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DownloadError(
                f"Resource {resource.resource_id} is not valid UTF-8 CSV."
            ) from exc

        try:
            reader = csv.reader(io.StringIO(text))
            header = next(reader)
        except (csv.Error, StopIteration) as exc:
            raise DownloadError(
                f"Resource {resource.resource_id} has no readable CSV header."
            ) from exc

        stripped_header = {
            column.strip()
            for column in header
            if column.strip()
        }

        missing = _REQUIRED_HEADER_COLUMNS - stripped_header

        if missing:
            missing_text = ", ".join(sorted(missing))

            raise DownloadError(
                f"Resource {resource.resource_id} is missing mandatory "
                f"identity columns: {missing_text}"
            )

    def _target_path(
        self,
        *,
        resource: ResourceRef,
        sha256: str,
    ) -> Path:
        filename = _filename_from_url(resource.download_url)

        directory = (
            self._raw_data_dir
            / resource.resource_kind
            / resource.resource_id
        )

        versioned_filename = f"{sha256[:16]}_{filename}"

        return directory / versioned_filename

    def _write_if_new(
        self,
        *,
        path: Path,
        payload: bytes,
    ) -> bool:
        if path.exists():
            return False

        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")

        try:
            temporary_path.write_bytes(payload)
            temporary_path.replace(path)
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)

            raise DownloadError(
                f"Could not persist raw resource to {path}"
            ) from exc

        return True


def _filename_from_url(url: str) -> str:
    """Create a safe local filename from a download URL."""
    parsed = urlsplit(url)
    raw_name = unquote(Path(parsed.path).name)

    if not raw_name:
        raw_name = "resource.csv"

    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        raw_name,
    ).strip("._")

    if not safe_name:
        safe_name = "resource.csv"

    if not safe_name.casefold().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    return safe_name