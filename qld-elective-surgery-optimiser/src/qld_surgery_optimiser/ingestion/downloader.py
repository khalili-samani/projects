"""Download and persist immutable Queensland elective-surgery source files."""

from __future__ import annotations

import csv
import hashlib
import io
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from qld_surgery_optimiser.exceptions import DownloadError
from qld_surgery_optimiser.ingestion.models import (
    DownloadResult,
    ResourceRef,
)


REQUIRED_IDENTITY_COLUMNS = frozenset(
    {
        "Facility_Code",
        "Facility_Name",
        "Report_Month",
    }
)


class ResourceDownloader:
    """Download, validate and persist source resources immutably."""

    def __init__(
        self,
        *,
        raw_data_dir: Path,
        timeout_seconds: int,
        max_retries: int,
        retry_backoff_seconds: float,
        user_agent: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialise the HTTP resource downloader."""

        self.raw_data_dir = Path(raw_data_dir)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.user_agent = user_agent

        self._client = httpx.Client(
            timeout=httpx.Timeout(
                timeout_seconds
            ),
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": (
                    "text/csv,"
                    "application/csv,"
                    "text/plain,"
                    "application/octet-stream,"
                    "*/*"
                ),
            },
            transport=transport,
        )

    def __enter__(
        self,
    ) -> ResourceDownloader:
        """Enter the downloader context manager."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close the underlying HTTP client."""

        self.close()

    def close(
        self,
    ) -> None:
        """Close the HTTP client."""

        self._client.close()

    def download(
        self,
        resource: ResourceRef,
    ) -> DownloadResult:
        """Download and persist one source resource.

        Raw files are content-versioned by SHA-256. If an identical
        payload has already been persisted, the existing immutable file
        is reused rather than rewritten.
        """

        response = self._request(
            resource=resource
        )

        payload = response.content

        content_type = response.headers.get(
            "content-type"
        )

        self._validate_payload(
            payload=payload,
            content_type=content_type,
            resource=resource,
        )

        sha256 = hashlib.sha256(
            payload
        ).hexdigest()

        retrieved_at = datetime.now(
            UTC
        )

        source_filename = (
            self._source_filename(
                resource=resource
            )
        )

        resource_directory = (
            self.raw_data_dir
            / resource.resource_kind
            / resource.resource_id
        )

        resource_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_path = (
            resource_directory
            / (
                f"{sha256[:16]}_"
                f"{source_filename}"
            )
        )

        downloaded = False

        if not local_path.exists():
            local_path.write_bytes(
                payload
            )

            downloaded = True

        return DownloadResult(
            resource=resource,
            local_path=local_path,
            sha256=sha256,
            byte_count=len(payload),
            retrieved_at=retrieved_at,
            content_type=content_type,
            downloaded=downloaded,
        )

    def _request(
        self,
        *,
        resource: ResourceRef,
    ) -> httpx.Response:
        """Retrieve one resource using bounded retry behaviour."""

        attempts = (
            self.max_retries
            + 1
        )

        last_error: Exception | None = None

        for attempt in range(
            attempts
        ):
            try:
                response = self._client.get(
                    resource.download_url
                )

                response.raise_for_status()

                return response

            except (
                httpx.HTTPError,
                httpx.TimeoutException,
            ) as exc:
                last_error = exc

                final_attempt = (
                    attempt
                    >= attempts - 1
                )

                if final_attempt:
                    break

                delay = (
                    self.retry_backoff_seconds
                    * (2**attempt)
                )

                if delay > 0:
                    time.sleep(
                        delay
                    )

        raise DownloadError(
            "Failed to download resource "
            f"{resource.resource_id} "
            f"from {resource.download_url} "
            f"after {attempts} attempt(s)."
        ) from last_error

    def _validate_payload(
        self,
        *,
        payload: bytes,
        content_type: str | None,
        resource: ResourceRef,
    ) -> None:
        """Validate that a downloaded payload is plausibly a source CSV."""

        if not payload:
            raise DownloadError(
                f"Resource {resource.resource_id} "
                "returned an empty body."
            )

        normalised_content_type = (
            content_type.casefold()
            if content_type is not None
            else ""
        )

        if "text/html" in normalised_content_type:
            raise DownloadError(
                f"Resource {resource.resource_id} "
                "returned HTML instead of CSV."
            )

        # `payload` is bytes. bytes supports `.lower()`, not
        # str.casefold().
        prefix = (
            payload[:512]
            .lstrip()
            .lower()
        )

        if (
            prefix.startswith(
                b"<!doctype html"
            )
            or prefix.startswith(
                b"<html"
            )
        ):
            raise DownloadError(
                f"Resource {resource.resource_id} "
                "returned HTML instead of CSV."
            )

        try:
            decoded = payload.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError as exc:
            raise DownloadError(
                f"Resource {resource.resource_id} "
                "could not be decoded as UTF-8 CSV."
            ) from exc

        try:
            reader = csv.reader(
                io.StringIO(
                    decoded
                )
            )

            header = next(
                reader,
                None,
            )

        except csv.Error as exc:
            raise DownloadError(
                f"Resource {resource.resource_id} "
                "could not be parsed as CSV."
            ) from exc

        if not header:
            raise DownloadError(
                f"Resource {resource.resource_id} "
                "did not contain a CSV header."
            )

        normalised_header = {
            column.strip()
            for column in header
            if column is not None
            and column.strip()
        }

        missing_columns = (
            REQUIRED_IDENTITY_COLUMNS
            - normalised_header
        )

        if missing_columns:
            missing = ", ".join(
                sorted(
                    missing_columns
                )
            )

            raise DownloadError(
                f"Resource {resource.resource_id} "
                "is missing mandatory identity "
                f"columns: {missing}."
            )

    @staticmethod
    def _source_filename(
        *,
        resource: ResourceRef,
    ) -> str:
        """Derive a safe filename from the source download URL."""

        parsed_url = urlparse(
            resource.download_url
        )

        filename = Path(
            unquote(
                parsed_url.path
            )
        ).name

        if not filename:
            filename = (
                f"{resource.resource_id}.csv"
            )

        filename = (
            ResourceDownloader
            ._sanitise_filename(
                filename
            )
        )

        if not filename.casefold().endswith(
            ".csv"
        ):
            filename = (
                f"{filename}.csv"
            )

        return filename

    @staticmethod
    def _sanitise_filename(
        filename: str,
    ) -> str:
        """Return a filesystem-safe source filename."""

        safe_characters: list[str] = []

        for character in filename:
            if (
                character.isalnum()
                or character
                in {
                    ".",
                    "-",
                    "_",
                }
            ):
                safe_characters.append(
                    character
                )
            else:
                safe_characters.append(
                    "_"
                )

        sanitised = "".join(
            safe_characters
        ).strip(
            "._"
        )

        if not sanitised:
            return "source.csv"

        return sanitised