"""Persistent lineage manifest for raw source resources."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from qld_surgery_optimiser.exceptions import DownloadError
from qld_surgery_optimiser.ingestion.models import (
    DatasetMetadata,
    DownloadResult,
)

logger = logging.getLogger(__name__)


_MANIFEST_FIELDS = [
    "dataset_id",
    "dataset_title",
    "source_organisation",
    "source_licence",
    "resource_id",
    "resource_name",
    "resource_kind",
    "resource_format",
    "source_url",
    "source_hash",
    "source_created",
    "source_last_modified",
    "retrieved_at",
    "local_path",
    "sha256",
    "byte_count",
    "content_type",
]


@dataclass(frozen=True)
class ManifestRecord:
    """One immutable source-resource lineage record."""

    dataset_id: str
    dataset_title: str
    source_organisation: str
    source_licence: str
    resource_id: str
    resource_name: str
    resource_kind: str
    resource_format: str
    source_url: str
    source_hash: str
    source_created: str
    source_last_modified: str
    retrieved_at: str
    local_path: str
    sha256: str
    byte_count: int
    content_type: str


class RawManifest:
    """CSV-backed record of retrieved raw resources."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Return the manifest path."""
        return self._path

    def add(
        self,
        *,
        dataset: DatasetMetadata,
        result: DownloadResult,
    ) -> bool:
        """Add a retrieval record if this exact content is not recorded.

        Returns:
            True when a new row was written, otherwise False.
        """
        record = self._build_record(
            dataset=dataset,
            result=result,
        )

        existing = self.read()

        duplicate = any(
            row["resource_id"] == record.resource_id
            and row["sha256"] == record.sha256
            for row in existing
        )

        if duplicate:
            logger.info(
                "Manifest already contains resource checksum",
                extra={
                    "resource_id": record.resource_id,
                    "sha256": record.sha256,
                },
            )

            return False

        rows = existing + [
            {
                key: str(value)
                for key, value in asdict(record).items()
            }
        ]

        self._write_atomic(rows)

        return True

    def read(self) -> list[dict[str, str]]:
        """Read all current manifest records."""
        if not self._path.exists():
            return []

        try:
            with self._path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(file)

                if reader.fieldnames is None:
                    return []

                missing = set(_MANIFEST_FIELDS) - set(reader.fieldnames)

                if missing:
                    missing_text = ", ".join(sorted(missing))
                    raise DownloadError(
                        "Raw-data manifest is missing expected fields: "
                        f"{missing_text}"
                    )

                return [
                    {
                        key: value or ""
                        for key, value in row.items()
                        if key is not None
                    }
                    for row in reader
                ]

        except OSError as exc:
            raise DownloadError(
                f"Could not read raw-data manifest: {self._path}"
            ) from exc

    def _build_record(
        self,
        *,
        dataset: DatasetMetadata,
        result: DownloadResult,
    ) -> ManifestRecord:
        resource = result.resource

        return ManifestRecord(
            dataset_id=dataset.dataset_id,
            dataset_title=dataset.title,
            source_organisation=dataset.organisation or "",
            source_licence=dataset.licence_title or "",
            resource_id=resource.resource_id,
            resource_name=resource.name,
            resource_kind=resource.resource_kind,
            resource_format=resource.format,
            source_url=resource.download_url,
            source_hash=resource.source_hash or "",
            source_created=resource.created or "",
            source_last_modified=resource.last_modified or "",
            retrieved_at=result.retrieved_at.isoformat(),
            local_path=str(result.local_path),
            sha256=result.sha256,
            byte_count=result.byte_count,
            content_type=result.content_type or "",
        )

    def _write_atomic(
        self,
        rows: list[dict[str, str]],
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = self._path.with_suffix(
            self._path.suffix + ".tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=_MANIFEST_FIELDS,
                    extrasaction="ignore",
                )

                writer.writeheader()
                writer.writerows(rows)

            temporary_path.replace(self._path)

        except OSError as exc:
            temporary_path.unlink(missing_ok=True)

            raise DownloadError(
                f"Could not write raw-data manifest: {self._path}"
            ) from exc