"""Typed models used by the ingestion layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


ResourceKind = Literal["category", "specialty"]


class DatasetMetadata(BaseModel):
    """Metadata describing the upstream CKAN dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    dataset_name: str
    title: str
    organisation: str | None = None
    licence_title: str | None = None
    licence_url: str | None = None
    metadata_created: str | None = None
    metadata_modified: str | None = None


class ResourceRef(BaseModel):
    """A downloadable elective-surgery CKAN resource."""

    model_config = ConfigDict(frozen=True)

    resource_id: str
    package_id: str
    name: str
    resource_kind: ResourceKind
    format: str
    download_url: str

    mimetype: str | None = None
    source_hash: str | None = None
    created: str | None = None
    last_modified: str | None = None
    metadata_modified: str | None = None


class DownloadResult(BaseModel):
    """Outcome of retrieving one raw source resource."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    resource: ResourceRef
    local_path: Path
    sha256: str
    byte_count: int
    retrieved_at: datetime
    content_type: str | None
    downloaded: bool