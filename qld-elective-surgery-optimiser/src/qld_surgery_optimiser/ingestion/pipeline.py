"""High-level raw-data ingestion workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from qld_surgery_optimiser.config import (
    AppSettings,
    BaseConfig,
)
from qld_surgery_optimiser.ingestion.ckan_client import CkanClient
from qld_surgery_optimiser.ingestion.downloader import ResourceDownloader
from qld_surgery_optimiser.ingestion.manifest import RawManifest
from qld_surgery_optimiser.ingestion.models import ResourceRef

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionSummary:
    """Summary of one raw-data ingestion run."""

    dataset_id: str
    resources_discovered: int
    resources_selected: int
    resources_downloaded: int
    resources_already_present: int
    manifest_records_added: int
    manifest_path: Path


def run_ingestion(
    *,
    settings: AppSettings,
    base_config: BaseConfig,
    latest_only: bool = False,
) -> IngestionSummary:
    """Discover and retrieve elective-surgery resources."""
    source_config = base_config.sources.queensland_open_data

    with CkanClient(
        source_config,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.request_max_retries,
        retry_backoff_seconds=settings.request_retry_backoff_seconds,
        user_agent=settings.user_agent,
    ) as ckan:
        dataset, resources = ckan.get_dataset()

    selected = (
        _select_latest_by_kind(resources)
        if latest_only
        else resources
    )

    manifest = RawManifest(
        settings.raw_data_dir / "manifest.csv"
    )

    downloaded_count = 0
    existing_count = 0
    manifest_count = 0

    with ResourceDownloader(
        raw_data_dir=settings.raw_data_dir,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.request_max_retries,
        retry_backoff_seconds=settings.request_retry_backoff_seconds,
        user_agent=settings.user_agent,
    ) as downloader:
        for resource in selected:
            result = downloader.download(resource)

            if result.downloaded:
                downloaded_count += 1
            else:
                existing_count += 1

            if manifest.add(
                dataset=dataset,
                result=result,
            ):
                manifest_count += 1

    summary = IngestionSummary(
        dataset_id=dataset.dataset_id,
        resources_discovered=len(resources),
        resources_selected=len(selected),
        resources_downloaded=downloaded_count,
        resources_already_present=existing_count,
        manifest_records_added=manifest_count,
        manifest_path=manifest.path,
    )

    logger.info(
        "Ingestion run completed",
        extra={
            "dataset_id": summary.dataset_id,
            "resources_discovered": summary.resources_discovered,
            "resources_selected": summary.resources_selected,
            "resources_downloaded": summary.resources_downloaded,
            "resources_already_present": (
                summary.resources_already_present
            ),
            "manifest_records_added": summary.manifest_records_added,
            "manifest_path": str(summary.manifest_path),
        },
    )

    return summary


def _select_latest_by_kind(
    resources: list[ResourceRef],
) -> list[ResourceRef]:
    """Select one most-recent resource for each resource family.

    Source metadata timestamps are preferred. Resource names are used as
    deterministic tie-breakers when timestamps are absent or equal.
    """
    selected: list[ResourceRef] = []

    for kind in ("category", "specialty"):
        matching = [
            resource
            for resource in resources
            if resource.resource_kind == kind
        ]

        if not matching:
            continue

        latest = max(
            matching,
            key=lambda resource: (
                resource.last_modified or resource.created or "",
                resource.name,
                resource.resource_id,
            ),
        )

        selected.append(latest)

    return selected