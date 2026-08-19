"""Client for deterministic Queensland Open Data resource discovery."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx

from qld_surgery_optimiser.config import QueenslandOpenDataConfig
from qld_surgery_optimiser.exceptions import SourceDiscoveryError
from qld_surgery_optimiser.ingestion.models import (
    DatasetMetadata,
    ResourceKind,
    ResourceRef,
)

logger = logging.getLogger(__name__)


class CkanClient:
    """Small CKAN API client specialised for the elective-surgery dataset."""

    def __init__(
        self,
        config: QueenslandOpenDataConfig,
        *,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        user_agent: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": user_agent,
            },
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> CkanClient:
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

    def _request_json(
        self,
        action: str,
        params: Mapping[str, str],
    ) -> dict[str, Any]:
        """Call a CKAN action and return its result mapping."""
        url = f"{self._config.api_base_url.rstrip('/')}/{action}"

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()

                payload = response.json()

                if not isinstance(payload, dict):
                    raise SourceDiscoveryError(
                        "CKAN response root is not a JSON object."
                    )

                if payload.get("success") is not True:
                    error = payload.get("error")
                    raise SourceDiscoveryError(
                        f"CKAN action {action!r} reported failure: {error}"
                    )

                result = payload.get("result")

                if not isinstance(result, dict):
                    raise SourceDiscoveryError(
                        f"CKAN action {action!r} did not return a result mapping."
                    )

                return result

            except (
                httpx.HTTPError,
                ValueError,
                SourceDiscoveryError,
            ) as exc:
                last_error = exc

                if attempt >= self._max_retries:
                    break

                delay = self._retry_backoff_seconds * (2**attempt)

                logger.warning(
                    "CKAN request failed; retrying",
                    extra={
                        "action": action,
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )

                time.sleep(delay)

        raise SourceDiscoveryError(
            f"CKAN request failed after "
            f"{self._max_retries + 1} attempt(s): {action}"
        ) from last_error

    def get_dataset(
        self,
    ) -> tuple[DatasetMetadata, list[ResourceRef]]:
        """Retrieve dataset metadata and eligible resources."""
        result = self._request_json(
            "package_show",
            {"id": self._config.dataset_id},
        )

        metadata = self._parse_dataset_metadata(result)
        resources = self._parse_resources(result)

        if not resources:
            raise SourceDiscoveryError(
                "No eligible elective-surgery CSV resources were found."
            )

        logger.info(
            "Elective-surgery resources discovered",
            extra={
                "dataset_id": metadata.dataset_id,
                "resource_count": len(resources),
                "category_count": sum(
                    resource.resource_kind == "category"
                    for resource in resources
                ),
                "specialty_count": sum(
                    resource.resource_kind == "specialty"
                    for resource in resources
                ),
            },
        )

        return metadata, resources

    def _parse_dataset_metadata(
        self,
        result: Mapping[str, Any],
    ) -> DatasetMetadata:
        organisation_data = result.get("organization")

        organisation: str | None = None

        if isinstance(organisation_data, dict):
            candidate = organisation_data.get("title")
            if isinstance(candidate, str):
                organisation = candidate

        return DatasetMetadata(
            dataset_id=str(result.get("id", self._config.dataset_id)),
            dataset_name=str(result.get("name", self._config.dataset_id)),
            title=str(result.get("title", self._config.dataset_id)),
            organisation=organisation,
            licence_title=_optional_string(result.get("license_title")),
            licence_url=_optional_string(result.get("license_url")),
            metadata_created=_optional_string(
                result.get("metadata_created")
            ),
            metadata_modified=_optional_string(
                result.get("metadata_modified")
            ),
        )

    def _parse_resources(
        self,
        result: Mapping[str, Any],
    ) -> list[ResourceRef]:
        raw_resources = result.get("resources")

        if not isinstance(raw_resources, list):
            raise SourceDiscoveryError(
                "CKAN dataset contains no resource list."
            )

        eligible: list[ResourceRef] = []

        for raw_resource in raw_resources:
            if not isinstance(raw_resource, dict):
                continue

            resource = self._parse_resource(raw_resource)

            if resource is not None:
                eligible.append(resource)

        eligible.sort(
            key=lambda item: (
                item.resource_kind,
                item.name.casefold(),
                item.resource_id,
            )
        )

        return eligible

    def _parse_resource(
        self,
        raw: Mapping[str, Any],
    ) -> ResourceRef | None:
        resource_format = str(raw.get("format", "")).strip()

        allowed_formats = {
            value.casefold()
            for value in self._config.allowed_formats
        }

        if resource_format.casefold() not in allowed_formats:
            return None

        name = str(raw.get("name", "")).strip()
        url = str(raw.get("url", "")).strip()
        resource_id = str(raw.get("id", "")).strip()

        if not name or not url or not resource_id:
            return None

        resource_kind = self._classify_resource(name, url)

        if resource_kind is None:
            return None

        return ResourceRef(
            resource_id=resource_id,
            package_id=str(
                raw.get("package_id", self._config.dataset_id)
            ),
            name=name,
            resource_kind=resource_kind,
            format=resource_format,
            download_url=url,
            mimetype=_optional_string(raw.get("mimetype")),
            source_hash=_optional_string(raw.get("hash")),
            created=_optional_string(raw.get("created")),
            last_modified=_optional_string(raw.get("last_modified")),
            metadata_modified=_optional_string(
                raw.get("metadata_modified")
            ),
        )

    def _classify_resource(
        self,
        name: str,
        url: str,
    ) -> ResourceKind | None:
        searchable = f"{name} {url}".casefold()

        if any(
            pattern.casefold() in searchable
            for pattern in self._config.category_patterns
        ):
            return "category"

        if any(
            pattern.casefold() in searchable
            for pattern in self._config.specialty_patterns
        ):
            return "specialty"

        return None


def _optional_string(value: object) -> str | None:
    """Convert a CKAN metadata value to an optional string."""
    if value is None:
        return None

    text = str(value).strip()
    return text or None