"""Source discovery, retrieval and raw-data lineage management."""

from qld_surgery_optimiser.ingestion.models import (
    DatasetMetadata,
    DownloadResult,
    ResourceRef,
)
from qld_surgery_optimiser.ingestion.pipeline import (
    IngestionSummary,
    run_ingestion,
)

__all__ = [
    "DatasetMetadata",
    "DownloadResult",
    "IngestionSummary",
    "ResourceRef",
    "run_ingestion",
]