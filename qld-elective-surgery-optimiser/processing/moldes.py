"""Typed models for canonical processing and warehouse construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WarehouseBuildSummary:
    """Summary of one canonical warehouse build."""

    validated_files: int
    canonical_rows: int
    longitudinal_rows: int

    facility_count: int
    specialty_count: int
    urgency_category_count: int
    reporting_period_count: int

    unresolved_facilities: int
    duplicate_canonical_keys_removed: int

    database_path: Path
    canonical_parquet_path: Path
    longitudinal_parquet_path: Path
    reconciliation_report_path: Path