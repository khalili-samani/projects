"""Typed models for source-data validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class QualityIssue:
    """One detected data-quality issue."""

    rule_id: str
    severity: IssueSeverity
    message: str
    row_index: int | None = None
    column: str | None = None
    observed_value: str | None = None


@dataclass
class FileValidationResult:
    """Validation outcome for one raw source file."""

    source_path: Path
    resource_kind: str

    original_rows: int
    valid_rows: int
    quarantined_rows: int

    dataframe: pd.DataFrame
    quarantine: pd.DataFrame

    issues: list[QualityIssue]

    missing_columns: list[str]
    unexpected_columns: list[str]

    @property
    def passed(self) -> bool:
        """Return whether the source passed file-level validation."""
        return not self.missing_columns


@dataclass(frozen=True)
class ValidationRunSummary:
    """Summary of a complete raw-data validation run."""

    files_processed: int
    files_passed: int
    files_failed: int

    rows_read: int
    rows_valid: int
    rows_quarantined: int

    reports_written: int

    interim_directory: Path
    quarantine_directory: Path
    report_path: Path