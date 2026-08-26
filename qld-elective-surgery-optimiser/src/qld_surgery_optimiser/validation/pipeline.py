"""Raw elective-surgery validation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from qld_surgery_optimiser.config import (
    AppSettings,
    BaseConfig,
)
from qld_surgery_optimiser.exceptions import DataValidationError
from qld_surgery_optimiser.validation.coercion import (
    coerce_known_columns,
)
from qld_surgery_optimiser.validation.models import (
    FileValidationResult,
    QualityIssue,
    ValidationRunSummary,
)
from qld_surgery_optimiser.validation.quality_rules import (
    run_quality_rules,
)
from qld_surgery_optimiser.validation.quarantine import (
    build_quarantine_frame,
    write_quarantine,
)
from qld_surgery_optimiser.validation.report import (
    build_validation_report,
    write_validation_report,
)
from qld_surgery_optimiser.validation.schemas import (
    missing_required_columns,
    unexpected_columns,
)

logger = logging.getLogger(__name__)


def _resource_kind_from_path(
    path: Path,
) -> str:
    """Infer source family from the immutable raw directory layout."""
    parts = {
        part.casefold()
        for part in path.parts
    }

    if "category" in parts:
        return "category"

    if "specialty" in parts:
        return "specialty"

    raise DataValidationError(
        f"Could not determine resource kind from path: {path}"
    )


def _read_raw_csv(
    path: Path,
) -> pd.DataFrame:
    """Read raw source conservatively as strings."""
    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        raise DataValidationError(
            f"Could not read raw CSV: {path}"
        ) from exc


def validate_file(
    path: Path,
    *,
    settings: AppSettings,
    base_config: BaseConfig,
) -> FileValidationResult:
    """Validate one immutable raw source file."""
    resource_kind = _resource_kind_from_path(
        path
    )

    original = _read_raw_csv(path)

    config = base_config.validation

    missing = missing_required_columns(
        original.columns,
        kind=resource_kind,
        config=config,
    )

    unexpected = unexpected_columns(
        original.columns,
        config=config,
    )

    issues: list[QualityIssue] = []

    for column in missing:
        issues.append(
            QualityIssue(
                rule_id="MISSING_REQUIRED_COLUMN",
                severity="error",
                message=(
                    f"Required source column missing: "
                    f"{column}"
                ),
                column=column,
            )
        )

    for column in unexpected:
        issues.append(
            QualityIssue(
                rule_id="UNEXPECTED_COLUMN",
                severity="warning",
                message=(
                    f"Unexpected source column detected: "
                    f"{column}"
                ),
                column=column,
            )
        )

    if (
        missing
        and config.fail_on_missing_required_columns
    ):
        return FileValidationResult(
            source_path=path,
            resource_kind=resource_kind,
            original_rows=len(original),
            valid_rows=0,
            quarantined_rows=0,
            dataframe=pd.DataFrame(),
            quarantine=pd.DataFrame(),
            issues=issues,
            missing_columns=missing,
            unexpected_columns=unexpected,
        )

    coerced, failures = coerce_known_columns(
        original,
        config=config,
    )

    row_issues, invalid_rows = run_quality_rules(
        coerced,
        original=original,
        failure_masks=failures,
        resource_kind=resource_kind,
        config=config,
    )

    issues.extend(row_issues)

    valid = coerced.loc[
        ~coerced.index.isin(invalid_rows)
    ].copy()

    quarantine = build_quarantine_frame(
        original,
        invalid_rows=invalid_rows,
        issues=issues,
        source_path=path,
        resource_kind=resource_kind,
    )

    interim_directory = (
        settings.interim_data_dir
        / resource_kind
    )

    interim_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_path = (
        interim_directory
        / f"{path.stem}_validated.parquet"
    )

    valid.to_parquet(
        valid_path,
        index=False,
    )

    if config.quarantine_invalid_rows:
        write_quarantine(
            quarantine,
            source_path=path,
            quarantine_directory=(
                settings.quarantine_data_dir
                / resource_kind
            ),
        )

    return FileValidationResult(
        source_path=path,
        resource_kind=resource_kind,
        original_rows=len(original),
        valid_rows=len(valid),
        quarantined_rows=len(quarantine),
        dataframe=valid,
        quarantine=quarantine,
        issues=issues,
        missing_columns=missing,
        unexpected_columns=unexpected,
    )


def _raw_csv_files(
    raw_data_dir: Path,
) -> list[Path]:
    """Return immutable raw source CSVs, excluding the manifest."""
    candidates = [
        path
        for path in raw_data_dir.rglob("*.csv")
        if path.name != "manifest.csv"
    ]

    return sorted(candidates)


def run_validation(
    *,
    settings: AppSettings,
    base_config: BaseConfig,
) -> ValidationRunSummary:
    """Validate all currently retrieved raw elective-surgery files."""
    files = _raw_csv_files(
        settings.raw_data_dir
    )

    if not files:
        raise DataValidationError(
            "No raw CSV resources were found. "
            "Run ingestion before validation."
        )

    results: list[FileValidationResult] = []

    for path in files:
        logger.info(
            "Validating raw source",
            extra={
                "source_path": str(path),
            },
        )

        result = validate_file(
            path,
            settings=settings,
            base_config=base_config,
        )

        results.append(result)

    report = build_validation_report(
        results
    )

    report_path = (
        settings.reports_dir
        / "outputs"
        / "data_quality_summary.json"
    )

    write_validation_report(
        report,
        path=report_path,
    )

    summary = ValidationRunSummary(
        files_processed=len(results),
        files_passed=sum(
            result.passed
            for result in results
        ),
        files_failed=sum(
            not result.passed
            for result in results
        ),
        rows_read=sum(
            result.original_rows
            for result in results
        ),
        rows_valid=sum(
            result.valid_rows
            for result in results
        ),
        rows_quarantined=sum(
            result.quarantined_rows
            for result in results
        ),
        reports_written=1,
        interim_directory=settings.interim_data_dir,
        quarantine_directory=(
            settings.quarantine_data_dir
        ),
        report_path=report_path,
    )

    logger.info(
        "Validation run completed",
        extra={
            "files_processed": (
                summary.files_processed
            ),
            "files_passed": summary.files_passed,
            "files_failed": summary.files_failed,
            "rows_read": summary.rows_read,
            "rows_valid": summary.rows_valid,
            "rows_quarantined": (
                summary.rows_quarantined
            ),
            "report_path": str(
                summary.report_path
            ),
        },
    )

    return summary