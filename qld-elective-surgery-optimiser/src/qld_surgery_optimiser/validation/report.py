"""Validation-report creation and serialisation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from qld_surgery_optimiser.validation.models import (
    FileValidationResult,
)


def build_validation_report(
    results: list[FileValidationResult],
) -> dict[str, object]:
    """Build a machine-readable validation summary."""
    rule_counts: Counter[str] = Counter()

    for result in results:
        rule_counts.update(
            issue.rule_id
            for issue in result.issues
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "files_processed": len(results),
        "files_passed": sum(
            result.passed
            for result in results
        ),
        "files_failed": sum(
            not result.passed
            for result in results
        ),
        "rows_read": sum(
            result.original_rows
            for result in results
        ),
        "rows_valid": sum(
            result.valid_rows
            for result in results
        ),
        "rows_quarantined": sum(
            result.quarantined_rows
            for result in results
        ),
        "rule_counts": dict(
            sorted(rule_counts.items())
        ),
        "files": [
            {
                "source_path": str(result.source_path),
                "resource_kind": result.resource_kind,
                "passed": result.passed,
                "original_rows": result.original_rows,
                "valid_rows": result.valid_rows,
                "quarantined_rows": (
                    result.quarantined_rows
                ),
                "missing_columns": (
                    result.missing_columns
                ),
                "unexpected_columns": (
                    result.unexpected_columns
                ),
                "issues": [
                    asdict(issue)
                    for issue in result.issues
                ],
            }
            for result in results
        ],
    }


def write_validation_report(
    report: dict[str, object],
    *,
    path: Path,
) -> None:
    """Write validation results atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)