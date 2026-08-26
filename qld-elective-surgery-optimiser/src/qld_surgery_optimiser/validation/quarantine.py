"""Persistence of invalid source observations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qld_surgery_optimiser.validation.models import QualityIssue


def build_quarantine_frame(
    original: pd.DataFrame,
    *,
    invalid_rows: set[int],
    issues: list[QualityIssue],
    source_path: Path,
    resource_kind: str,
) -> pd.DataFrame:
    """Create quarantine rows with machine-readable failure metadata."""
    if not invalid_rows:
        return pd.DataFrame()

    issue_map: dict[int, list[QualityIssue]] = {}

    for issue in issues:
        if issue.row_index is None:
            continue

        issue_map.setdefault(
            issue.row_index,
            [],
        ).append(issue)

    records: list[dict[str, object]] = []

    for index in sorted(invalid_rows):
        source_record = original.loc[index].to_dict()

        row_issues = issue_map.get(
            index,
            [],
        )

        source_record["_source_row_index"] = index
        source_record["_source_path"] = str(source_path)
        source_record["_resource_kind"] = resource_kind

        source_record["_quality_rule_ids"] = json.dumps(
            sorted(
                {
                    issue.rule_id
                    for issue in row_issues
                }
            )
        )

        source_record["_quality_messages"] = json.dumps(
            [
                issue.message
                for issue in row_issues
            ]
        )

        records.append(source_record)

    return pd.DataFrame(records)


def write_quarantine(
    dataframe: pd.DataFrame,
    *,
    source_path: Path,
    quarantine_directory: Path,
) -> Path | None:
    """Write quarantined observations as Parquet."""
    if dataframe.empty:
        return None

    quarantine_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        quarantine_directory
        / f"{source_path.stem}_quarantine.parquet"
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    return output_path