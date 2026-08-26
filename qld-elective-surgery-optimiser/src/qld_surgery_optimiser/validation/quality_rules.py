"""Explicit row-level and dataset-level data-quality rules."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from qld_surgery_optimiser.config import ValidationConfig
from qld_surgery_optimiser.validation.models import QualityIssue


def detect_parse_failures(
    original: pd.DataFrame,
    failure_masks: dict[str, pd.Series],
) -> tuple[list[QualityIssue], set[int]]:
    """Convert coercion failures into quality issues."""
    issues: list[QualityIssue] = []
    invalid_rows: set[int] = set()

    for column, mask in failure_masks.items():
        failing_indices = original.index[mask]

        for index in failing_indices:
            invalid_rows.add(int(index))

            value = original.at[index, column]

            issues.append(
                QualityIssue(
                    rule_id="PARSE_FAILURE",
                    severity="error",
                    message=(
                        f"Value in {column} could not be parsed "
                        "to its expected type."
                    ),
                    row_index=int(index),
                    column=column,
                    observed_value=str(value),
                )
            )

    return issues, invalid_rows


def check_required_values(
    dataframe: pd.DataFrame,
    *,
    columns: Iterable[str],
) -> tuple[list[QualityIssue], set[int]]:
    """Reject rows missing essential business-key fields."""
    issues: list[QualityIssue] = []
    invalid_rows: set[int] = set()

    for column in columns:
        if column not in dataframe.columns:
            continue

        mask = dataframe[column].isna()

        for index in dataframe.index[mask]:
            invalid_rows.add(int(index))

            issues.append(
                QualityIssue(
                    rule_id="MISSING_REQUIRED_VALUE",
                    severity="error",
                    message=f"Required value missing in {column}.",
                    row_index=int(index),
                    column=column,
                )
            )

    return issues, invalid_rows


def check_non_negative_volumes(
    dataframe: pd.DataFrame,
    *,
    volume_columns: Iterable[str],
) -> tuple[list[QualityIssue], set[int]]:
    """Reject negative patient-volume observations."""
    issues: list[QualityIssue] = []
    invalid_rows: set[int] = set()

    for column in volume_columns:
        if column not in dataframe.columns:
            continue

        mask = dataframe[column].notna() & (
            dataframe[column] < 0
        )

        for index in dataframe.index[mask]:
            invalid_rows.add(int(index))

            issues.append(
                QualityIssue(
                    rule_id="NEGATIVE_VOLUME",
                    severity="error",
                    message=(
                        f"{column} must not be negative."
                    ),
                    row_index=int(index),
                    column=column,
                    observed_value=str(
                        dataframe.at[index, column]
                    ),
                )
            )

    return issues, invalid_rows


def check_percentage_ranges(
    dataframe: pd.DataFrame,
    *,
    percentage_columns: Iterable[str],
    minimum: float,
    maximum: float,
) -> tuple[list[QualityIssue], set[int]]:
    """Reject percentages outside the configured bounds."""
    issues: list[QualityIssue] = []
    invalid_rows: set[int] = set()

    for column in percentage_columns:
        if column not in dataframe.columns:
            continue

        values = dataframe[column]

        mask = values.notna() & (
            (values < minimum)
            | (values > maximum)
        )

        for index in dataframe.index[mask]:
            invalid_rows.add(int(index))

            issues.append(
                QualityIssue(
                    rule_id="PERCENTAGE_OUT_OF_RANGE",
                    severity="error",
                    message=(
                        f"{column} must be between "
                        f"{minimum} and {maximum}."
                    ),
                    row_index=int(index),
                    column=column,
                    observed_value=str(
                        dataframe.at[index, column]
                    ),
                )
            )

    return issues, invalid_rows


def check_long_wait_consistency(
    dataframe: pd.DataFrame,
) -> tuple[list[QualityIssue], set[int]]:
    """Ensure long waits do not exceed total waiting volume."""
    required = {
        "Vol_Waiting",
        "Vol_LongWaits",
    }

    if not required.issubset(dataframe.columns):
        return [], set()

    mask = (
        dataframe["Vol_Waiting"].notna()
        & dataframe["Vol_LongWaits"].notna()
        & (
            dataframe["Vol_LongWaits"]
            > dataframe["Vol_Waiting"]
        )
    )

    issues: list[QualityIssue] = []
    invalid_rows: set[int] = set()

    for index in dataframe.index[mask]:
        invalid_rows.add(int(index))

        issues.append(
            QualityIssue(
                rule_id="LONG_WAITS_EXCEED_WAITING",
                severity="error",
                message=(
                    "Vol_LongWaits exceeds Vol_Waiting."
                ),
                row_index=int(index),
                column="Vol_LongWaits",
                observed_value=str(
                    dataframe.at[
                        index,
                        "Vol_LongWaits",
                    ]
                ),
            )
        )

    return issues, invalid_rows


def check_long_wait_components(
    dataframe: pd.DataFrame,
) -> tuple[list[QualityIssue], set[int]]:
    """Check RFS and NRFS long-wait components against total long waits."""
    columns = {
        "Vol_LongWaits",
        "Vol_LongWaits_RFS",
        "Vol_LongWaits_NRFS",
    }

    if not columns.issubset(dataframe.columns):
        return [], set()

    total = dataframe["Vol_LongWaits"]
    components = (
        dataframe["Vol_LongWaits_RFS"].fillna(0)
        + dataframe["Vol_LongWaits_NRFS"].fillna(0)
    )

    mask = (
        total.notna()
        & dataframe["Vol_LongWaits_RFS"].notna()
        & dataframe["Vol_LongWaits_NRFS"].notna()
        & (components != total)
    )

    issues: list[QualityIssue] = []

    for index in dataframe.index[mask]:
        issues.append(
            QualityIssue(
                rule_id="LONG_WAIT_COMPONENT_MISMATCH",
                severity="warning",
                message=(
                    "RFS and NRFS long-wait components "
                    "do not equal Vol_LongWaits."
                ),
                row_index=int(index),
                column="Vol_LongWaits",
                observed_value=str(
                    dataframe.at[
                        index,
                        "Vol_LongWaits",
                    ]
                ),
            )
        )

    return issues, set()


def check_duplicates(
    dataframe: pd.DataFrame,
    *,
    resource_kind: str,
) -> tuple[list[QualityIssue], set[int]]:
    """Detect duplicate source business keys."""
    if resource_kind == "specialty":
        candidates = [
            "Facility_Code",
            "Report_Month",
            "Specialty_Code",
        ]
    elif resource_kind == "category":
        candidates = [
            "Facility_Code",
            "Report_Month",
            "Category",
        ]
    else:
        raise ValueError(
            f"Unsupported resource kind: {resource_kind}"
        )

    key = [
        column
        for column in candidates
        if column in dataframe.columns
    ]

    if len(key) != len(candidates):
        return [], set()

    mask = dataframe.duplicated(
        subset=key,
        keep=False,
    )

    issues: list[QualityIssue] = []
    invalid_rows: set[int] = set()

    for index in dataframe.index[mask]:
        invalid_rows.add(int(index))

        issues.append(
            QualityIssue(
                rule_id="DUPLICATE_BUSINESS_KEY",
                severity="error",
                message=(
                    "Duplicate row detected for business key: "
                    + ", ".join(key)
                ),
                row_index=int(index),
            )
        )

    return issues, invalid_rows


def run_quality_rules(
    dataframe: pd.DataFrame,
    *,
    original: pd.DataFrame,
    failure_masks: dict[str, pd.Series],
    resource_kind: str,
    config: ValidationConfig,
) -> tuple[list[QualityIssue], set[int]]:
    """Execute all current row-level quality rules."""
    all_issues: list[QualityIssue] = []
    invalid_rows: set[int] = set()

    checks = [
        detect_parse_failures(
            original,
            failure_masks,
        ),
        check_required_values(
            dataframe,
            columns=[
                "Facility_Code",
                "Facility_Name",
                "Report_Month",
            ],
        ),
        check_non_negative_volumes(
            dataframe,
            volume_columns=config.numeric_volume_columns,
        ),
        check_percentage_ranges(
            dataframe,
            percentage_columns=config.percentage_columns,
            minimum=config.percentage_minimum,
            maximum=config.percentage_maximum,
        ),
        check_long_wait_consistency(
            dataframe,
        ),
        check_long_wait_components(
            dataframe,
        ),
        check_duplicates(
            dataframe,
            resource_kind=resource_kind,
        ),
    ]

    for issues, rows in checks:
        all_issues.extend(issues)
        invalid_rows.update(rows)

    return all_issues, invalid_rows