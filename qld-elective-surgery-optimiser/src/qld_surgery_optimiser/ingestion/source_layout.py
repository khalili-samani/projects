"""Recognise supported Queensland elective-surgery CSV layouts.

Raw payloads are never rewritten. Layout detection determines how
validation should interpret the original publisher-supplied columns.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Literal

type ResourceKind = Literal["category", "specialty"]
type LayoutMode = Literal[
    "headered",
    "legacy_headered",
    "headerless",
]

CATEGORY_COLUMNS = (
    "Facility_Code",
    "Facility_Name",
    "Report_Month",
    "Category",
    "Vol_Treated",
    "Percent_Treated_InTime",
    "Percent_Variation_Treated_Prior_Year",
    "Vol_Waiting",
    "Vol_LongWaits",
    "Percent_Waiting_InTime_Total",
    "data_last_update",
    "Vol_LongWaits_RFS",
    "Vol_LongWaits_NRFS",
    "Percent_Waiting_InTime_RFS",
)

SPECIALTY_COLUMNS = (
    "Facility_Code",
    "Facility_Name",
    "Report_Month",
    "Specialty_Code",
    "Specialty_Desc",
    "Vol_Treated",
    "Percent_Treated_InTime",
    "Percent_Variation_Treated_Prior_Year",
    "Vol_Waiting",
    "Vol_LongWaits",
    "Percent_Waiting_InTime_Total",
    "data_last_update",
    "Vol_LongWaits_RFS",
    "Vol_LongWaits_NRFS",
    "Percent_Waiting_InTime_RFS",
)

CATEGORY_HEADERLESS_COLUMNS = CATEGORY_COLUMNS

LEGACY_CATEGORY_COLUMNS = (
    "FACILITY_CODE",
    "FACILITY_DESC",
    "QTR_MON_YEAR",
    "CATEGORY",
    "TREATED_QTR_CURRENT",
    "PERC_TREATED_IN_TIME_QTR",
    "TREATED_VARIANCE_QTR",
    "WAITING_EOQ",
    "LONG_WAIT_EOQ",
    "PERCENT_WAITING_IN_TIME_EOQ",
    "Data Date",
    "RFS_LONG_WAIT_EOQ",
    "NRFS_LONG_WAIT_EOQ",
    "PERCENT_RFS_WAITING_IN_TIME_EOQ",
)

LEGACY_CATEGORY_RENAME = dict(
    zip(
        LEGACY_CATEGORY_COLUMNS,
        CATEGORY_COLUMNS,
        strict=True,
    )
)

CATEGORY_COLUMN_COUNT = len(CATEGORY_COLUMNS)
SPECIALTY_COLUMN_COUNT = len(SPECIALTY_COLUMNS)
LEGACY_PADDED_CATEGORY_COUNT = 17

REQUIRED_IDENTITY_COLUMNS = frozenset(
    {
        "Facility_Code",
        "Facility_Name",
        "Report_Month",
    }
)

_PERIOD = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceLayout:
    """Interpretation of one recognised publisher CSV."""

    mode: LayoutMode
    resource_kind: ResourceKind
    source_width: int
    columns: tuple[str, ...]
    padding_count: int = 0


def first_csv_row(
    payload: str,
) -> list[str] | None:
    """Return the first non-empty CSV record."""
    reader = csv.reader(
        io.StringIO(payload),
        strict=True,
    )

    return next(
        (
            row
            for row in reader
            if row and any(value.strip() for value in row)
        ),
        None,
    )


def _parse_rows(
    payload: str,
) -> list[list[str]]:
    """Parse source records, retaining original field values."""
    reader = csv.reader(
        io.StringIO(payload),
        strict=True,
    )

    return [
        row
        for row in reader
        if row and any(value.strip() for value in row)
    ]


def _has_data_identity(
    row: list[str],
    *,
    resource_kind: ResourceKind,
) -> bool:
    """Identify a headerless record without inspecting metric values."""
    if len(row) < 5:
        return False

    facility_code = row[0].strip()
    facility_name = row[1].strip()
    report_month = row[2].strip()
    service_code = row[3].strip()

    if not (
        facility_code.isdecimal()
        and facility_name
        and _PERIOD.fullmatch(report_month) is not None
    ):
        return False

    if resource_kind == "category":
        return service_code.upper() in {
            "1",
            "2",
            "3",
            "ALL",
        }

    return bool(
        service_code
        and row[4].strip()
    )


def is_headerless_category_row(
    row: list[str],
    *,
    resource_kind: str,
) -> bool:
    """Recognise the known category layouts, including empty padding."""
    if resource_kind != "category":
        return False

    if len(row) not in {
        CATEGORY_COLUMN_COUNT,
        LEGACY_PADDED_CATEGORY_COUNT,
    }:
        return False

    if not _has_data_identity(
        row,
        resource_kind="category",
    ):
        return False

    if len(row) == LEGACY_PADDED_CATEGORY_COUNT:
        return all(
            not value.strip()
            for value in row[CATEGORY_COLUMN_COUNT:]
        )

    return True


def _check_record_widths(
    rows: list[list[str]],
    *,
    expected_width: int,
    start_record: int,
) -> None:
    """Reject inconsistent physical row widths."""
    for record_number, row in enumerate(
        rows,
        start=start_record,
    ):
        if len(row) != expected_width:
            raise ValueError(
                "CSV has inconsistent column count "
                f"in record {record_number}: "
                f"expected {expected_width}, "
                f"found {len(row)}."
            )


def _check_empty_padding(
    rows: list[list[str]],
    *,
    first_record: int,
) -> None:
    """Reject any populated extra field in legacy category exports."""
    for record_number, row in enumerate(
        rows,
        start=first_record,
    ):
        if any(
            value.strip()
            for value in row[CATEGORY_COLUMN_COUNT:]
        ):
            raise ValueError(
                "Legacy category CSV has non-empty trailing fields "
                f"in record {record_number}."
            )


def detect_source_layout(
    payload: str,
    *,
    resource_kind: ResourceKind,
) -> SourceLayout | None:
    """Detect a verified source layout.

    Return None for an unknown layout.

    Raise ValueError for inconsistent or populated extra fields
    in an otherwise recognisable source.
    """
    rows = _parse_rows(payload)

    if not rows:
        return None

    first = rows[0]
    stripped = tuple(
        value.strip()
        for value in first
    )

    if REQUIRED_IDENTITY_COLUMNS.issubset(
        set(stripped)
    ):
        _check_record_widths(
            rows[1:],
            expected_width=len(first),
            start_record=2,
        )

        return SourceLayout(
            mode="headered",
            resource_kind=resource_kind,
            source_width=len(first),
            columns=stripped,
        )

    if (
        resource_kind == "category"
        and stripped == LEGACY_CATEGORY_COLUMNS
    ):
        _check_record_widths(
            rows[1:],
            expected_width=CATEGORY_COLUMN_COUNT,
            start_record=2,
        )

        return SourceLayout(
            mode="legacy_headered",
            resource_kind="category",
            source_width=CATEGORY_COLUMN_COUNT,
            columns=CATEGORY_COLUMNS,
        )

    if not _has_data_identity(
        first,
        resource_kind=resource_kind,
    ):
        return None

    if resource_kind == "category":
        allowed_widths = {
            CATEGORY_COLUMN_COUNT,
            LEGACY_PADDED_CATEGORY_COUNT,
        }
        canonical_columns = CATEGORY_COLUMNS
    else:
        allowed_widths = {
            SPECIALTY_COLUMN_COUNT,
        }
        canonical_columns = SPECIALTY_COLUMNS

    if len(first) not in allowed_widths:
        return None

    _check_record_widths(
        rows,
        expected_width=len(first),
        start_record=1,
    )

    padding_count = (
        len(first) - CATEGORY_COLUMN_COUNT
        if resource_kind == "category"
        else 0
    )

    if padding_count:
        _check_empty_padding(
            rows,
            first_record=1,
        )

    return SourceLayout(
        mode="headerless",
        resource_kind=resource_kind,
        source_width=len(first),
        columns=canonical_columns,
        padding_count=padding_count,
    )


def detect_headerless_category_layout(
    payload: str,
    *,
    resource_kind: str,
) -> int | None:
    """Backward-compatible category-only layout detection."""
    if resource_kind != "category":
        return None

    layout = detect_source_layout(
        payload,
        resource_kind="category",
    )

    if layout is None or layout.mode != "headerless":
        return None

    return layout.source_width