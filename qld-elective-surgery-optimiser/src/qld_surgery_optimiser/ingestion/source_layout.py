"""Recognise supported legacy elective-surgery CSV source layouts."""

from __future__ import annotations

import csv
import io
import re

CATEGORY_HEADERLESS_COLUMNS = (
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

CATEGORY_DATA_COLUMN_COUNT = len(CATEGORY_HEADERLESS_COLUMNS)
LEGACY_CATEGORY_PADDED_COLUMN_COUNT = 17

_PERIOD = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}",
    re.I,
)


def is_headerless_category_row(
    row: list[str],
    *,
    resource_kind: str,
) -> bool:
    """Recognise a category data row from its first four fields.

    The 17-column legacy variant is recognised only when its three
    trailing fields are empty.
    """
    if resource_kind != "category":
        return False

    if len(row) not in {
        CATEGORY_DATA_COLUMN_COUNT,
        LEGACY_CATEGORY_PADDED_COLUMN_COUNT,
    }:
        return False

    facility_code, facility_name, report_month, category = (
        value.strip()
        for value in row[:4]
    )

    identity_matches = (
        facility_code.isdecimal()
        and bool(facility_name)
        and _PERIOD.fullmatch(report_month) is not None
        and category.upper() in {"1", "2", "3", "ALL"}
    )

    if not identity_matches:
        return False

    if len(row) == LEGACY_CATEGORY_PADDED_COLUMN_COUNT:
        return all(
            not value.strip()
            for value in row[CATEGORY_DATA_COLUMN_COUNT:]
        )

    return True


def first_csv_row(
    payload: str,
) -> list[str] | None:
    """Return the first nonblank CSV record."""
    reader = csv.reader(
        io.StringIO(payload),
        strict=True,
    )

    return next(
        (
            row
            for row in reader
            if row and any(cell.strip() for cell in row)
        ),
        None,
    )


def detect_headerless_category_layout(
    payload: str,
    *,
    resource_kind: str,
) -> int | None:
    """Return the width of a supported headerless category layout.

    All records must have the same width. In the 17-column variant,
    columns 15-17 must be empty in every record.

    Returns None for sources that do not resemble a supported
    headerless category layout.

    Raises ValueError if a recognised legacy category layout contains
    inconsistent row widths or populated trailing padding.
    """
    if resource_kind != "category":
        return None

    reader = csv.reader(
        io.StringIO(payload),
        strict=True,
    )

    first = next(
        (
            row
            for row in reader
            if row and any(cell.strip() for cell in row)
        ),
        None,
    )

    if first is None:
        return None

    if len(first) not in {
        CATEGORY_DATA_COLUMN_COUNT,
        LEGACY_CATEGORY_PADDED_COLUMN_COUNT,
    }:
        return None

    if not is_headerless_category_row(
        first[:CATEGORY_DATA_COLUMN_COUNT],
        resource_kind="category",
    ):
        return None

    width = len(first)

    if (
        width == LEGACY_CATEGORY_PADDED_COLUMN_COUNT
        and any(
            value.strip()
            for value in first[CATEGORY_DATA_COLUMN_COUNT:]
        )
    ):
        raise ValueError(
            "Legacy category CSV has non-empty trailing fields "
            "in record 1."
        )

    for record_number, row in enumerate(
        reader,
        start=2,
    ):
        if len(row) != width:
            raise ValueError(
                "Legacy category CSV has inconsistent column count "
                f"in record {record_number}: "
                f"expected {width}, found {len(row)}."
            )

        if (
            width == LEGACY_CATEGORY_PADDED_COLUMN_COUNT
            and any(
                value.strip()
                for value in row[CATEGORY_DATA_COLUMN_COUNT:]
            )
        ):
            raise ValueError(
                "Legacy category CSV has non-empty trailing fields "
                f"in record {record_number}."
            )

    return width