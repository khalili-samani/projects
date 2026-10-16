"""Recognise documented legacy headerless category CSVs without changing raw bytes."""

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

_PERIOD = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}",
    re.I,
)


def is_headerless_category_row(
    row: list[str],
    *,
    resource_kind: str,
) -> bool:
    """Recognise first data rows conservatively; do not guess specialty layouts."""
    if resource_kind != "category" or len(row) != len(CATEGORY_HEADERLESS_COLUMNS):
        return False

    facility_code, facility_name, report_month, category = (
        value.strip()
        for value in row[:4]
    )

    return (
        facility_code.isdecimal()
        and bool(facility_name)
        and _PERIOD.fullmatch(report_month) is not None
        and category.upper() in {"1", "2", "3", "ALL"}
    )


def first_csv_row(
    payload: str,
) -> list[str] | None:
    """Read the first nonblank CSV record without modifying the source text."""
    reader = csv.reader(
        io.StringIO(payload)
    )

    return next(
        (
            row
            for row in reader
            if row and any(cell.strip() for cell in row)
        ),
        None,
    )