"""Tests for conservative recognition of legacy category CSV layouts."""

from pathlib import Path

from qld_surgery_optimiser.ingestion.source_layout import (
    CATEGORY_HEADERLESS_COLUMNS,
    first_csv_row,
    is_headerless_category_row,
)
from qld_surgery_optimiser.validation.pipeline import _read_raw_csv

LEGACY_ROW = (
    "211,Atherton Hospital,Dec-23,1,16,100,6.7,4,1,75,"
    " 16-Jan-2024,1,0,66.7"
)


def test_recognises_documented_category_layout() -> None:
    first = first_csv_row(
        LEGACY_ROW
    )

    assert first is not None

    assert is_headerless_category_row(
        first,
        resource_kind="category",
    )

    assert not is_headerless_category_row(
        first,
        resource_kind="specialty",
    )


def test_reads_headerless_category_without_dropping_first_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "category.csv"

    path.write_text(
        LEGACY_ROW
        + "\n"
        + LEGACY_ROW.replace(
            "Dec-23,1",
            "Dec-23,2",
        )
        + "\n",
        encoding="utf-8",
    )

    data = _read_raw_csv(
        path,
        resource_kind="category",
    )

    assert len(data) == 2
    assert list(data.columns) == list(
        CATEGORY_HEADERLESS_COLUMNS
    )

    assert data.loc[
        0,
        "Facility_Code",
    ] == "211"

    assert data.loc[
        1,
        "Category",
    ] == "2"


def test_headered_csv_stays_headered(
    tmp_path: Path,
) -> None:
    path = tmp_path / "category.csv"

    path.write_text(
        "Facility_Code,Facility_Name,Report_Month\n"
        "211,Atherton Hospital,Dec-23\n",
        encoding="utf-8",
    )

    data = _read_raw_csv(
        path,
        resource_kind="category",
    )

    assert len(data) == 1

    assert list(data.columns) == [
        "Facility_Code",
        "Facility_Name",
        "Report_Month",
    ]