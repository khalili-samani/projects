"""Tests for supported Queensland elective-surgery CSV layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from qld_surgery_optimiser.exceptions import DataValidationError
from qld_surgery_optimiser.ingestion.source_layout import (
    CATEGORY_COLUMNS,
    LEGACY_CATEGORY_COLUMNS,
    SPECIALTY_COLUMNS,
    detect_source_layout,
    first_csv_row,
    is_headerless_category_row,
)
from qld_surgery_optimiser.validation.pipeline import _read_raw_csv


CATEGORY_ROW = (
    "211,Atherton Hospital,Dec-23,1,16,100,6.7,4,1,75,"
    " 16-Jan-2024,1,0,66.7"
)

PADDED_CATEGORY_ROW = CATEGORY_ROW + ",,,"

SPECIALTY_ROW = (
    "211,Atherton Hospital,Dec-23,1,cardio-thoracic surgery,"
    "0,-,0,0,0,-,16-Jan-2024,0,0,-"
)


def test_recognises_headerless_category_row() -> None:
    first = first_csv_row(CATEGORY_ROW)

    assert first is not None

    assert is_headerless_category_row(
        first,
        resource_kind="category",
    )

    assert not is_headerless_category_row(
        first,
        resource_kind="specialty",
    )


def test_detects_modern_headered_category() -> None:
    text = (
        ",".join(CATEGORY_COLUMNS)
        + "\n"
        + CATEGORY_ROW
        + "\n"
    )

    layout = detect_source_layout(
        text,
        resource_kind="category",
    )

    assert layout is not None
    assert layout.mode == "headered"
    assert layout.source_width == 14


def test_detects_legacy_headered_category() -> None:
    text = (
        ",".join(LEGACY_CATEGORY_COLUMNS)
        + "\n"
        + CATEGORY_ROW
        + "\n"
    )

    layout = detect_source_layout(
        text,
        resource_kind="category",
    )

    assert layout is not None
    assert layout.mode == "legacy_headered"
    assert layout.columns == CATEGORY_COLUMNS


def test_detects_modern_headered_specialty() -> None:
    text = (
        ",".join(SPECIALTY_COLUMNS)
        + "\n"
        + SPECIALTY_ROW
        + "\n"
    )

    layout = detect_source_layout(
        text,
        resource_kind="specialty",
    )

    assert layout is not None
    assert layout.mode == "headered"
    assert layout.source_width == 15


def test_detects_headerless_specialty() -> None:
    layout = detect_source_layout(
        SPECIALTY_ROW + "\n",
        resource_kind="specialty",
    )

    assert layout is not None
    assert layout.mode == "headerless"
    assert layout.columns == SPECIALTY_COLUMNS


def test_detects_headerless_category() -> None:
    layout = detect_source_layout(
        CATEGORY_ROW + "\n",
        resource_kind="category",
    )

    assert layout is not None
    assert layout.mode == "headerless"
    assert layout.source_width == 14


def test_detects_padded_category() -> None:
    payload = (
        PADDED_CATEGORY_ROW + "\n"
    ) * 216

    layout = detect_source_layout(
        payload,
        resource_kind="category",
    )

    assert layout is not None
    assert layout.mode == "headerless"
    assert layout.source_width == 17
    assert layout.padding_count == 3


def test_rejects_populated_padding_in_first_record() -> None:
    text = CATEGORY_ROW + ",unexpected,,\n"

    with pytest.raises(
        ValueError,
        match="non-empty trailing fields",
    ):
        detect_source_layout(
            text,
            resource_kind="category",
        )


def test_rejects_populated_padding_in_later_record() -> None:
    text = (
        PADDED_CATEGORY_ROW
        + "\n"
        + CATEGORY_ROW
        + ",unexpected,,\n"
    )

    with pytest.raises(
        ValueError,
        match="non-empty trailing fields",
    ):
        detect_source_layout(
            text,
            resource_kind="category",
        )


def test_rejects_inconsistent_column_counts() -> None:
    text = (
        PADDED_CATEGORY_ROW
        + "\n"
        + CATEGORY_ROW
        + "\n"
    )

    with pytest.raises(
        ValueError,
        match="inconsistent column count",
    ):
        detect_source_layout(
            text,
            resource_kind="category",
        )


def test_rejects_unknown_header() -> None:
    layout = detect_source_layout(
        "name,value\none,1\n",
        resource_kind="category",
    )

    assert layout is None


def test_reads_legacy_category_headers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy_category.csv"

    path.write_text(
        ",".join(LEGACY_CATEGORY_COLUMNS)
        + "\n"
        + CATEGORY_ROW
        + "\n",
        encoding="utf-8",
    )

    data = _read_raw_csv(
        path,
        resource_kind="category",
    )

    assert len(data) == 1
    assert list(data.columns) == list(CATEGORY_COLUMNS)

    assert data.loc[0, "Facility_Code"] == "211"
    assert data.loc[0, "Report_Month"] == "Dec-23"


def test_reads_headerless_category_without_losing_first_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "category.csv"

    path.write_text(
        CATEGORY_ROW
        + "\n"
        + CATEGORY_ROW.replace(
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
    assert list(data.columns) == list(CATEGORY_COLUMNS)

    assert data.loc[0, "Facility_Code"] == "211"
    assert data.loc[1, "Category"] == "2"


def test_reads_padded_category_without_losing_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "june_2022.csv"

    path.write_text(
        (
            PADDED_CATEGORY_ROW
            + "\n"
        ) * 216,
        encoding="utf-8",
    )

    data = _read_raw_csv(
        path,
        resource_kind="category",
    )

    assert len(data) == 216
    assert list(data.columns) == list(CATEGORY_COLUMNS)

    assert data.loc[0, "Facility_Code"] == "211"
    assert data.loc[215, "Category"] == "1"


def test_reads_headerless_specialty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "specialty.csv"

    path.write_text(
        SPECIALTY_ROW + "\n",
        encoding="utf-8",
    )

    data = _read_raw_csv(
        path,
        resource_kind="specialty",
    )

    assert len(data) == 1
    assert list(data.columns) == list(SPECIALTY_COLUMNS)

    assert data.loc[0, "Facility_Code"] == "211"
    assert data.loc[0, "Specialty_Code"] == "1"
    assert (
        data.loc[0, "Specialty_Desc"]
        == "cardio-thoracic surgery"
    )


def test_rejects_invalid_padding_during_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.csv"

    path.write_text(
        PADDED_CATEGORY_ROW
        + "\n"
        + CATEGORY_ROW
        + ",important_value,,\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DataValidationError,
        match="Could not read raw CSV",
    ):
        _read_raw_csv(
            path,
            resource_kind="category",
        )


def test_preserves_modern_headered_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "modern.csv"

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