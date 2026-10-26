"""Tests for conservative recognition of legacy category CSV layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from qld_surgery_optimiser.ingestion.source_layout import (
    CATEGORY_HEADERLESS_COLUMNS,
    detect_headerless_category_layout,
    first_csv_row,
    is_headerless_category_row,
)
from qld_surgery_optimiser.validation.pipeline import _read_raw_csv

LEGACY_ROW = (
    "211,Atherton Hospital,Dec-23,1,16,100,6.7,4,1,75,"
    " 16-Jan-2024,1,0,66.7"
)

PADDED_ROW = LEGACY_ROW + ",,,"


def test_recognises_documented_category_layout() -> None:
    """Recognise the 14-column category row but not specialty rows."""
    first = first_csv_row(LEGACY_ROW)

    assert first is not None

    assert is_headerless_category_row(
        first,
        resource_kind="category",
    )

    assert not is_headerless_category_row(
        first,
        resource_kind="specialty",
    )


def test_detects_14_column_headerless_category() -> None:
    """Accept a consistent documented 14-column source."""
    layout_width = detect_headerless_category_layout(
        LEGACY_ROW + "\n",
        resource_kind="category",
    )

    assert layout_width == 14


def test_detects_17_column_category_with_empty_padding() -> None:
    """Accept the June 2022 format only when padding is empty."""
    payload = "\n".join(
        PADDED_ROW
        for _ in range(216)
    ) + "\n"

    layout_width = detect_headerless_category_layout(
        payload,
        resource_kind="category",
    )

    assert layout_width == 17


def test_rejects_populated_padding_in_later_record() -> None:
    """A later populated extra field must not be silently discarded."""
    payload = (
        PADDED_ROW
        + "\n"
        + LEGACY_ROW
        + ",unexpected,,\n"
    )

    with pytest.raises(
        ValueError,
        match="non-empty trailing fields",
    ):
        detect_headerless_category_layout(
            payload,
            resource_kind="category",
        )


def test_rejects_populated_padding_in_first_record() -> None:
    """The first record's padding must also be checked."""
    payload = LEGACY_ROW + ",unexpected,,\n"

    with pytest.raises(
        ValueError,
        match="non-empty trailing fields",
    ):
        detect_headerless_category_layout(
            payload,
            resource_kind="category",
        )


def test_rejects_inconsistent_record_width() -> None:
    """Every record must conform to one verified layout width."""
    payload = PADDED_ROW + "\n" + LEGACY_ROW + "\n"

    with pytest.raises(
        ValueError,
        match="inconsistent column count",
    ):
        detect_headerless_category_layout(
            payload,
            resource_kind="category",
        )


def test_reads_headerless_category_without_dropping_first_row(
    tmp_path: Path,
) -> None:
    """Preserve every observation in a 14-column category source."""
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

    assert data.loc[0, "Facility_Code"] == "211"
    assert data.loc[1, "Category"] == "2"


def test_reads_216_padded_records_without_losing_data(
    tmp_path: Path,
) -> None:
    """Drop verified empty padding while retaining all 216 records."""
    path = tmp_path / "june_2022.csv"

    path.write_text(
        "\n".join(
            PADDED_ROW
            for _ in range(216)
        )
        + "\n",
        encoding="utf-8",
    )

    data = _read_raw_csv(
        path,
        resource_kind="category",
    )

    assert len(data) == 216

    assert list(data.columns) == list(
        CATEGORY_HEADERLESS_COLUMNS
    )

    assert data.loc[0, "Facility_Code"] == "211"
    assert data.loc[0, "Report_Month"] == "Dec-23"
    assert data.loc[215, "Category"] == "1"


def test_validation_rejects_populated_padding(
    tmp_path: Path,
) -> None:
    """Validation must not silently discard populated extra fields."""
    path = tmp_path / "june_2022.csv"

    path.write_text(
        PADDED_ROW
        + "\n"
        + LEGACY_ROW
        + ",important_value,,\n",
        encoding="utf-8",
    )

    with pytest.raises(
        Exception,
        match="Could not read raw CSV",
    ):
        _read_raw_csv(
            path,
            resource_kind="category",
        )


def test_headered_csv_stays_headered(
    tmp_path: Path,
) -> None:
    """An ordinary headered CSV must retain its declared columns."""
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


def test_specialty_does_not_infer_category_layout() -> None:
    """Category-specific inference must not be applied to specialty."""
    assert (
        detect_headerless_category_layout(
            PADDED_ROW,
            resource_kind="specialty",
        )
        is None
    )