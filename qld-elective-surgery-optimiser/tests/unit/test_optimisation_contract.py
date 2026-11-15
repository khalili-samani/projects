
"""Tests for the optimisation input data contract."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pytest

from qld_surgery_optimiser.optimisation.contract import (
    build_optimisation_inputs,
)


def _valid_frame() -> pd.DataFrame:
    """Return a minimal valid canonical optimisation dataset."""
    return pd.DataFrame(
        {
            "facility_key": [
                "facility-1",
                "facility-1",
                "facility-2",
            ],
            "facility_name": [
                "Hospital A",
                "Hospital A",
                "Hospital B",
            ],
            "resource_kind": [
                "specialty",
                "specialty",
                "specialty",
            ],
            "report_month": [
                pd.Timestamp("2025-06-01"),
                pd.Timestamp("2025-06-01"),
                pd.Timestamp("2025-06-01"),
            ],
            "service_code": [
                "GS",
                "ENT",
                "GS",
            ],
            "service_name": [
                "General Surgery",
                "Ear Nose and Throat",
                "General Surgery",
            ],
            "vol_waiting": [
                100,
                80,
                60,
            ],
            "vol_long_waits": [
                20,
                10,
                5,
            ],
        }
    )


def _later_period_frame() -> pd.DataFrame:
    """Return one valid observation from a later reporting period."""
    return pd.DataFrame(
        {
            "facility_key": ["facility-3"],
            "facility_name": ["Hospital C"],
            "resource_kind": ["specialty"],
            "report_month": [
                pd.Timestamp("2025-09-01"),
            ],
            "service_code": ["ORTH"],
            "service_name": ["Orthopaedics"],
            "vol_waiting": [120],
            "vol_long_waits": [30],
        }
    )


def _aggregate_frame() -> pd.DataFrame:
    """Return overlapping service and statewide reporting totals."""
    return pd.DataFrame(
        {
            "facility_key": [
                "facility-1",
                "facility-2",
                "statewide",
                "statewide",
                "statewide",
            ],
            "facility_name": [
                "Hospital A",
                "Hospital B",
                "Queensland Reporting Hospitals",
                "Queensland Reporting Hospitals",
                "Queensland Reporting Hospitals",
            ],
            "resource_kind": [
                "specialty",
                "specialty",
                "specialty",
                "specialty",
                "specialty",
            ],
            "report_month": [
                pd.Timestamp("2025-06-01"),
            ] * 5,
            "service_code": [
                "ALL",
                "ALL",
                "GS",
                "ENT",
                "ALL",
            ],
            "service_name": [
                "ALL",
                "ALL",
                "General Surgery",
                "Ear Nose and Throat",
                "ALL",
            ],
            "vol_waiting": [
                180,
                60,
                160,
                80,
                240,
            ],
            "vol_long_waits": [
                30,
                5,
                25,
                10,
                35,
            ],
        }
    )


def test_builds_valid_optimisation_inputs() -> None:
    """Valid analytical rows should become typed optimisation inputs."""
    dataframe = _valid_frame()

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3

    first = rows[0]

    assert first.facility_key == "facility-1"
    assert first.facility_name == "Hospital A"
    assert first.service_key == "GS"
    assert first.service_name == "General Surgery"
    assert first.resource_kind == "specialty"
    assert first.reporting_period == date(2025, 6, 1)
    assert first.vol_waiting == 100
    assert first.vol_long_waits == 20


def test_uses_latest_reporting_period_when_not_supplied() -> None:
    """The latest analytical reporting period should be selected."""
    dataframe = pd.concat(
        [
            _valid_frame(),
            _later_period_frame(),
        ],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 1
    assert rows[0].reporting_period == date(2025, 9, 1)
    assert rows[0].facility_key == "facility-3"


def test_selects_requested_reporting_period() -> None:
    """A specific available reporting period should be selectable."""
    dataframe = pd.concat(
        [
            _valid_frame(),
            _later_period_frame(),
        ],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
        reporting_period=date(2025, 6, 1),
    )

    assert len(rows) == 3

    assert all(
        row.reporting_period == date(2025, 6, 1)
        for row in rows
    )


def test_filters_requested_resource_kind() -> None:
    """Specialty and category source families must remain separate."""
    specialty = _valid_frame()

    category = pd.DataFrame(
        {
            "facility_key": ["facility-9"],
            "facility_name": ["Hospital Z"],
            "resource_kind": ["category"],
            "report_month": [
                pd.Timestamp("2025-06-01"),
            ],
            "service_code": ["1"],
            "service_name": ["Category 1"],
            "vol_waiting": [40],
            "vol_long_waits": [3],
        }
    )

    dataframe = pd.concat(
        [specialty, category],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="category",
    )

    assert len(rows) == 1
    assert rows[0].resource_kind == "category"
    assert rows[0].facility_key == "facility-9"
    assert rows[0].service_key == "1"


def test_rejects_missing_required_columns() -> None:
    """Missing analytical fields should fail before optimisation."""
    dataframe = _valid_frame().drop(
        columns=["vol_waiting"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_missing_resource_kind_rows() -> None:
    """The requested published source family must exist."""
    dataframe = _valid_frame()

    with pytest.raises(
        ValueError,
        match="No analytical rows found for resource_kind",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="category",
        )


def test_rejects_unavailable_reporting_period() -> None:
    """A requested period must exist in the selected source family."""
    dataframe = _valid_frame()

    with pytest.raises(
        ValueError,
        match=(
            "Requested reporting period "
            "2025-09-01 is not available"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
            reporting_period=date(2025, 9, 1),
        )


def test_rejects_invalid_reporting_period_values() -> None:
    """Unparseable reporting periods should fail explicitly."""
    dataframe = _valid_frame()

    # Permit deliberately invalid data without dtype warnings.
    dataframe["report_month"] = (
        dataframe["report_month"].astype("object")
    )

    dataframe.loc[
        0,
        "report_month",
    ] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="invalid reporting period",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_duplicate_facility_service_grain() -> None:
    """Duplicate facility-service rows must not reach the solver."""
    dataframe = _valid_frame()

    duplicate = dataframe.iloc[[0]].copy()

    dataframe = pd.concat(
        [dataframe, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate facility-service rows",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_negative_waiting_volume() -> None:
    """Waiting volume must be non-negative."""
    dataframe = _valid_frame()

    dataframe.loc[0, "vol_waiting"] = -1

    with pytest.raises(
        ValueError,
        match="vol_waiting must not be negative",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_negative_long_wait_volume() -> None:
    """Long-wait volume must be non-negative."""
    dataframe = _valid_frame()

    dataframe.loc[0, "vol_long_waits"] = -1

    with pytest.raises(
        ValueError,
        match="vol_long_waits must not be negative",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_long_waits_above_waiting_volume() -> None:
    """Long waits cannot exceed total waiting volume."""
    dataframe = _valid_frame()

    dataframe.loc[0, "vol_waiting"] = 10
    dataframe.loc[0, "vol_long_waits"] = 11

    with pytest.raises(
        ValueError,
        match="vol_long_waits must not exceed vol_waiting",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_fractional_waiting_volume() -> None:
    """Patient-volume inputs must represent whole cases."""
    dataframe = _valid_frame()

    # Allow fractional values without incompatible-dtype warnings.
    dataframe["vol_waiting"] = (
        dataframe["vol_waiting"].astype("float64")
    )

    dataframe.loc[0, "vol_waiting"] = 10.5

    with pytest.raises(
        ValueError,
        match="vol_waiting must be a whole number",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_missing_service_code() -> None:
    """A service identity is mandatory at the optimisation grain."""
    dataframe = _valid_frame()

    dataframe.loc[0, "service_code"] = None

    with pytest.raises(
        ValueError,
        match="service_code must not be missing",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_empty_facility_key() -> None:
    """Facility identifiers must be non-empty."""
    dataframe = _valid_frame()

    dataframe.loc[0, "facility_key"] = "   "

    with pytest.raises(
        ValueError,
        match="facility_key must not be empty",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_excludes_aggregate_service_totals() -> None:
    """All-service totals are not independent allocation destinations."""
    aggregates = _aggregate_frame().iloc[[0, 1]]

    dataframe = pd.concat(
        [_valid_frame(), aggregates],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3

    assert all(
        row.service_key != "ALL"
        for row in rows
    )


def test_excludes_statewide_reporting_aggregates() -> None:
    """Statewide totals must not become allocation destinations."""
    aggregates = _aggregate_frame().iloc[[2, 3]]

    dataframe = pd.concat(
        [_valid_frame(), aggregates],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3

    assert all(
        row.facility_key != "statewide"
        for row in rows
    )


def test_excludes_overlapping_aggregates_without_double_counting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overlapping exclusion reasons must count each row only once."""
    dataframe = pd.concat(
        [_valid_frame(), _aggregate_frame()],
        ignore_index=True,
    )

    with caplog.at_level(
        logging.INFO,
        logger=(
            "qld_surgery_optimiser.optimisation.contract"
        ),
    ):
        rows = build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )

    assert len(rows) == 3

    assert sum(
        row.vol_waiting for row in rows
    ) == 240

    assert sum(
        row.vol_long_waits for row in rows
    ) == 35

    assert "source_rows=8" in caplog.text
    assert "aggregate_service_rows=3" in caplog.text
    assert "aggregate_facility_rows=3" in caplog.text
    assert "overlapping_aggregate_rows=1" in caplog.text
    assert "excluded_rows=5" in caplog.text
    assert "eligible_rows=3" in caplog.text


def test_aggregate_matching_is_case_and_whitespace_insensitive() -> None:
    """Publisher label formatting must not bypass aggregate exclusion."""
    aggregates = _aggregate_frame()

    aggregates.loc[0, "service_code"] = "  aLl  "
    aggregates.loc[0, "service_name"] = " all "

    aggregates.loc[2, "facility_name"] = (
        "  QUEENSLAND   Reporting Hospitals  "
    )

    dataframe = pd.concat(
        [_valid_frame(), aggregates],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3


@pytest.mark.parametrize(
    "service_name",
    [
        "All",
        "All specialties",
        "All categories",
        "All urgency categories",
    ],
)
def test_excludes_aggregate_service_names(
    service_name: str,
) -> None:
    """Known aggregate names should be excluded even with another code."""
    aggregate = _aggregate_frame().iloc[[0]].copy()

    aggregate.loc[:, "service_code"] = "TOTAL"
    aggregate.loc[:, "service_name"] = service_name

    dataframe = pd.concat(
        [_valid_frame(), aggregate],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3


def test_does_not_exclude_legitimate_names_containing_all() -> None:
    """Aggregate detection uses exact labels rather than substrings."""
    dataframe = _valid_frame()

    dataframe.loc[0, "facility_name"] = "Allora Hospital"

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3
    assert rows[0].facility_name == "Allora Hospital"


def test_excludes_category_aggregates() -> None:
    """Aggregate exclusions must apply to category scenarios."""
    dataframe = _valid_frame()

    dataframe["resource_kind"] = "category"

    dataframe["service_code"] = [
        "1",
        "2",
        "1",
    ]

    dataframe["service_name"] = [
        "Category 1",
        "Category 2",
        "Category 1",
    ]

    aggregate = dataframe.iloc[[0]].copy()

    aggregate.loc[:, "service_code"] = "ALL"
    aggregate.loc[:, "service_name"] = "All categories"

    statewide = dataframe.iloc[[0]].copy()

    statewide.loc[:, "facility_key"] = "statewide"

    statewide.loc[:, "facility_name"] = (
        "Queensland Reporting Hospitals"
    )

    combined = pd.concat(
        [dataframe, aggregate, statewide],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        combined,
        resource_kind="category",
    )

    assert len(rows) == 3

    assert all(
        row.resource_kind == "category"
        for row in rows
    )

    assert all(
        row.facility_key != "statewide"
        and row.service_key != "ALL"
        for row in rows
    )


def test_rejects_period_containing_only_aggregates() -> None:
    """An aggregate-only period cannot be solved as a detailed scenario."""
    dataframe = _aggregate_frame()

    with pytest.raises(
        ValueError,
        match="No eligible detailed optimisation rows remain",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_does_not_fall_back_to_older_detailed_period() -> None:
    """A latest aggregate-only period must fail rather than fall back."""
    detailed = _valid_frame()
    aggregates = _aggregate_frame()

    # Align datetime dtypes explicitly before concatenation.
    detailed["report_month"] = (
        detailed["report_month"].astype("datetime64[ns]")
    )

    aggregates["report_month"] = pd.Series(
        pd.Timestamp("2025-09-01"),
        index=aggregates.index,
        dtype="datetime64[ns]",
    )

    dataframe = pd.concat(
        [detailed, aggregates],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="No eligible detailed optimisation rows remain",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
        reporting_period=date(2025, 6, 1),
    )

    assert len(rows) == 3


def test_preserves_original_analytical_dataframe() -> None:
    """Candidate filtering must not modify the analytical source."""
    dataframe = pd.concat(
        [_valid_frame(), _aggregate_frame()],
        ignore_index=True,
    )

    original = dataframe.copy(deep=True)

    build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )


def test_accepts_canonical_iso_reporting_period_strings() -> None:
    """Canonical ISO date strings should be accepted explicitly."""
    dataframe = _valid_frame()

    dataframe["report_month"] = [
        "2025-06-01",
        "2025-06-01",
        "2025-06-01",
    ]

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3

    assert all(
        row.reporting_period == date(2025, 6, 1)
        for row in rows
    )


def test_normalises_reporting_period_timestamps() -> None:
    """Timestamp time components must not prevent period selection."""
    dataframe = _valid_frame()

    dataframe["report_month"] = [
        pd.Timestamp("2025-06-01 08:00:00"),
        pd.Timestamp("2025-06-01 12:00:00"),
        pd.Timestamp("2025-06-01 16:00:00"),
    ]

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3

    assert all(
        row.reporting_period == date(2025, 6, 1)
        for row in rows
    )


@pytest.mark.parametrize(
    "invalid_period",
    [
        "01/06/2025",
        "Jun-25",
        "2025-02-30",
        "2025",
        20250601,
        None,
    ],
)
def test_rejects_noncanonical_reporting_periods(
    invalid_period: str | int | None,
) -> None:
    """The optimisation boundary requires valid canonical dates."""
    dataframe = _valid_frame()

    dataframe["report_month"] = (
        dataframe["report_month"].astype("object")
    )

    dataframe.at[0, "report_month"] = invalid_period

    with pytest.raises(
        ValueError,
        match="invalid reporting period",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_duplicate_keys_with_whitespace_differences() -> None:
    """Normalisation must not introduce duplicate allocation keys."""
    dataframe = _valid_frame()

    duplicate = dataframe.iloc[[0]].copy()

    duplicate.loc[:, "facility_key"] = " facility-1 "
    duplicate.loc[:, "service_code"] = " GS "

    dataframe = pd.concat(
        [dataframe, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate facility-service rows",
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_preserves_zero_waiting_detailed_observations() -> None:
    """Detailed rows with zero demand remain valid analytical inputs."""
    dataframe = _valid_frame()

    dataframe.loc[0, "vol_waiting"] = 0
    dataframe.loc[0, "vol_long_waits"] = 0

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(rows) == 3
    assert rows[0].vol_waiting == 0
    assert rows[0].vol_long_waits == 0