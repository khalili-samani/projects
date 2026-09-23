"""Tests for the optimisation input data contract."""

from __future__ import annotations

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
                pd.Timestamp(
                    "2025-06-01"
                ),
                pd.Timestamp(
                    "2025-06-01"
                ),
                pd.Timestamp(
                    "2025-06-01"
                ),
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


def test_builds_valid_optimisation_inputs() -> None:
    """Valid analytical rows should become typed optimisation inputs."""
    dataframe = _valid_frame()

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(
        rows
    ) == 3

    first = rows[
        0
    ]

    assert (
        first.facility_key
        == "facility-1"
    )

    assert (
        first.facility_name
        == "Hospital A"
    )

    assert (
        first.service_key
        == "GS"
    )

    assert (
        first.service_name
        == "General Surgery"
    )

    assert (
        first.resource_kind
        == "specialty"
    )

    assert (
        first.reporting_period
        == date(
            2025,
            6,
            1,
        )
    )

    assert (
        first.vol_waiting
        == 100
    )

    assert (
        first.vol_long_waits
        == 20
    )


def test_uses_latest_reporting_period_when_not_supplied() -> None:
    """The latest analytical reporting period should be selected."""
    dataframe = pd.concat(
        [
            _valid_frame(),
            pd.DataFrame(
                {
                    "facility_key": [
                        "facility-3",
                    ],
                    "facility_name": [
                        "Hospital C",
                    ],
                    "resource_kind": [
                        "specialty",
                    ],
                    "report_month": [
                        pd.Timestamp(
                            "2025-09-01"
                        ),
                    ],
                    "service_code": [
                        "ORTH",
                    ],
                    "service_name": [
                        "Orthopaedics",
                    ],
                    "vol_waiting": [
                        120,
                    ],
                    "vol_long_waits": [
                        30,
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
    )

    assert len(
        rows
    ) == 1

    assert (
        rows[0].reporting_period
        == date(
            2025,
            9,
            1,
        )
    )

    assert (
        rows[0].facility_key
        == "facility-3"
    )


def test_selects_requested_reporting_period() -> None:
    """A specific available reporting period should be selectable."""
    dataframe = pd.concat(
        [
            _valid_frame(),
            pd.DataFrame(
                {
                    "facility_key": [
                        "facility-3",
                    ],
                    "facility_name": [
                        "Hospital C",
                    ],
                    "resource_kind": [
                        "specialty",
                    ],
                    "report_month": [
                        pd.Timestamp(
                            "2025-09-01"
                        ),
                    ],
                    "service_code": [
                        "ORTH",
                    ],
                    "service_name": [
                        "Orthopaedics",
                    ],
                    "vol_waiting": [
                        120,
                    ],
                    "vol_long_waits": [
                        30,
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="specialty",
        reporting_period=date(
            2025,
            6,
            1,
        ),
    )

    assert len(
        rows
    ) == 3

    assert all(
        row.reporting_period
        == date(
            2025,
            6,
            1,
        )
        for row
        in rows
    )


def test_filters_requested_resource_kind() -> None:
    """Specialty and category source families must remain separate."""
    specialty = _valid_frame()

    category = pd.DataFrame(
        {
            "facility_key": [
                "facility-9",
            ],
            "facility_name": [
                "Hospital Z",
            ],
            "resource_kind": [
                "category",
            ],
            "report_month": [
                pd.Timestamp(
                    "2025-06-01"
                ),
            ],
            "service_code": [
                "1",
            ],
            "service_name": [
                "Category 1",
            ],
            "vol_waiting": [
                40,
            ],
            "vol_long_waits": [
                3,
            ],
        }
    )

    dataframe = pd.concat(
        [
            specialty,
            category,
        ],
        ignore_index=True,
    )

    rows = build_optimisation_inputs(
        dataframe,
        resource_kind="category",
    )

    assert len(
        rows
    ) == 1

    assert (
        rows[0].resource_kind
        == "category"
    )

    assert (
        rows[0].facility_key
        == "facility-9"
    )

    assert (
        rows[0].service_key
        == "1"
    )


def test_rejects_missing_required_columns() -> None:
    """Missing analytical fields should fail before optimisation."""
    dataframe = _valid_frame().drop(
        columns=[
            "vol_waiting",
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "missing required columns"
        ),
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
        match=(
            "No analytical rows found "
            "for resource_kind"
        ),
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
            reporting_period=date(
                2025,
                9,
                1,
            ),
        )


def test_rejects_invalid_reporting_period_values() -> None:
    """Unparseable reporting periods should fail explicitly."""
    dataframe = _valid_frame()

    dataframe.loc[
        0,
        "report_month",
    ] = "not-a-date"

    with pytest.raises(
        ValueError,
        match=(
            "invalid reporting period"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_duplicate_facility_service_grain() -> None:
    """Duplicate facility-service rows must not reach the solver."""
    dataframe = _valid_frame()

    duplicate = dataframe.iloc[
        [
            0,
        ]
    ].copy()

    dataframe = pd.concat(
        [
            dataframe,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate facility-service rows"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_negative_waiting_volume() -> None:
    """Waiting volume must be non-negative."""
    dataframe = _valid_frame()

    dataframe.loc[
        0,
        "vol_waiting",
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            "vol_waiting must not be negative"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_negative_long_wait_volume() -> None:
    """Long-wait volume must be non-negative."""
    dataframe = _valid_frame()

    dataframe.loc[
        0,
        "vol_long_waits",
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            "vol_long_waits must not be negative"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_long_waits_above_waiting_volume() -> None:
    """Long waits cannot exceed total waiting volume."""
    dataframe = _valid_frame()

    dataframe.loc[
        0,
        "vol_waiting",
    ] = 10

    dataframe.loc[
        0,
        "vol_long_waits",
    ] = 11

    with pytest.raises(
        ValueError,
        match=(
            "vol_long_waits must not exceed "
            "vol_waiting"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_fractional_waiting_volume() -> None:
    """Patient-volume inputs must represent whole cases."""
    dataframe = _valid_frame()

    dataframe.loc[
        0,
        "vol_waiting",
    ] = 10.5

    with pytest.raises(
        ValueError,
        match=(
            "vol_waiting must be a whole number"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_missing_service_code() -> None:
    """A service identity is mandatory at the optimisation grain."""
    dataframe = _valid_frame()

    dataframe.loc[
        0,
        "service_code",
    ] = None

    with pytest.raises(
        ValueError,
        match=(
            "service_code must not be missing"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )


def test_rejects_empty_facility_key() -> None:
    """Facility identifiers must be non-empty."""
    dataframe = _valid_frame()

    dataframe.loc[
        0,
        "facility_key",
    ] = "   "

    with pytest.raises(
        ValueError,
        match=(
            "facility_key must not be empty"
        ),
    ):
        build_optimisation_inputs(
            dataframe,
            resource_kind="specialty",
        )