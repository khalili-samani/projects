"""Tests for row-level healthcare data-quality rules."""

from __future__ import annotations

import pandas as pd

from qld_surgery_optimiser.validation.quality_rules import (
    check_duplicates,
    check_long_wait_consistency,
    check_non_negative_volumes,
    check_percentage_ranges,
)


def test_long_waits_cannot_exceed_waiting_volume() -> None:
    frame = pd.DataFrame(
        {
            "Vol_Waiting": [100, 10],
            "Vol_LongWaits": [20, 15],
        }
    )

    issues, rows = check_long_wait_consistency(
        frame
    )

    assert rows == {1}
    assert len(issues) == 1

    assert (
        issues[0].rule_id
        == "LONG_WAITS_EXCEED_WAITING"
    )


def test_negative_volumes_are_invalid() -> None:
    frame = pd.DataFrame(
        {
            "Vol_Treated": [
                10,
                -1,
            ]
        }
    )

    issues, rows = check_non_negative_volumes(
        frame,
        volume_columns=["Vol_Treated"],
    )

    assert rows == {1}
    assert issues[0].rule_id == "NEGATIVE_VOLUME"


def test_percentages_outside_zero_to_one_hundred_fail() -> None:
    frame = pd.DataFrame(
        {
            "Percent_Treated_InTime": [
                95.0,
                105.0,
                -2.0,
            ]
        }
    )

    issues, rows = check_percentage_ranges(
        frame,
        percentage_columns=[
            "Percent_Treated_InTime"
        ],
        minimum=0.0,
        maximum=100.0,
    )

    assert rows == {1, 2}
    assert len(issues) == 2


def test_specialty_duplicate_business_key_is_detected() -> None:
    frame = pd.DataFrame(
        {
            "Facility_Code": [
                "101",
                "101",
            ],
            "Report_Month": [
                "2025-06",
                "2025-06",
            ],
            "Specialty_Code": [
                "GS",
                "GS",
            ],
        }
    )

    issues, rows = check_duplicates(
        frame,
        resource_kind="specialty",
    )

    assert rows == {0, 1}
    assert len(issues) == 2

    assert {
        issue.rule_id
        for issue in issues
    } == {
        "DUPLICATE_BUSINESS_KEY"
    }