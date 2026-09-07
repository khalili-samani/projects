"""Tests for longitudinal record construction."""

from __future__ import annotations

import pandas as pd

from qld_surgery_optimiser.processing.longitudinal import (
    build_longitudinal_frame,
)


def test_backlog_change_uses_previous_reporting_period() -> None:
    frame = pd.DataFrame(
        {
            "record_id": [
                "one",
                "two",
            ],
            "canonical_facility_code": [
                "101",
                "101",
            ],
            "report_month": [
                "2025-06-01",
                "2025-09-01",
            ],
            "resource_kind": [
                "specialty",
                "specialty",
            ],
            "service_code": [
                "GS",
                "GS",
            ],
            "service_name": [
                "General Surgery",
                "General Surgery",
            ],
            "vol_waiting": [
                100,
                120,
            ],
            "vol_long_waits": [
                10,
                15,
            ],
            "vol_treated": [
                50,
                55,
            ],
            "data_last_update": [
                "2025-07-01",
                "2025-10-01",
            ],
            "source_retrieved_at": [
                "2025-07-02T00:00:00Z",
                "2025-10-02T00:00:00Z",
            ],
            "source_sha256": [
                "a",
                "b",
            ],
        }
    )

    result, removed = (
        build_longitudinal_frame(
            frame
        )
    )

    assert removed == 0

    second = result.iloc[1]

    assert second[
        "previous_vol_waiting"
    ] == 100

    assert second[
        "backlog_change"
    ] == 20

    assert second[
        "long_wait_change"
    ] == 5


def test_latest_source_version_wins_duplicate_business_key() -> None:
    frame = pd.DataFrame(
        {
            "record_id": [
                "old",
                "new",
            ],
            "canonical_facility_code": [
                "101",
                "101",
            ],
            "report_month": [
                "2025-09-01",
                "2025-09-01",
            ],
            "resource_kind": [
                "specialty",
                "specialty",
            ],
            "service_code": [
                "GS",
                "GS",
            ],
            "service_name": [
                "General Surgery",
                "General Surgery",
            ],
            "vol_waiting": [
                100,
                105,
            ],
            "vol_long_waits": [
                10,
                11,
            ],
            "vol_treated": [
                50,
                51,
            ],
            "data_last_update": [
                "2025-10-01",
                "2025-10-02",
            ],
            "source_retrieved_at": [
                "2025-10-01T00:00:00Z",
                "2025-10-02T00:00:00Z",
            ],
            "source_sha256": [
                "oldhash",
                "newhash",
            ],
        }
    )

    result, removed = (
        build_longitudinal_frame(
            frame
        )
    )

    assert removed == 1
    assert len(result) == 1
    assert result.iloc[0]["record_id"] == "new"