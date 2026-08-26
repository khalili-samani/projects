"""Tests for quarantine dataset construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qld_surgery_optimiser.validation.models import (
    QualityIssue,
)
from qld_surgery_optimiser.validation.quarantine import (
    build_quarantine_frame,
)


def test_quarantine_preserves_original_record() -> None:
    frame = pd.DataFrame(
        {
            "Facility_Code": [
                "101",
            ],
            "Facility_Name": [
                "Example Hospital",
            ],
            "Vol_Waiting": [
                "10",
            ],
            "Vol_LongWaits": [
                "20",
            ],
        }
    )

    issues = [
        QualityIssue(
            rule_id="LONG_WAITS_EXCEED_WAITING",
            severity="error",
            message=(
                "Vol_LongWaits exceeds Vol_Waiting."
            ),
            row_index=0,
            column="Vol_LongWaits",
            observed_value="20",
        )
    ]

    quarantine = build_quarantine_frame(
        frame,
        invalid_rows={0},
        issues=issues,
        source_path=Path(
            "example.csv"
        ),
        resource_kind="specialty",
    )

    assert len(quarantine) == 1

    assert (
        quarantine.loc[
            0,
            "Facility_Name",
        ]
        == "Example Hospital"
    )

    rule_ids = json.loads(
        quarantine.loc[
            0,
            "_quality_rule_ids",
        ]
    )

    assert rule_ids == [
        "LONG_WAITS_EXCEED_WAITING"
    ]