"""Tests for canonical source normalisation."""

from __future__ import annotations

import pandas as pd

from qld_surgery_optimiser.processing.normalise import (
    normalise_validated_frame,
)


def test_specialty_source_maps_to_canonical_schema() -> None:
    frame = pd.DataFrame(
        {
            "Facility_Code": ["101.0"],
            "Facility_Name": [
                "  Example   Hospital "
            ],
            "Report_Month": [
                pd.Timestamp("2025-09-30")
            ],
            "Specialty_Code": ["01"],
            "Specialty_Desc": [
                "General Surgery"
            ],
            "Vol_Treated": [20],
            "Vol_Waiting": [100],
            "Vol_LongWaits": [10],
        }
    )

    canonical = normalise_validated_frame(
        frame,
        resource_kind="specialty",
        source_metadata={
            "resource_id": "abc",
            "sha256": "a" * 64,
            "source_url": "https://example.test/data.csv",
            "retrieved_at": "2026-08-30T01:00:00Z",
            "source_file": "source.csv",
        },
    )

    assert len(canonical) == 1

    assert (
        canonical.loc[
            0,
            "facility_code",
        ]
        == "101"
    )

    assert (
        canonical.loc[
            0,
            "facility_name",
        ]
        == "Example Hospital"
    )

    assert (
        canonical.loc[
            0,
            "service_name",
        ]
        == "General Surgery"
    )

    assert (
        canonical.loc[
            0,
            "report_month",
        ]
        == pd.Timestamp("2025-09-01")
    )


def test_category_source_uses_category_as_service_name() -> None:
    frame = pd.DataFrame(
        {
            "Facility_Code": ["101"],
            "Facility_Name": [
                "Example Hospital"
            ],
            "Report_Month": [
                "2025-09-01"
            ],
            "Category": ["Category 1"],
            "Vol_Treated": [10],
            "Vol_Waiting": [30],
            "Vol_LongWaits": [2],
        }
    )

    canonical = normalise_validated_frame(
        frame,
        resource_kind="category",
        source_metadata={
            "resource_id": "xyz",
            "sha256": "b" * 64,
            "source_url": "https://example.test/category.csv",
            "retrieved_at": "2026-08-30T01:00:00Z",
            "source_file": "category.csv",
        },
    )

    assert (
        canonical.loc[
            0,
            "service_name",
        ]
        == "Category 1"
    )

    assert pd.isna(
        canonical.loc[
            0,
            "service_code",
        ]
    )