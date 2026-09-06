"""Tests for deterministic facility entity resolution."""

from __future__ import annotations

import pandas as pd

from qld_surgery_optimiser.processing.entities import (
    resolve_facilities,
)


def test_explicit_alias_replaces_facility_identity() -> None:
    frame = pd.DataFrame(
        {
            "facility_code": ["101"],
            "facility_name": [
                "Old Hospital Name"
            ],
        }
    )

    aliases = pd.DataFrame(
        {
            "alias_name": [
                "Old Hospital Name"
            ],
            "canonical_name": [
                "Canonical Hospital"
            ],
            "canonical_code": [
                "999"
            ],
            "hhs": [
                "Example HHS"
            ],
            "region": [
                "Metro"
            ],
            "active": [
                "true"
            ],
            "_alias_key": [
                "old hospital name"
            ],
        }
    )

    resolved = resolve_facilities(
        frame,
        aliases=aliases,
    )

    assert (
        resolved.loc[
            0,
            "canonical_facility_code",
        ]
        == "999"
    )

    assert (
        resolved.loc[
            0,
            "canonical_facility_name",
        ]
        == "Canonical Hospital"
    )

    assert (
        resolved.loc[
            0,
            "facility_resolution_status",
        ]
        == "alias"
    )


def test_unmapped_facility_preserves_source_identity() -> None:
    frame = pd.DataFrame(
        {
            "facility_code": ["101"],
            "facility_name": [
                "Example Hospital"
            ],
        }
    )

    aliases = pd.DataFrame(
        columns=[
            "alias_name",
            "canonical_name",
            "canonical_code",
            "hhs",
            "region",
            "active",
            "_alias_key",
        ]
    )

    resolved = resolve_facilities(
        frame,
        aliases=aliases,
    )

    assert (
        resolved.loc[
            0,
            "canonical_facility_code",
        ]
        == "101"
    )

    assert (
        resolved.loc[
            0,
            "canonical_facility_name",
        ]
        == "Example Hospital"
    )

    assert (
        resolved.loc[
            0,
            "facility_resolution_status",
        ]
        == "source"
    )