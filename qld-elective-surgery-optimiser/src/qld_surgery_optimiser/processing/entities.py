"""Deterministic facility entity resolution."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from qld_surgery_optimiser.exceptions import EntityResolutionError


ALIAS_COLUMNS = [
    "alias_name",
    "canonical_name",
    "canonical_code",
    "hhs",
    "region",
    "active",
]


def normalise_facility_name(
    value: object,
) -> str:
    """Create deterministic facility-name matching key."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().casefold()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def load_facility_aliases(
    path: Path,
) -> pd.DataFrame:
    """Load reviewed facility aliases.

    An empty alias file containing only headers is valid.
    """
    if not path.exists():
        raise EntityResolutionError(
            f"Facility alias file does not exist: {path}"
        )

    try:
        aliases = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise EntityResolutionError(
            f"Could not read facility aliases: {path}"
        ) from exc

    missing = set(ALIAS_COLUMNS) - set(
        aliases.columns
    )

    if missing:
        raise EntityResolutionError(
            "Facility alias file is missing columns: "
            + ", ".join(sorted(missing))
        )

    aliases = aliases[
        ALIAS_COLUMNS
    ].copy()

    aliases["_alias_key"] = aliases[
        "alias_name"
    ].map(normalise_facility_name)

    active = aliases["active"].map(
        lambda value: str(value).strip().casefold()
        not in {"false", "0", "no", "n"}
    )

    aliases = aliases.loc[
        active
    ].copy()

    duplicate_aliases = aliases.loc[
        aliases["_alias_key"].duplicated(
            keep=False
        )
        & aliases["_alias_key"].ne("")
    ]

    if not duplicate_aliases.empty:
        raise EntityResolutionError(
            "Facility alias file contains duplicate "
            "active alias names."
        )

    return aliases


def resolve_facilities(
    dataframe: pd.DataFrame,
    *,
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve facility names using explicit reviewed aliases.

    Unmapped facilities retain their observed source identity.
    """
    frame = dataframe.copy()

    frame["_facility_key"] = frame[
        "facility_name"
    ].map(normalise_facility_name)

    lookup = aliases.set_index(
        "_alias_key"
    ).to_dict(
        orient="index"
    )

    canonical_codes: list[str | None] = []
    canonical_names: list[str | None] = []
    hhs_values: list[str | None] = []
    regions: list[str | None] = []
    statuses: list[str] = []

    for row in frame.itertuples(
        index=False
    ):
        key = normalise_facility_name(
            row.facility_name
        )

        match = lookup.get(key)

        if match is None:
            canonical_codes.append(
                row.facility_code
            )
            canonical_names.append(
                row.facility_name
            )
            hhs_values.append(None)
            regions.append(None)
            statuses.append("source")
            continue

        canonical_code = (
            match.get("canonical_code")
            or row.facility_code
        )

        canonical_name = (
            match.get("canonical_name")
            or row.facility_name
        )

        canonical_codes.append(
            str(canonical_code)
            if canonical_code is not None
            else None
        )

        canonical_names.append(
            str(canonical_name)
            if canonical_name is not None
            else None
        )

        hhs = match.get("hhs")
        region = match.get("region")

        hhs_values.append(
            str(hhs) if hhs else None
        )

        regions.append(
            str(region) if region else None
        )

        statuses.append("alias")

    frame["canonical_facility_code"] = (
        canonical_codes
    )

    frame["canonical_facility_name"] = (
        canonical_names
    )

    frame["hhs"] = hhs_values
    frame["region"] = regions
    frame["facility_resolution_status"] = statuses

    return frame.drop(
        columns=["_facility_key"]
    )


def count_unresolved_facilities(
    dataframe: pd.DataFrame,
) -> int:
    """Count distinct source facilities not resolved through an alias."""
    unresolved = dataframe.loc[
        dataframe[
            "facility_resolution_status"
        ].eq("source"),
        [
            "facility_code",
            "facility_name",
        ],
    ]

    return len(
        unresolved.drop_duplicates()
    )