"""Construction of stable longitudinal elective-surgery records."""

from __future__ import annotations

import hashlib

import pandas as pd

from qld_surgery_optimiser.exceptions import DataValidationError


BUSINESS_KEY = [
    "canonical_facility_code",
    "report_month",
    "resource_kind",
    "service_code",
    "service_name",
]


def _business_key_hash(
    row: pd.Series,
) -> str:
    """Create deterministic analytical business-key identifier."""
    values = [
        row.get(column)
        for column in BUSINESS_KEY
    ]

    payload = "|".join(
        "" if pd.isna(value) else str(value)
        for value in values
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def build_longitudinal_frame(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Create one latest canonical observation per business key.

    When multiple source versions exist for the same canonical key, the
    latest data_last_update is preferred, then the latest retrieval time.
    """
    if dataframe.empty:
        return dataframe.copy(), 0

    missing = set(BUSINESS_KEY) - set(
        dataframe.columns
    )

    if missing:
        raise DataValidationError(
            "Canonical data is missing longitudinal key fields: "
            + ", ".join(sorted(missing))
        )

    frame = dataframe.copy()

    frame["report_month"] = pd.to_datetime(
        frame["report_month"],
        errors="coerce",
    )

    frame["data_last_update"] = pd.to_datetime(
        frame["data_last_update"],
        errors="coerce",
    )

    frame["source_retrieved_at"] = pd.to_datetime(
        frame["source_retrieved_at"],
        errors="coerce",
        utc=True,
    )

    frame["business_key_id"] = frame.apply(
        _business_key_hash,
        axis=1,
    )

    before = len(frame)

    frame = frame.sort_values(
        by=[
            "business_key_id",
            "data_last_update",
            "source_retrieved_at",
            "source_sha256",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
        na_position="first",
    )

    frame = frame.drop_duplicates(
        subset=["business_key_id"],
        keep="last",
    )

    removed = before - len(frame)

    frame = frame.sort_values(
        by=[
            "report_month",
            "canonical_facility_code",
            "resource_kind",
            "service_name",
        ],
        na_position="last",
    ).reset_index(drop=True)

    group_columns = [
        "canonical_facility_code",
        "resource_kind",
        "service_code",
        "service_name",
    ]

    frame["previous_vol_waiting"] = (
        frame.groupby(
            group_columns,
            dropna=False,
        )["vol_waiting"]
        .shift(1)
    )

    frame["backlog_change"] = (
        frame["vol_waiting"]
        - frame["previous_vol_waiting"]
    )

    frame["previous_vol_long_waits"] = (
        frame.groupby(
            group_columns,
            dropna=False,
        )["vol_long_waits"]
        .shift(1)
    )

    frame["long_wait_change"] = (
        frame["vol_long_waits"]
        - frame["previous_vol_long_waits"]
    )

    frame["long_wait_share"] = (
        frame["vol_long_waits"]
        .div(
            frame["vol_waiting"].where(
                frame["vol_waiting"].ne(0)
            )
        )
    )

    frame["treatment_to_waiting_ratio"] = (
        frame["vol_treated"]
        .div(
            frame["vol_waiting"].where(
                frame["vol_waiting"].ne(0)
            )
        )
    )

    return frame, removed