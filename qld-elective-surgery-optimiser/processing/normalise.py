"""Conversion from validated source schemas to the canonical data model."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from qld_surgery_optimiser.exceptions import DataValidationError


CANONICAL_COLUMNS = [
    "record_id",
    "resource_kind",
    "source_resource_id",
    "source_sha256",
    "source_url",
    "source_retrieved_at",
    "source_file",
    "facility_code",
    "facility_name",
    "report_month",
    "service_code",
    "service_name",
    "vol_treated",
    "pct_treated_in_time",
    "pct_variation_treated_prior_year",
    "vol_waiting",
    "vol_long_waits",
    "pct_waiting_in_time_total",
    "data_last_update",
    "vol_long_waits_rfs",
    "vol_long_waits_nrfs",
    "pct_waiting_in_time_rfs",
]


COLUMN_ALIASES = {
    # Modern names
    "Facility_Code": "facility_code",
    "Facility_Name": "facility_name",
    "Report_Month": "report_month",
    "Specialty_Code": "service_code",
    "Specialty_Desc": "service_name",
    "Category": "service_name",
    "Vol_Treated": "vol_treated",
    "Percent_Treated_InTime": "pct_treated_in_time",
    "Percent_Variation_Treated_Prior_Year": (
        "pct_variation_treated_prior_year"
    ),
    "Vol_Waiting": "vol_waiting",
    "Vol_LongWaits": "vol_long_waits",
    "Percent_Waiting_InTime_Total": (
        "pct_waiting_in_time_total"
    ),
    "data_last_update": "data_last_update",
    "Vol_LongWaits_RFS": "vol_long_waits_rfs",
    "Vol_LongWaits_NRFS": "vol_long_waits_nrfs",
    "Percent_Waiting_InTime_RFS": (
        "pct_waiting_in_time_rfs"
    ),

    # Legacy published names
    "FACILITY_CODE": "facility_code",
    "FACILITY_DESC": "facility_name",
    "QTR_MON_YEAR": "report_month",
    "SPECIALTY_CODE": "service_code",
    "SPECIALTY_DESC": "service_name",
    "CATEGORY": "service_name",
    "TREATED_QTR_CURRENT": "vol_treated",
    "PERC_TREATED_IN_TIME_QTR": (
        "pct_treated_in_time"
    ),
    "TREATED_VARIANCE_QTR": (
        "pct_variation_treated_prior_year"
    ),
    "WAITING_EOQ": "vol_waiting",
    "LONG_WAIT_EOQ": "vol_long_waits",
    "PERCENT_WAITING_IN_TIME_EOQ": (
        "pct_waiting_in_time_total"
    ),
    "Data Date": "data_last_update",
    "RFS_LONG_WAIT_EOQ": "vol_long_waits_rfs",
    "NRFS_LONG_WAIT_EOQ": "vol_long_waits_nrfs",
    "PERCENT_RFS_WAITING_IN_TIME_EOQ": (
        "pct_waiting_in_time_rfs"
    ),
}


NUMERIC_COLUMNS = [
    "vol_treated",
    "pct_treated_in_time",
    "pct_variation_treated_prior_year",
    "vol_waiting",
    "vol_long_waits",
    "vol_long_waits_rfs",
    "vol_long_waits_nrfs",
    "pct_waiting_in_time_total",
    "pct_waiting_in_time_rfs",
]


def _clean_text(value: object) -> str | None:
    """Normalise whitespace while preserving meaningful text."""
    if value is None or pd.isna(value):
        return None

    text = re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )

    return text or None


def _normalise_code(value: object) -> str | None:
    """Convert identifiers to stable strings without decimal suffixes."""
    text = _clean_text(value)

    if text is None:
        return None

    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    return text


def _normalise_report_month(
    value: object,
) -> pd.Timestamp | pd.NaT:
    """Normalise reporting periods to month-level timestamps."""
    if value is None or pd.isna(value):
        return pd.NaT

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return pd.NaT

    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=1,
    )


def _make_record_id(
    row: pd.Series,
) -> str:
    """Create deterministic canonical row identifier."""
    components = [
        str(row.get("resource_kind", "")),
        str(row.get("source_resource_id", "")),
        str(row.get("facility_code", "")),
        str(row.get("report_month", "")),
        str(row.get("service_code", "")),
        str(row.get("service_name", "")),
    ]

    payload = "|".join(components)

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _rename_source_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Map recognised publisher columns to canonical names."""
    rename_map = {
        column: COLUMN_ALIASES[column]
        for column in dataframe.columns
        if column in COLUMN_ALIASES
    }

    renamed = dataframe.rename(
        columns=rename_map
    )

    duplicated = renamed.columns[
        renamed.columns.duplicated()
    ].tolist()

    if duplicated:
        raise DataValidationError(
            "Source normalisation produced duplicate canonical "
            f"columns: {sorted(set(duplicated))}"
        )

    return renamed


def normalise_validated_frame(
    dataframe: pd.DataFrame,
    *,
    resource_kind: str,
    source_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Convert one validated source frame into canonical records."""
    if resource_kind not in {
        "category",
        "specialty",
    }:
        raise DataValidationError(
            f"Unsupported resource kind: {resource_kind}"
        )

    frame = _rename_source_columns(
        dataframe.copy()
    )

    required = {
        "facility_code",
        "facility_name",
        "report_month",
        "vol_treated",
        "vol_waiting",
        "vol_long_waits",
        "service_name",
    }

    missing = required - set(frame.columns)

    if missing:
        raise DataValidationError(
            "Validated input cannot be normalised because "
            "canonical fields are missing: "
            + ", ".join(sorted(missing))
        )

    if "service_code" not in frame.columns:
        frame["service_code"] = pd.NA

    frame["resource_kind"] = resource_kind

    frame["facility_code"] = frame[
        "facility_code"
    ].map(_normalise_code)

    frame["facility_name"] = frame[
        "facility_name"
    ].map(_clean_text)

    frame["service_code"] = frame[
        "service_code"
    ].map(_normalise_code)

    frame["service_name"] = frame[
        "service_name"
    ].map(_clean_text)

    frame["report_month"] = frame[
        "report_month"
    ].map(_normalise_report_month)

    if "data_last_update" in frame.columns:
        frame["data_last_update"] = pd.to_datetime(
            frame["data_last_update"],
            errors="coerce",
        )
    else:
        frame["data_last_update"] = pd.NaT

    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["source_resource_id"] = (
        source_metadata.get("resource_id")
    )

    frame["source_sha256"] = (
        source_metadata.get("sha256")
    )

    frame["source_url"] = (
        source_metadata.get("source_url")
    )

    frame["source_retrieved_at"] = (
        pd.to_datetime(
            source_metadata.get("retrieved_at"),
            errors="coerce",
            utc=True,
        )
    )

    frame["source_file"] = (
        source_metadata.get("source_file")
    )

    frame["record_id"] = frame.apply(
        _make_record_id,
        axis=1,
    )

    return frame.reindex(
        columns=CANONICAL_COLUMNS
    )


def infer_resource_kind(
    path: Path,
) -> str:
    """Infer resource family from a validated Parquet location."""
    parts = {
        part.casefold()
        for part in path.parts
    }

    if "category" in parts:
        return "category"

    if "specialty" in parts:
        return "specialty"

    raise DataValidationError(
        f"Could not determine resource kind: {path}"
    )