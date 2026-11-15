
"""Build and validate optimisation-ready inputs from analytical data."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, cast

import pandas as pd

from qld_surgery_optimiser.optimisation.models import (
    OptimisationInputRow,
    ResourceKind,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset(
    {
        "facility_key",
        "facility_name",
        "resource_kind",
        "report_month",
        "service_code",
        "service_name",
        "vol_waiting",
        "vol_long_waits",
    }
)

AGGREGATE_SERVICE_CODES = frozenset(
    {
        "all",
    }
)

AGGREGATE_SERVICE_NAMES = frozenset(
    {
        "all",
        "all specialties",
        "all categories",
        "all urgency categories",
    }
)

AGGREGATE_FACILITY_NAMES = frozenset(
    {
        "queensland reporting hospitals",
    }
)


def build_optimisation_inputs(
    dataframe: pd.DataFrame,
    *,
    resource_kind: ResourceKind,
    reporting_period: date | pd.Timestamp | None = None,
) -> list[OptimisationInputRow]:
    """Convert canonical analytical data into optimisation input rows.

    The optimisation layer operates on one published source family at
    a time. Specialty and category data are not merged into a synthetic
    joint distribution.

    Aggregate reporting observations are retained in the analytical
    datasets but excluded from operational allocation candidates.

    Parameters
    ----------
    dataframe:
        Canonical or longitudinal analytical dataset.
    resource_kind:
        Published source family to optimise. Supported values are
        ``"specialty"`` and ``"category"``.
    reporting_period:
        Optional reporting period. When omitted, the latest available
        period for the selected resource family is used.

    Returns
    -------
    list[OptimisationInputRow]
        Validated, non-aggregate optimisation input rows.

    Raises
    ------
    ValueError
        If required fields are missing, optimisation values are invalid,
        duplicate optimisation keys exist, or no eligible detailed
        observations remain for the selected family and period.
    """
    _validate_required_columns(dataframe)

    filtered = _filter_resource_kind(
        dataframe,
        resource_kind=resource_kind,
    )

    if filtered.empty:
        raise ValueError(
            "No analytical rows found for resource_kind "
            f"{resource_kind!r}."
        )

    filtered = _normalise_reporting_period(filtered)

    selected_period = _select_reporting_period(
        filtered,
        reporting_period=reporting_period,
    )

    filtered = filtered.loc[
        filtered["report_month"] == selected_period
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No analytical rows found for reporting period "
            f"{selected_period.date().isoformat()}."
        )

    filtered = _exclude_aggregate_observations(filtered)

    if filtered.empty:
        raise ValueError(
            "No eligible detailed optimisation rows remain "
            f"for resource_kind {resource_kind!r} and reporting period "
            f"{selected_period.date().isoformat()} after excluding "
            "aggregate reporting observations."
        )

    _validate_unique_grain(filtered)

    rows: list[OptimisationInputRow] = []

    for row in filtered.itertuples(index=False):
        facility_key = _required_text(
            row.facility_key,
            field_name="facility_key",
        )

        facility_name = _required_text(
            row.facility_name,
            field_name="facility_name",
        )

        service_code = _required_text(
            row.service_code,
            field_name="service_code",
        )

        service_name = _required_text(
            row.service_name,
            field_name="service_name",
        )

        vol_waiting = _required_non_negative_integer(
            row.vol_waiting,
            field_name="vol_waiting",
        )

        vol_long_waits = _required_non_negative_integer(
            row.vol_long_waits,
            field_name="vol_long_waits",
        )

        if vol_long_waits > vol_waiting:
            raise ValueError(
                "vol_long_waits must not exceed "
                "vol_waiting for optimisation input "
                f"{facility_key!r} / {service_code!r}."
            )

        rows.append(
            OptimisationInputRow(
                facility_key=facility_key,
                facility_name=facility_name,
                service_key=service_code,
                service_name=service_name,
                resource_kind=resource_kind,
                reporting_period=selected_period.date(),
                vol_waiting=vol_waiting,
                vol_long_waits=vol_long_waits,
            )
        )

    return rows


def _validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Ensure the analytical dataframe has the required contract."""
    missing = REQUIRED_COLUMNS - set(dataframe.columns)

    if not missing:
        return

    missing_text = ", ".join(sorted(missing))

    raise ValueError(
        "Optimisation input dataframe is missing required "
        f"columns: {missing_text}."
    )


def _filter_resource_kind(
    dataframe: pd.DataFrame,
    *,
    resource_kind: ResourceKind,
) -> pd.DataFrame:
    """Return rows belonging to one published source family."""
    values = (
        dataframe["resource_kind"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    return dataframe.loc[
        values == resource_kind
    ].copy()


def _normalise_reporting_period(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Parse canonical ISO reporting periods without format inference.

    Historical publisher-specific date formats are handled upstream
    by the source validation layer. The optimisation contract expects
    canonical ISO date values or datetime representations.
    """
    result = dataframe.copy()

    values = (
        result["report_month"]
        .astype("string")
        .str.strip()
    )

    # Restrict the contract to dates and timezone-naive ISO timestamps.
    # Reject numeric values and ambiguous publisher-specific labels.
    valid_format = values.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}"
        r"(?:[ T]\d{2}:\d{2}:\d{2}"
        r"(?:\.\d{1,9})?)?",
        na=False,
    )

    parsed = pd.to_datetime(
        values.where(valid_format),
        format="ISO8601",
        errors="coerce",
    )

    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())

        raise ValueError(
            "Optimisation input contains "
            f"{invalid_count} invalid reporting period value(s)."
        )

    result["report_month"] = parsed.dt.normalize()

    return result


def _select_reporting_period(
    dataframe: pd.DataFrame,
    *,
    reporting_period: date | pd.Timestamp | None,
) -> pd.Timestamp:
    """Resolve the reporting period used for optimisation."""
    if reporting_period is None:
        maximum = dataframe["report_month"].max()

        if pd.isna(maximum):
            raise ValueError(
                "Unable to determine the latest reporting period."
            )

        return pd.Timestamp(maximum).normalize()

    requested = pd.Timestamp(
        reporting_period
    ).normalize()

    available = {
        pd.Timestamp(value).normalize()
        for value in dataframe["report_month"].dropna()
    }

    if requested not in available:
        available_text = ", ".join(
            sorted(
                value.date().isoformat()
                for value in available
            )
        )

        raise ValueError(
            "Requested reporting period "
            f"{requested.date().isoformat()} is not available. "
            f"Available periods: {available_text}."
        )

    return requested


def _normalise_label(
    series: pd.Series,
) -> pd.Series:
    """Normalise labels for exact aggregate classification."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def _exclude_aggregate_observations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Exclude known overlapping aggregate reporting observations.

    Service totals and statewide reporting aggregates are useful
    for analytical reporting but are not independent allocation
    destinations.

    The input dataframe is not modified.
    """
    service_codes = _normalise_label(
        dataframe["service_code"]
    )

    service_names = _normalise_label(
        dataframe["service_name"]
    )

    facility_names = _normalise_label(
        dataframe["facility_name"]
    )

    aggregate_service_mask = (
        service_codes.isin(AGGREGATE_SERVICE_CODES)
        | service_names.isin(AGGREGATE_SERVICE_NAMES)
    )

    aggregate_facility_mask = facility_names.isin(
        AGGREGATE_FACILITY_NAMES
    )

    aggregate_mask = (
        aggregate_service_mask
        | aggregate_facility_mask
    )

    excluded_service_count = int(
        aggregate_service_mask.sum()
    )

    excluded_facility_count = int(
        aggregate_facility_mask.sum()
    )

    overlap_count = int(
        (
            aggregate_service_mask
            & aggregate_facility_mask
        ).sum()
    )

    excluded_total = int(aggregate_mask.sum())

    eligible = dataframe.loc[
        ~aggregate_mask
    ].copy()

    logger.info(
        "Optimisation candidate selection: "
        "source_rows=%s, "
        "aggregate_service_rows=%s, "
        "aggregate_facility_rows=%s, "
        "overlapping_aggregate_rows=%s, "
        "excluded_rows=%s, "
        "eligible_rows=%s",
        len(dataframe),
        excluded_service_count,
        excluded_facility_count,
        overlap_count,
        excluded_total,
        len(eligible),
    )

    return eligible


def _validate_unique_grain(
    dataframe: pd.DataFrame,
) -> None:
    """Ensure one row exists per facility and service.

    Keys are stripped before comparison so whitespace differences
    cannot produce duplicate typed optimisation identities.
    """
    key_columns = [
        "facility_key",
        "service_code",
    ]

    keys = dataframe[key_columns].copy()

    for column in key_columns:
        keys[column] = (
            keys[column]
            .astype("string")
            .str.strip()
        )

    duplicate_mask = keys.duplicated(
        subset=key_columns,
        keep=False,
    )

    if not duplicate_mask.any():
        return

    duplicates = keys.loc[
        duplicate_mask,
        key_columns,
    ]

    duplicate_keys = sorted(
        {
            (
                str(row["facility_key"]),
                str(row["service_code"]),
            )
            for _, row in duplicates.iterrows()
        }
    )

    formatted = "; ".join(
        f"{facility_key} / {service_code}"
        for facility_key, service_code in duplicate_keys
    )

    raise ValueError(
        "Optimisation input contains duplicate "
        "facility-service rows: "
        f"{formatted}."
    )


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:
    """Return a required non-empty text value."""
    if (
        value is None
        or bool(pd.isna(cast(Any, value)))
    ):
        raise ValueError(
            f"{field_name} must not be missing."
        )

    text = str(value).strip()

    if not text:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    return text


def _required_non_negative_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    """Return one required non-negative integer volume."""
    if (
        value is None
        or bool(pd.isna(cast(Any, value)))
    ):
        raise ValueError(
            f"{field_name} must not be missing."
        )

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be an integer."
        )

    numeric_series = pd.to_numeric(
        pd.Series([value], dtype="object"),
        errors="coerce",
    )

    numeric = numeric_series.iloc[0]

    if bool(pd.isna(cast(Any, numeric))):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    numeric_value = float(numeric)

    if not numeric_value.is_integer():
        raise ValueError(
            f"{field_name} must be a whole number."
        )

    integer_value = int(numeric_value)

    if integer_value < 0:
        raise ValueError(
            f"{field_name} must not be negative."
        )

    return integer_value