"""Build and validate optimisation-ready inputs from analytical data."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any, cast

import pandas as pd

from qld_surgery_optimiser.optimisation.models import (
    OptimisationInputRow,
    ResourceKind,
)


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


def build_optimisation_inputs(
    dataframe: pd.DataFrame,
    *,
    resource_kind: ResourceKind,
    reporting_period: date | pd.Timestamp | None = None,
) -> list[OptimisationInputRow]:
    """Convert canonical analytical data into optimisation input rows.

    The optimisation layer operates on one published source family at
    a time. Specialty and category data are therefore not merged into
    a synthetic joint distribution.

    Parameters
    ----------
    dataframe:
        Canonical or longitudinal analytical dataset.
    resource_kind:
        Published source family to optimise. Supported values are
        ``"specialty"`` and ``"category"``.
    reporting_period:
        Optional reporting period to select. When omitted, the latest
        reporting period available for the selected resource family is
        used.

    Returns
    -------
    list[OptimisationInputRow]
        Validated optimisation input rows.

    Raises
    ------
    ValueError
        If the dataframe is missing required fields, contains invalid
        optimisation values, contains duplicate optimisation keys, or
        has no rows for the requested source family/reporting period.
    """
    _validate_required_columns(
        dataframe
    )

    filtered = _filter_resource_kind(
        dataframe,
        resource_kind=resource_kind,
    )

    if filtered.empty:
        raise ValueError(
            "No analytical rows found for resource_kind "
            f"{resource_kind!r}."
        )

    filtered = _normalise_reporting_period(
        filtered
    )

    selected_period = _select_reporting_period(
        filtered,
        reporting_period=reporting_period,
    )

    filtered = filtered.loc[
        filtered["report_month"]
        == selected_period
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No analytical rows found for reporting period "
            f"{selected_period.date().isoformat()}."
        )

    _validate_unique_grain(
        filtered
    )

    rows: list[
        OptimisationInputRow
    ] = []

    for row in filtered.itertuples(
        index=False
    ):
        facility_key = _required_text(
            getattr(
                row,
                "facility_key",
            ),
            field_name="facility_key",
        )

        facility_name = _required_text(
            getattr(
                row,
                "facility_name",
            ),
            field_name="facility_name",
        )

        service_code = _required_text(
            getattr(
                row,
                "service_code",
            ),
            field_name="service_code",
        )

        service_name = _required_text(
            getattr(
                row,
                "service_name",
            ),
            field_name="service_name",
        )

        vol_waiting = _required_non_negative_integer(
            getattr(
                row,
                "vol_waiting",
            ),
            field_name="vol_waiting",
        )

        vol_long_waits = _required_non_negative_integer(
            getattr(
                row,
                "vol_long_waits",
            ),
            field_name="vol_long_waits",
        )

        if (
            vol_long_waits
            > vol_waiting
        ):
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
                reporting_period=(
                    selected_period.date()
                ),
                vol_waiting=vol_waiting,
                vol_long_waits=vol_long_waits,
            )
        )

    return rows


def _validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Ensure the analytical dataframe has the required contract."""
    missing = (
        REQUIRED_COLUMNS
        - set(
            dataframe.columns
        )
    )

    if not missing:
        return

    missing_text = ", ".join(
        sorted(
            missing
        )
    )

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
        dataframe[
            "resource_kind"
        ]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    return dataframe.loc[
        values
        == resource_kind
    ].copy()


def _normalise_reporting_period(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise reporting periods to pandas timestamps."""
    result = dataframe.copy()

    result[
        "report_month"
    ] = pd.to_datetime(
        cast(
            Any,
            result[
                "report_month"
            ],
        ),
        errors="coerce",
    )

    if result[
        "report_month"
    ].isna().any():
        invalid_count = int(
            result[
                "report_month"
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Optimisation input contains "
            f"{invalid_count} invalid reporting period value(s)."
        )

    return result


def _select_reporting_period(
    dataframe: pd.DataFrame,
    *,
    reporting_period: date | pd.Timestamp | None,
) -> pd.Timestamp:
    """Resolve the reporting period used for optimisation."""
    if reporting_period is None:
        maximum = dataframe[
            "report_month"
        ].max()

        if pd.isna(
            maximum
        ):
            raise ValueError(
                "Unable to determine the latest reporting period."
            )

        return pd.Timestamp(
            maximum
        ).normalize()

    requested = pd.Timestamp(
        reporting_period
    ).normalize()

    available = {
        pd.Timestamp(
            value
        ).normalize()
        for value
        in dataframe[
            "report_month"
        ].dropna()
    }

    if requested not in available:
        available_text = ", ".join(
            sorted(
                value.date().isoformat()
                for value
                in available
            )
        )

        raise ValueError(
            "Requested reporting period "
            f"{requested.date().isoformat()} is not available. "
            f"Available periods: {available_text}."
        )

    return requested


def _validate_unique_grain(
    dataframe: pd.DataFrame,
) -> None:
    """Ensure one row exists per facility and service."""
    key_columns = [
        "facility_key",
        "service_code",
    ]

    duplicate_mask = dataframe.duplicated(
        subset=key_columns,
        keep=False,
    )

    if not duplicate_mask.any():
        return

    duplicates = dataframe.loc[
        duplicate_mask,
        key_columns,
    ]

    duplicate_keys = sorted(
        {
            (
                str(
                    row[
                        "facility_key"
                    ]
                ),
                str(
                    row[
                        "service_code"
                    ]
                ),
            )
            for _, row
            in duplicates.iterrows()
        }
    )

    formatted = "; ".join(
        f"{facility_key} / {service_code}"
        for (
            facility_key,
            service_code,
        )
        in duplicate_keys
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
        or bool(
            pd.isna(
                cast(
                    Any,
                    value,
                )
            )
        )
    ):
        raise ValueError(
            f"{field_name} must not be missing."
        )

    text = str(
        value
    ).strip()

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
        or bool(
            pd.isna(
                cast(
                    Any,
                    value,
                )
            )
        )
    ):
        raise ValueError(
            f"{field_name} must not be missing."
        )

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{field_name} must be an integer."
        )

    numeric = pd.to_numeric(
        value,
        errors="coerce",
    )

    if bool(
        pd.isna(
            cast(
                Any,
                numeric,
            )
        )
    ):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    numeric_value = float(
        numeric
    )

    if not numeric_value.is_integer():
        raise ValueError(
            f"{field_name} must be a whole number."
        )

    integer_value = int(
        numeric_value
    )

    if integer_value < 0:
        raise ValueError(
            f"{field_name} must not be negative."
        )

    return integer_value