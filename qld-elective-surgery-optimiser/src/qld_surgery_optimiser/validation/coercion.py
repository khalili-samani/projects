"""Controlled source-value coercion.

Raw source values are intentionally read as strings first. This module
converts known fields while retaining original values for quality review.
"""

from __future__ import annotations

import re
from typing import Any, cast

import pandas as pd

from qld_surgery_optimiser.config import ValidationConfig

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%b-%y",
    "%b-%Y",
    "%B-%Y",
    "%b%Y",
    "%Y-%m",
)


def normalise_nulls(
    series: pd.Series,
    *,
    null_tokens: list[str],
) -> pd.Series:
    """Replace configured textual null representations with pandas NA."""
    token_set = {
        token.strip().casefold()
        for token in null_tokens
    }

    def convert(value: object) -> object:
        if value is None or bool(
            pd.isna(cast(Any, value))
        ):
            return pd.NA

        text = str(value).strip()

        if text.casefold() in token_set:
            return pd.NA

        return text

    return series.map(convert)


def parse_numeric_series(
    series: pd.Series,
    *,
    null_tokens: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Convert a textual series to numeric.

    Returns:
        Tuple containing:
        - parsed nullable numeric values;
        - boolean mask identifying non-null values that failed parsing.
    """
    cleaned = normalise_nulls(
        series,
        null_tokens=null_tokens,
    )

    cleaned = cleaned.map(
        lambda value: (
            re.sub(r"[,\s]", "", str(value))
            if not pd.isna(value)
            else pd.NA
        )
    )

    parsed = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    invalid_mask = (
        cleaned.notna()
        & parsed.isna()
    )

    return parsed, invalid_mask


def parse_percentage_series(
    series: pd.Series,
    *,
    null_tokens: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Parse percentage values such as '95', '95%' or '95.0 %'."""
    cleaned = normalise_nulls(
        series,
        null_tokens=null_tokens,
    )

    cleaned = cleaned.map(
        lambda value: (
            re.sub(
                r"[%\s,]",
                "",
                str(value),
            )
            if not pd.isna(value)
            else pd.NA
        )
    )

    parsed = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    invalid_mask = (
        cleaned.notna()
        & parsed.isna()
    )

    return parsed, invalid_mask


def parse_date_series(
    series: pd.Series,
    *,
    null_tokens: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Parse explicitly supported date and reporting-period formats.

    Australian day/month/year conventions are used for numeric dates.
    Month-year reporting labels are represented by the first day
    of their respective month.

    Unrecognised non-null values remain parse failures.
    """
    cleaned = normalise_nulls(
        series,
        null_tokens=null_tokens,
    )

    parsed = pd.Series(
        pd.NaT,
        index=cleaned.index,
        dtype="datetime64[ns]",
    )

    for date_format in DATE_FORMATS:
        remaining = cleaned.notna() & parsed.isna()

        if not remaining.any():
            break

        candidates = pd.to_datetime(
            cleaned.loc[remaining],
            format=date_format,
            errors="coerce",
        )

        parsed.loc[remaining] = candidates

    invalid_mask = (
        cleaned.notna()
        & parsed.isna()
    )

    return parsed, invalid_mask


def coerce_known_columns(
    dataframe: pd.DataFrame,
    *,
    config: ValidationConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.Series],
]:
    """Coerce known source fields and return parse-failure masks."""
    frame = dataframe.copy()

    failures: dict[str, pd.Series] = {}

    for column in config.numeric_volume_columns:
        if column not in frame.columns:
            continue

        parsed, invalid = parse_numeric_series(
            frame[column],
            null_tokens=config.null_tokens,
        )

        frame[column] = parsed
        failures[column] = invalid

    for column in config.percentage_columns:
        if column not in frame.columns:
            continue

        parsed, invalid = parse_percentage_series(
            frame[column],
            null_tokens=config.null_tokens,
        )

        frame[column] = parsed
        failures[column] = invalid

    for column in config.date_columns:
        if column not in frame.columns:
            continue

        parsed, invalid = parse_date_series(
            frame[column],
            null_tokens=config.null_tokens,
        )

        frame[column] = parsed
        failures[column] = invalid

    return frame, failures