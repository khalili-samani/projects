"""Controlled source-value coercion.

Raw source values are intentionally read as strings first. This module
converts known fields while retaining original values for quality review.
"""

from __future__ import annotations

import re

import pandas as pd

from qld_surgery_optimiser.config import ValidationConfig


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
        if value is None or pd.isna(value):
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
    """Parse dates without silently accepting unparseable non-null values."""
    cleaned = normalise_nulls(
        series,
        null_tokens=null_tokens,
    )

    parsed = pd.to_datetime(
        cleaned,
        errors="coerce",
        dayfirst=False,
    )

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