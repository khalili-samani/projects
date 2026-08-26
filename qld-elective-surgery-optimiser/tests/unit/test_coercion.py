"""Tests for controlled source-value coercion."""

from __future__ import annotations

import pandas as pd

from qld_surgery_optimiser.validation.coercion import (
    parse_numeric_series,
    parse_percentage_series,
)


NULL_TOKENS = [
    "",
    "NA",
    "N/A",
    "NULL",
    "null",
    "-",
    "--",
]


def test_numeric_parser_accepts_commas_and_whitespace() -> None:
    series = pd.Series(
        [
            "1,234",
            " 50 ",
            "0",
        ]
    )

    parsed, invalid = parse_numeric_series(
        series,
        null_tokens=NULL_TOKENS,
    )

    assert parsed.tolist() == [
        1234,
        50,
        0,
    ]

    assert invalid.sum() == 0


def test_numeric_parser_marks_invalid_non_null_values() -> None:
    series = pd.Series(
        [
            "10",
            "not available",
            "",
        ]
    )

    parsed, invalid = parse_numeric_series(
        series,
        null_tokens=NULL_TOKENS,
    )

    assert parsed.iloc[0] == 10
    assert pd.isna(parsed.iloc[1])
    assert pd.isna(parsed.iloc[2])

    assert invalid.tolist() == [
        False,
        True,
        False,
    ]


def test_percentage_parser_removes_percent_sign() -> None:
    series = pd.Series(
        [
            "95%",
            "87.5 %",
            "100",
        ]
    )

    parsed, invalid = parse_percentage_series(
        series,
        null_tokens=NULL_TOKENS,
    )

    assert parsed.tolist() == [
        95.0,
        87.5,
        100.0,
    ]

    assert invalid.sum() == 0