"""Regression tests for market-data validation and storage."""

from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from agents.market_data_agent import (
    MarketDatabase,
    OHLCVRecord,
    assess_forecast_readiness,
    calculate_daily_returns,
)


def make_record(**overrides):
    values = {
        "ticker": "BHP.AX",
        "date": "2026-11-16",
        "open": 50.0,
        "high": 52.0,
        "low": 49.0,
        "close": 51.0,
        "volume": 100_000,
        "daily_return": 0.02,
    }
    values.update(overrides)
    return OHLCVRecord(**values)


def test_valid_equity_record():
    record = make_record()

    assert record.ticker == "BHP.AX"
    assert record.close == 51.0
    assert record.daily_return == 0.02


@pytest.mark.parametrize(
    "overrides",
    [
        {"close": -1.0},
        {"close": float("nan")},
        {"high": 48.0},
        {"low": 53.0},
        {"volume": -100},
        {"daily_return": float("inf")},
        {"daily_return": 0.70},
    ],
)
def test_invalid_equity_records_are_rejected(overrides):
    with pytest.raises(ValidationError):
        make_record(**overrides)


def test_return_calculation_uses_preceding_close():
    history = pd.DataFrame(
        {"Close": [100.0, 102.0, 103.02]},
        index=pd.to_datetime(
            ["2026-11-12", "2026-11-13", "2026-11-16"]
        ),
    )

    result = calculate_daily_returns(
        history,
        start_date=date(2026, 11, 13),
        end_date=date(2026, 11, 17),
    )

    assert len(result) == 2
    assert result.iloc[0]["daily_return"] == pytest.approx(0.02)
    assert result.iloc[1]["daily_return"] == pytest.approx(0.01)


def test_missing_preceding_close_is_not_replaced_with_zero():
    history = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.to_datetime(["2026-11-16"]),
    )

    with pytest.raises(ValueError, match="calculable"):
        calculate_daily_returns(
            history,
            start_date=date(2026, 11, 16),
            end_date=date(2026, 11, 17),
        )


def test_aligned_benchmark_is_forecast_ready():
    dates = [
        "2026-11-13",
        "2026-11-16",
        "2026-11-17",
    ]

    ready, reason = assess_forecast_readiness(
        equity_dates=dates,
        benchmark_dates=dates,
    )

    assert ready is True
    assert "aligned" in reason.lower()


def test_missing_benchmark_session_blocks_forecast():
    ready, reason = assess_forecast_readiness(
        equity_dates=[
            "2026-11-13",
            "2026-11-16",
            "2026-11-17",
        ],
        benchmark_dates=[
            "2026-11-13",
            "2026-11-17",
        ],
    )

    assert ready is False
    assert "2026-11-16" in reason


def test_latest_benchmark_date_mismatch_blocks_forecast():
    ready, reason = assess_forecast_readiness(
        equity_dates=[
            "2026-11-13",
            "2026-11-16",
        ],
        benchmark_dates=[
            "2026-11-13",
        ],
    )

    assert ready is False
    assert "does not match" in reason


def test_ohlcv_upsert_is_idempotent():
    with MarketDatabase(":memory:") as db:
        record = make_record()

        db.upsert_ohlcv([record])
        db.upsert_ohlcv([record])

        count = db.conn.execute(
            "SELECT COUNT(*) FROM ohlcv"
        ).fetchone()[0]

        assert count == 1


def test_transaction_rolls_back_on_failure():
    with MarketDatabase(":memory:") as db:
        with pytest.raises(RuntimeError):
            with db.transaction():
                db.conn.execute(
                    """
                    INSERT INTO ohlcv
                    VALUES (
                        'BHP.AX',
                        DATE '2026-11-16',
                        50, 52, 49, 51, 100000, 0.02
                    )
                    """
                )

                raise RuntimeError("Simulated write failure")

        count = db.conn.execute(
            "SELECT COUNT(*) FROM ohlcv"
        ).fetchone()[0]

        assert count == 0