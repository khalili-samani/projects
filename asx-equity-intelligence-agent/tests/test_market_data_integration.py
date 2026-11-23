"""
Offline integration tests for market-data ingestion and forecast readiness.

The production MarketDataAgent and MarketDatabase are exercised with
deterministic provider responses and temporary DuckDB databases.

No Yahoo Finance network calls or existing market databases are used.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import agents.market_data_agent as market_module
from agents.market_data_agent import (
    FundamentalSnapshot,
    MarketDataAgent,
)


EQUITY = "BHP.AX"
BENCHMARK = "^AXJO"

TRADING_DATES = [
    "2026-11-13",
    "2026-11-16",
    "2026-11-17",
]


def make_equity_history(dates=None):
    """Return validated-format, synthetic OHLCV observations."""
    dates = dates or TRADING_DATES

    closes = [100.0, 102.0, 103.0][: len(dates)]

    # Construct plausible candles around each adjusted close.
    return pd.DataFrame(
        {
            "Open": [close - 0.5 for close in closes],
            "High": [close + 1.0 for close in closes],
            "Low": [close - 1.0 for close in closes],
            "Close": closes,
            "Volume": [100_000] * len(dates),
            "daily_return": [
                0.0,
                0.0200,
                0.0098039215686274,
            ][: len(dates)],
        },
        index=pd.to_datetime(dates),
    )


def make_benchmark_history(dates=None):
    """Return synthetic benchmark observations."""
    dates = dates or TRADING_DATES

    closes = [9000.0, 9050.0, 8980.0][: len(dates)]

    return pd.DataFrame(
        {
            "Close": closes,
            "daily_return": [
                0.0,
                0.0055555555555555,
                -0.0077348066298343,
            ][: len(dates)],
        },
        index=pd.to_datetime(dates),
    )


def make_fundamentals():
    """Construct a complete synthetic fundamentals snapshot."""
    return FundamentalSnapshot(
        ticker=EQUITY,
        fetched_at="2026-11-18T00:00:00",
        market_cap=150_000_000_000.0,
        pe_ratio=15.0,
        forward_pe=14.0,
        dividend_yield=0.04,
        fifty_two_week_high=110.0,
        fifty_two_week_low=80.0,
        beta=1.2,
        sector="Basic Materials",
        industry="Mining",
    )


@pytest.fixture
def ingestion_environment(tmp_path, monkeypatch):
    """
    Return a real market-data agent backed by a temporary database.

    Only network-dependent methods and the live date-window refresh
    are replaced. All database upserts and readiness logic remain real.
    """
    database_path = tmp_path / "ingestion_test.db"

    agent = MarketDataAgent(
        watchlist={EQUITY: "BHP"},
        macro_tickers={BENCHMARK: "ASX 200"},
        db_path=str(database_path),
    )

    # Keep fixture dates deterministic rather than using today's date.
    monkeypatch.setattr(
        agent,
        "_refresh_date_window",
        lambda: None,
    )

    agent.start_date = "2026-11-13"
    agent.end_date = "2026-11-18"

    histories = {
        EQUITY: make_equity_history(),
        BENCHMARK: make_benchmark_history(),
    }

    failures = set()
    fundamentals_failure = {"enabled": False}

    def fake_download(ticker, *args, **kwargs):
        if ticker in failures:
            raise RuntimeError(
                f"Simulated download failure: {ticker}"
            )

        if ticker not in histories:
            raise AssertionError(
                f"Unexpected provider request: {ticker}"
            )

        return histories[ticker].copy()

    def fake_fundamentals(ticker):
        assert ticker == EQUITY

        if fundamentals_failure["enabled"]:
            raise RuntimeError(
                "Simulated fundamentals failure"
            )

        return make_fundamentals()

    monkeypatch.setattr(
        agent,
        "_download_history",
        fake_download,
    )

    monkeypatch.setattr(
        agent,
        "_fetch_fundamentals",
        fake_fundamentals,
    )

    yield {
        "agent": agent,
        "histories": histories,
        "failures": failures,
        "fundamentals_failure": fundamentals_failure,
        "database_path": database_path,
    }

    agent.close()


def count_rows(agent, table, ticker):
    """Count stored observations using a parameterised query."""
    return agent.db.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE ticker = ?",
        [ticker],
    ).fetchone()[0]


def test_successful_ingestion_persists_aligned_market_data(
    ingestion_environment,
):
    environment = ingestion_environment
    agent = environment["agent"]

    result = agent.run()

    assert result["success"] == [EQUITY]
    assert result["failed"] == []

    assert result["macro_success"] == [BENCHMARK]
    assert result["macro_failed"] == []

    assert result["benchmark_ready"] is True
    assert result["forecast_ready"] == [EQUITY]
    assert result["forecast_blocked"] == []

    assert result["fundamentals_success"] == [EQUITY]
    assert result["fundamentals_failed"] == []

    assert result["total_records"] == 3

    assert count_rows(agent, "ohlcv", EQUITY) == 3
    assert count_rows(agent, "macro", BENCHMARK) == 3
    assert count_rows(agent, "fundamentals", EQUITY) == 1

    stored_equity_dates = [
        str(row[0])
        for row in agent.db.conn.execute(
            """
            SELECT date
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date
            """,
            [EQUITY],
        ).fetchall()
    ]

    stored_benchmark_dates = [
        str(row[0])
        for row in agent.db.conn.execute(
            """
            SELECT date
            FROM macro
            WHERE ticker = ?
            ORDER BY date
            """,
            [BENCHMARK],
        ).fetchall()
    ]

    assert stored_equity_dates == TRADING_DATES
    assert stored_benchmark_dates == TRADING_DATES


def test_missing_benchmark_session_blocks_forecasting(
    ingestion_environment,
):
    environment = ingestion_environment
    agent = environment["agent"]

    # Remove 16 November from the benchmark response.
    environment["histories"][BENCHMARK] = (
        make_benchmark_history(
            ["2026-11-13", "2026-11-17"]
        )
    )

    result = agent.run()

    assert result["success"] == [EQUITY]
    assert result["macro_success"] == [BENCHMARK]

    # Benchmark availability alone is insufficient for readiness.
    assert result["benchmark_ready"] is True

    assert result["forecast_ready"] == []
    assert result["forecast_blocked"] == [EQUITY]

    assert "2026-11-16" in (
        result["readiness_reasons"][EQUITY]
    )

    # The equity is retained; its price observations are valid.
    assert count_rows(agent, "ohlcv", EQUITY) == 3
    assert count_rows(agent, "macro", BENCHMARK) == 2


def test_benchmark_failure_does_not_discard_equity(
    ingestion_environment,
):
    environment = ingestion_environment
    agent = environment["agent"]

    environment["failures"].add(BENCHMARK)

    result = agent.run()

    assert result["macro_success"] == []
    assert result["macro_failed"] == [BENCHMARK]

    assert result["benchmark_ready"] is False

    assert result["success"] == [EQUITY]
    assert result["forecast_ready"] == []
    assert result["forecast_blocked"] == [EQUITY]

    assert count_rows(agent, "ohlcv", EQUITY) == 3
    assert count_rows(agent, "macro", BENCHMARK) == 0

    assert result["errors"]
    assert result["warnings"]


def test_equity_download_failure_is_reported(
    ingestion_environment,
):
    environment = ingestion_environment
    agent = environment["agent"]

    environment["failures"].add(EQUITY)

    result = agent.run()

    assert result["success"] == []
    assert result["failed"] == [EQUITY]

    assert result["macro_success"] == [BENCHMARK]
    assert result["forecast_ready"] == []

    assert result["total_records"] == 0

    assert count_rows(agent, "ohlcv", EQUITY) == 0
    assert count_rows(agent, "macro", BENCHMARK) == 3

    assert result["fundamentals_success"] == []

    assert any(
        "Equity ingestion failed" in error
        for error in result["errors"]
    )


def test_fundamentals_failure_preserves_price_readiness(
    ingestion_environment,
):
    environment = ingestion_environment
    agent = environment["agent"]

    environment["fundamentals_failure"]["enabled"] = True

    result = agent.run()

    assert result["success"] == [EQUITY]
    assert result["forecast_ready"] == [EQUITY]

    assert result["fundamentals_success"] == []
    assert result["fundamentals_failed"] == [EQUITY]

    assert count_rows(agent, "ohlcv", EQUITY) == 3
    assert count_rows(agent, "macro", BENCHMARK) == 3
    assert count_rows(agent, "fundamentals", EQUITY) == 0

    assert any(
        "Fundamentals ingestion failed" in error
        for error in result["errors"]
    )


def test_repeated_ingestion_does_not_duplicate_observations(
    ingestion_environment,
):
    agent = ingestion_environment["agent"]

    first = agent.run()
    second = agent.run()

    assert first["forecast_ready"] == [EQUITY]
    assert second["forecast_ready"] == [EQUITY]

    # Both runs processed observations, but the database uses
    # ticker/date uniqueness to prevent duplicates.
    assert first["total_records"] == 3
    assert second["total_records"] == 3

    assert count_rows(agent, "ohlcv", EQUITY) == 3
    assert count_rows(agent, "macro", BENCHMARK) == 3


def test_download_history_excludes_incomplete_observation(
    tmp_path,
    monkeypatch,
):
    """
    Exercise the real Yahoo download adapter using a fake response.

    The exclusive end date is 18 November. Even if Yahoo includes
    an 18 November candle, it must not enter the returned history.
    """
    agent = MarketDataAgent(
        watchlist={EQUITY: "BHP"},
        macro_tickers={BENCHMARK: "ASX 200"},
        db_path=str(tmp_path / "download_test.db"),
    )

    try:
        agent.start_date = "2026-11-13"
        agent.end_date = "2026-11-18"

        provider_dates = pd.to_datetime(
            [
                "2026-11-12",
                "2026-11-13",
                "2026-11-16",
                "2026-11-17",
                "2026-11-18",
            ]
        )

        closes = [100.0, 102.0, 103.0, 104.0, 105.0]

        provider_history = pd.DataFrame(
            {
                "Open": closes,
                "High": [value + 1 for value in closes],
                "Low": [value - 1 for value in closes],
                "Close": closes,
                "Volume": [100_000] * len(closes),
            },
            index=provider_dates,
        )

        requests = []

        def fake_yahoo_download(ticker, **kwargs):
            requests.append((ticker, kwargs))
            return provider_history.copy()

        monkeypatch.setattr(
            market_module.yf,
            "download",
            fake_yahoo_download,
        )

        # The fixture dates are historical. Freshness itself is
        # tested separately, so suppress it for this adapter test.
        monkeypatch.setattr(
            agent,
            "_validate_freshness",
            lambda *args, **kwargs: None,
        )

        result = agent._download_history(
            EQUITY,
            required_columns=(
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ),
        )

        assert requests
        assert requests[0][0] == EQUITY

        assert len(result) == 3

        assert result.index[0].date() == date(
            2026, 11, 13
        )

        assert result.index[-1].date() == date(
            2026, 11, 17
        )

        assert date(2026, 11, 18) not in {
            timestamp.date()
            for timestamp in result.index
        }

        # The pre-window close from 12 November is needed
        # to calculate the first in-window return correctly.
        assert result.iloc[0]["daily_return"] == (
            pytest.approx(0.02)
        )

    finally:
        agent.close()


def test_freshness_validation_rejects_stale_data():
    reference_date = date(2026, 11, 18)

    MarketDataAgent._validate_freshness(
        ticker=EQUITY,
        latest_date=date(2026, 11, 17),
        reference_date=reference_date,
    )

    with pytest.raises(ValueError, match="stale"):
        MarketDataAgent._validate_freshness(
            ticker=EQUITY,
            latest_date=reference_date - timedelta(days=30),
            reference_date=reference_date,
        )

    with pytest.raises(ValueError, match="current-day"):
        MarketDataAgent._validate_freshness(
            ticker=EQUITY,
            latest_date=reference_date,
            reference_date=reference_date,
        )

    with pytest.raises(ValueError, match="future"):
        MarketDataAgent._validate_freshness(
            ticker=EQUITY,
            latest_date=reference_date + timedelta(days=1),
            reference_date=reference_date,
        )