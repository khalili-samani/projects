"""
Offline integration tests for the forecasting, risk and strategy agents.

These tests use a temporary DuckDB file and deterministic synthetic
market data. No Yahoo Finance calls, API credentials or historical
production databases are required.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents.forecasting_agent import (
    FEATURE_COLS,
    ForecastingAgent,
    engineer_features,
)
from agents.market_data_agent import MarketDatabase
from agents.risk_agent import PortfolioRisk, RiskAgent
from agents.strategy_agent import StrategyAgent


TICKER = "BHP.AX"
BENCHMARK = "^AXJO"
OBSERVATIONS = 340


def make_market_history(
    observations: int = OBSERVATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate repeatable equity and benchmark observations.

    Equity returns respond to benchmark returns, with an additional
    deterministic stock-specific component. No random numbers are used.
    """
    dates = pd.bdate_range(
        start="2025-01-06",
        periods=observations,
    )

    equity_rows = []
    benchmark_rows = []

    equity_close = 45.0
    benchmark_close = 8_000.0

    for index, timestamp in enumerate(dates):
        market_return = (
            0.0015 * math.sin(index * 0.31)
            + 0.0020 * math.cos(index * 0.13)
        )

        equity_return = (
            1.3 * market_return
            + 0.0030 * math.sin(index * 0.19)
            + 0.0001
        )

        if index == 0:
            market_return = 0.0
            equity_return = 0.0

        previous_equity_close = equity_close
        previous_benchmark_close = benchmark_close

        equity_close *= 1.0 + equity_return
        benchmark_close *= 1.0 + market_return

        volume = int(
            150_000
            + 20_000 * (1.0 + math.sin(index * 0.17))
        )

        equity_rows.append(
            {
                "ticker": TICKER,
                "date": timestamp.date(),
                "open": previous_equity_close,
                "high": max(
                    previous_equity_close,
                    equity_close,
                ) * 1.005,
                "low": min(
                    previous_equity_close,
                    equity_close,
                ) * 0.995,
                "close": equity_close,
                "volume": volume,
                "daily_return": equity_return,
            }
        )

        benchmark_rows.append(
            {
                "ticker": BENCHMARK,
                "date": timestamp.date(),
                "close": benchmark_close,
                "daily_return": market_return,
            }
        )

    return (
        pd.DataFrame(equity_rows),
        pd.DataFrame(benchmark_rows),
    )


@pytest.fixture
def integration_database(tmp_path):
    """
    Create and populate a temporary database using the application's
    real schema.

    The database is isolated from local historical and validation data.
    """
    path = tmp_path / "integration_market_data.db"

    equities, benchmark = make_market_history()

    with MarketDatabase(str(path)) as database:
        equity_values = [
            (
                row.ticker,
                row.date,
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
                row.daily_return,
            )
            for row in equities.itertuples(index=False)
        ]

        database.conn.executemany(
            """
            INSERT INTO ohlcv (
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                daily_return
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            equity_values,
        )

        benchmark_values = [
            (
                row.ticker,
                row.date,
                row.close,
                row.daily_return,
            )
            for row in benchmark.itertuples(index=False)
        ]

        database.conn.executemany(
            """
            INSERT INTO macro (
                ticker,
                date,
                close,
                daily_return
            )
            VALUES (?, ?, ?, ?)
            """,
            benchmark_values,
        )

        stored_equities = database.conn.execute(
            "SELECT COUNT(*) FROM ohlcv"
        ).fetchone()[0]

        stored_benchmark = database.conn.execute(
            "SELECT COUNT(*) FROM macro WHERE ticker = ?",
            [BENCHMARK],
        ).fetchone()[0]

        assert stored_equities == OBSERVATIONS
        assert stored_benchmark == OBSERVATIONS

    return str(path)


def run_forecast(database_path: str):
    """Run the production forecasting agent against synthetic data."""
    agent = ForecastingAgent(
        db_path=database_path,
        min_training_rows=100,
    )

    try:
        return agent.run(tickers=[TICKER])
    finally:
        agent.close()


def test_complete_forecasting_run(integration_database):
    """
    Exercise feature preparation, chronological model validation,
    final model fitting and forecast construction.
    """
    forecasts = run_forecast(integration_database)

    assert len(forecasts) == 1

    forecast = forecasts[0]

    assert forecast.ticker == TICKER
    assert forecast.signal in {"BUY", "HOLD", "REDUCE"}

    assert math.isfinite(forecast.predicted_return)
    assert math.isfinite(forecast.mae)

    assert forecast.mae >= 0.0
    assert 0.0 <= forecast.confidence <= 1.0

    assert forecast.as_of_date == (
        make_market_history()[0].iloc[-1]["date"].isoformat()
    )

    assert forecast.feature_importance
    assert all(
        name in FEATURE_COLS
        for name in forecast.feature_importance
    )


def test_latest_observation_is_not_a_training_target():
    """
    The latest observation is a feature row for the next session.
    Its future return is unknown and cannot be a training label.
    """
    equities, benchmark = make_market_history()

    features = engineer_features(
        equities.drop(columns=["ticker"]),
        benchmark.drop(columns=["ticker"]),
    )

    assert len(features) == OBSERVATIONS

    last_row = features.iloc[-1]
    previous_row = features.iloc[-2]

    assert pd.isna(last_row["target"])

    assert previous_row["target"] == pytest.approx(
        equities.iloc[-1]["daily_return"]
    )

    # Feature merging may convert datetime.date to pandas Timestamp.
    # Compare calendar dates rather than different Python date types.
    assert (
        pd.Timestamp(last_row["date"]).date()
        == equities.iloc[-1]["date"]
    )

    assert all(
        pd.notna(last_row[column])
        for column in FEATURE_COLS
    )


def test_complete_stock_risk_assessment(
    integration_database,
):
    """
    Exercise actual database reads and risk calculations for a
    generated forecast rather than a mock StockForecast object.
    """
    forecasts = run_forecast(integration_database)

    agent = RiskAgent(db_path=integration_database)

    try:
        risk = agent.assess_stock(forecasts[0])
    finally:
        agent.close()

    assert risk is not None
    assert risk.ticker == TICKER

    assert math.isfinite(risk.volatility_annual)
    assert risk.volatility_annual > 0.0

    assert -1.0 <= risk.var_99 <= risk.var_95
    assert risk.var_95 <= 1.0

    assert -1.0 <= risk.max_drawdown <= 0.0

    assert risk.beta is not None
    assert math.isfinite(risk.beta)

    assert risk.risk_level in {"LOW", "MEDIUM", "HIGH"}

    assert risk.recommendation_modifier in {
        "CONFIRM",
        "DOWNGRADE",
        "BLOCK",
    }


def test_full_analytical_chain(
    integration_database,
):
    """
    Exercise the real forecasting, risk and strategy agents.

    This does not include ingestion or report generation.
    """
    forecasts = run_forecast(integration_database)

    risk_agent = RiskAgent(
        db_path=integration_database
    )

    try:
        risk = risk_agent.assess_stock(forecasts[0])
    finally:
        risk_agent.close()

    assert risk is not None

    portfolio_risk = PortfolioRisk(
        total_stocks=1,
        sector_weights={"Basic Materials": 1.0},
        most_concentrated_sector="Basic Materials",
        concentration_risk="HIGH",
        average_beta=risk.beta,
        average_volatility=risk.volatility_annual,
        watchlist_var_95=risk.var_95,
        risk_summary="Synthetic one-security watchlist.",
    )

    strategy = StrategyAgent()

    output = strategy.run(
        forecasts=forecasts,
        stock_risks=[risk],
        portfolio_risk=portfolio_risk,
    )

    assert len(output.recommendations) == 1

    recommendation = output.recommendations[0]

    assert recommendation.ticker == TICKER
    assert recommendation.as_of_date == forecasts[0].as_of_date
    assert recommendation.forecast_signal == forecasts[0].signal

    assert recommendation.final_signal in {
        "BUY",
        "HOLD",
        "REDUCE",
    }

    assert recommendation.risk_modifier == (
        risk.recommendation_modifier
    )

    assert output.market_bias in {
        "BULLISH",
        "NEUTRAL",
        "BEARISH",
    }

    assert 0.0 <= recommendation.position_size <= 1.0

    if recommendation.final_signal == "HOLD":
        assert recommendation.position_size == 0.0
        assert recommendation.position_label == "NONE"

    assert recommendation.rationale


def test_synthetic_data_contains_consistent_returns():
    """
    Check that synthetic returns correspond to changes in closing
    prices, excluding the first seed observation.
    """
    equities, benchmark = make_market_history()

    for frame in (equities, benchmark):
        implied_returns = frame["close"].pct_change().iloc[1:]
        stored_returns = frame["daily_return"].iloc[1:]

        np.testing.assert_allclose(
            implied_returns.to_numpy(),
            stored_returns.to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )


def test_generated_dates_are_ordered_and_aligned():
    equities, benchmark = make_market_history()

    assert len(equities) == OBSERVATIONS
    assert len(benchmark) == OBSERVATIONS

    assert equities["date"].is_monotonic_increasing
    assert benchmark["date"].is_monotonic_increasing

    assert equities["date"].tolist() == (
        benchmark["date"].tolist()
    )

    assert isinstance(equities.iloc[-1]["date"], date)