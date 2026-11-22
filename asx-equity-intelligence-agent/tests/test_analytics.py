"""Regression tests for forecasting, risk and strategy logic."""

import numpy as np
import pandas as pd
import pytest

from agents.forecasting_agent import (
    derive_signal,
    engineer_features,
)
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent


@pytest.mark.parametrize(
    ("predicted_return", "expected"),
    [
        (0.010, "BUY"),
        (0.005, "HOLD"),
        (0.000, "HOLD"),
        (-0.005, "HOLD"),
        (-0.010, "REDUCE"),
    ],
)
def test_forecast_signal_thresholds(
    predicted_return,
    expected,
):
    assert derive_signal(predicted_return) == expected


def test_feature_engineering_preserves_unlabelled_latest_row():
    dates = pd.bdate_range(
        "2026-01-01",
        periods=90,
    )

    time = np.arange(len(dates))

    closes = (
        100.0
        + 0.12 * time
        + 3.0 * np.sin(time / 5.0)
    )

    returns = pd.Series(closes).pct_change()

    prices = pd.DataFrame(
        {
            "date": dates,
            "close": closes,
            "volume": 100_000
            + 5_000 * np.cos(time / 3.0),
            "daily_return": returns,
        }
    )

    macro = pd.DataFrame(
        {
            "date": dates,
            "daily_return": (
                0.001 * np.sin(time / 7.0)
            ),
        }
    )

    result = engineer_features(prices, macro)

    assert len(result) == len(prices)
    assert result.iloc[-1]["date"] == dates[-1]
    assert pd.isna(result.iloc[-1]["target"])

    # Historical labelled observations must remain available.
    assert result["target"].notna().sum() == len(prices) - 1


def test_beta_aligns_observations_by_date():
    agent = RiskAgent(db_path=":memory:")

    try:
        market = pd.Series(
            [0.01, -0.01, 0.02, -0.02] * 10,
            index=pd.bdate_range(
                "2026-01-01",
                periods=40,
            ),
        )

        # Intentionally reverse the order of stock observations.
        # Positional alignment would produce an incorrect beta.
        stock = (2.0 * market).iloc[::-1]

        beta = agent._calculate_beta(
            stock,
            market,
        )

        assert beta == pytest.approx(2.0, abs=0.01)

    finally:
        agent.close()


def test_beta_is_unavailable_with_insufficient_overlap():
    agent = RiskAgent(db_path=":memory:")

    try:
        stock = pd.Series(
            [0.01] * 10,
            index=pd.bdate_range(
                "2026-01-01",
                periods=10,
            ),
        )

        market = pd.Series(
            [0.02] * 10,
            index=pd.bdate_range(
                "2026-03-01",
                periods=10,
            ),
        )

        beta = agent._calculate_beta(
            stock,
            market,
        )

        assert beta is None

    finally:
        agent.close()


def test_maximum_drawdown_uses_peak_to_trough():
    agent = RiskAgent(db_path=":memory:")

    try:
        closes = pd.Series(
            [100.0, 120.0, 90.0, 110.0]
        )

        drawdown = agent._calculate_max_drawdown(
            closes
        )

        assert drawdown == pytest.approx(-0.25)

    finally:
        agent.close()


@pytest.mark.parametrize(
    ("signal", "modifier", "expected"),
    [
        ("BUY", "CONFIRM", "BUY"),
        ("BUY", "DOWNGRADE", "HOLD"),
        ("BUY", "BLOCK", "HOLD"),
        ("REDUCE", "CONFIRM", "REDUCE"),
        ("REDUCE", "BLOCK", "HOLD"),
        ("HOLD", "CONFIRM", "HOLD"),
    ],
)
def test_strategy_signal_resolution(
    signal,
    modifier,
    expected,
):
    agent = StrategyAgent()

    assert agent._resolve_signal(
        signal,
        modifier,
    ) == expected


def test_hold_has_no_position_adjustment():
    agent = StrategyAgent()

    size, label = agent._calculate_position_size(
        final_signal="HOLD",
        confidence=0.90,
        risk_level="LOW",
        sharpe_score=1.0,
    )

    assert size == 0.0
    assert label == "NONE"


@pytest.mark.parametrize(
    ("confidence", "expected_size", "expected_label"),
    [
        (0.31, 0.038, "MINIMAL"),
        (0.51, 0.075, "REDUCED"),
        (0.65, 0.113, "REDUCED"),
    ],
)
def test_medium_risk_position_adjustments(
    confidence,
    expected_size,
    expected_label,
):
    agent = StrategyAgent()

    size, label = agent._calculate_position_size(
        final_signal="BUY",
        confidence=confidence,
        risk_level="MEDIUM",
        sharpe_score=0.2,
    )

    assert size == pytest.approx(expected_size, abs=0.001)
    assert label == expected_label