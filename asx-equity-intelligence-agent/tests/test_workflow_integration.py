"""
Offline integration tests for the LangGraph orchestration pipeline.

These tests use the real run_pipeline entry point, with deterministic
test doubles for the agents. They check pipeline-level routing and
error handling without accessing Yahoo Finance or Gemini.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import orchestrator


TEST_TICKER = "BHP.AX"
TEST_DATE = "2026-11-17"


@pytest.fixture
def workflow_fakes(monkeypatch, tmp_path):
    """
    Replace external agent work with deterministic test doubles.

    The shared event list records which stages were invoked. Tests
    may configure failures without modifying production source.
    """
    events = []
    settings = {
        "fail_forecasting": False,
        "fail_risk": False,
        "fail_reporting": False,
    }

    forecast = SimpleNamespace(
        ticker=TEST_TICKER,
        as_of_date=TEST_DATE,
        forecast_date=TEST_DATE,
        predicted_return=0.008,
        confidence=0.65,
        mae=0.009,
        baseline_mae=0.010,
        signal="BUY",
        feature_importance={"return_5d": 0.35},
    )

    stock_risk = SimpleNamespace(
        ticker=TEST_TICKER,
        volatility_annual=0.25,
        var_95=-0.020,
        var_99=-0.035,
        max_drawdown=-0.15,
        beta=1.10,
        sharpe_score=0.32,
        risk_level="MEDIUM",
        recommendation_modifier="CONFIRM",
    )

    portfolio_risk = SimpleNamespace(
        total_stocks=1,
        sector_weights={"Basic Materials": 1.0},
        most_concentrated_sector="Basic Materials",
        concentration_risk="HIGH",
        average_beta=1.10,
        average_volatility=0.25,
        watchlist_var_95=-0.020,
        risk_summary="Synthetic test portfolio.",
    )

    recommendation = SimpleNamespace(
        ticker=TEST_TICKER,
        as_of_date=TEST_DATE,
        forecast_signal="BUY",
        final_signal="BUY",
        predicted_return=0.008,
        confidence=0.65,
        risk_level="MEDIUM",
        risk_modifier="CONFIRM",
        position_size=0.10,
        position_label="REDUCED",
        rationale="Synthetic test recommendation.",
        priority=1,
        sharpe_score=0.32,
    )

    strategy_output = SimpleNamespace(
        recommendations=[recommendation],
        actionable_signals=[recommendation],
        portfolio_risk=portfolio_risk,
        market_bias="NEUTRAL",
        strategy_summary="Synthetic test strategy.",
    )

    class BaseFakeAgent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def close(self):
            pass

    class FakeMarketDataAgent(BaseFakeAgent):
        def run(self, *args, **kwargs):
            events.append("market_data")

            return {
                "success": [TEST_TICKER],
                "failed": [],
                "total_records": 340,
                "macro_success": ["^AXJO"],
                "macro_failed": [],
                "fundamentals_success": [TEST_TICKER],
                "fundamentals_failed": [],
                "benchmark_ready": True,
                "forecast_ready": [TEST_TICKER],
                "forecast_blocked": [],
                "readiness_reasons": {},
                "warnings": [],
                "errors": [],
            }

    class FakeForecastingAgent(BaseFakeAgent):
        def run(self, tickers=None, **kwargs):
            events.append("forecasting")

            if settings["fail_forecasting"]:
                raise RuntimeError(
                    "Simulated forecasting failure"
                )

            if tickers is not None:
                assert TEST_TICKER in tickers

            return [forecast]

    class FakeRiskAgent(BaseFakeAgent):
        def run(self, forecasts, *args, **kwargs):
            events.append("risk")

            if settings["fail_risk"]:
                raise RuntimeError(
                    "Simulated risk failure"
                )

            assert len(forecasts) == 1
            assert forecasts[0].ticker == TEST_TICKER

            return [stock_risk], portfolio_risk

    class FakeStrategyAgent(BaseFakeAgent):
        def run(
            self,
            forecasts,
            stock_risks,
            portfolio_risk,
            **kwargs,
        ):
            events.append("strategy")

            assert len(forecasts) == 1
            assert len(stock_risks) == 1
            assert portfolio_risk is not None

            return strategy_output

    class FakeReportWriterAgent(BaseFakeAgent):
        def run(self, *args, **kwargs):
            events.append("reporting")

            if settings["fail_reporting"]:
                raise RuntimeError(
                    "Simulated report-writing failure"
                )

            report_path = (
                tmp_path / "asx_brief_20261117.md"
            )

            report_path.write_text(
                "# Synthetic ASX Research Brief\n",
                encoding="utf-8",
            )

            return str(report_path)

    monkeypatch.setattr(
        orchestrator,
        "MarketDataAgent",
        FakeMarketDataAgent,
    )
    monkeypatch.setattr(
        orchestrator,
        "ForecastingAgent",
        FakeForecastingAgent,
    )
    monkeypatch.setattr(
        orchestrator,
        "RiskAgent",
        FakeRiskAgent,
    )
    monkeypatch.setattr(
        orchestrator,
        "StrategyAgent",
        FakeStrategyAgent,
    )
    monkeypatch.setattr(
        orchestrator,
        "ReportWriterAgent",
        FakeReportWriterAgent,
    )

    return {
        "events": events,
        "settings": settings,
        "tmp_path": tmp_path,
    }


def execute_pipeline(fakes):
    return orchestrator.run_pipeline(
        db_path=str(
            fakes["tmp_path"] / "workflow_test.db"
        ),
        report_dir=str(fakes["tmp_path"]),
        use_llm=False,
    )


def test_successful_pipeline_runs_all_five_stages(
    workflow_fakes,
):
    result = execute_pipeline(workflow_fakes)

    assert result["success"] is True
    assert not result["errors"]

    assert workflow_fakes["events"] == [
        "market_data",
        "forecasting",
        "risk",
        "strategy",
        "reporting",
    ]

    assert result["report_path"] is not None

    report_path = Path(result["report_path"])

    assert report_path.is_file()
    assert "Synthetic ASX Research Brief" in (
        report_path.read_text(encoding="utf-8")
    )


def test_forecasting_failure_stops_later_stages(
    workflow_fakes,
):
    workflow_fakes["settings"][
        "fail_forecasting"
    ] = True

    result = execute_pipeline(workflow_fakes)

    assert result["success"] is False
    assert result["errors"]

    assert workflow_fakes["events"] == [
        "market_data",
        "forecasting",
    ]

    assert "risk" not in workflow_fakes["events"]
    assert "strategy" not in workflow_fakes["events"]
    assert "reporting" not in workflow_fakes["events"]


def test_risk_failure_stops_strategy_and_reporting(
    workflow_fakes,
):
    workflow_fakes["settings"]["fail_risk"] = True

    result = execute_pipeline(workflow_fakes)

    assert result["success"] is False
    assert result["errors"]

    assert workflow_fakes["events"] == [
        "market_data",
        "forecasting",
        "risk",
    ]

    assert "strategy" not in workflow_fakes["events"]
    assert "reporting" not in workflow_fakes["events"]


def test_report_failure_does_not_claim_success(
    workflow_fakes,
):
    workflow_fakes["settings"][
        "fail_reporting"
    ] = True

    result = execute_pipeline(workflow_fakes)

    assert result["success"] is False
    assert result["errors"]

    assert workflow_fakes["events"] == [
        "market_data",
        "forecasting",
        "risk",
        "strategy",
        "reporting",
    ]

    assert not result.get("report_path")