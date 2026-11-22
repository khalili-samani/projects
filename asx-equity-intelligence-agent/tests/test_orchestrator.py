"""Regression tests for LangGraph workflow safety."""

from types import SimpleNamespace

import orchestrator


def initial_state():
    return {
        "db_path": ":memory:",
        "report_dir": "reports",
        "use_llm": False,
        "run_date": "2026-11-18",
        "market_data_success": False,
        "market_data_tickers": [],
        "forecast_ready_tickers": [],
        "forecasts": [],
        "stock_risks": [],
        "portfolio_risk": None,
        "strategy_output": None,
        "report_path": None,
        "errors": [],
        "warnings": [],
        "current_stage": "starting",
    }


def test_market_data_without_ready_tickers_routes_to_error():
    state = initial_state()

    state.update(
        {
            "current_stage": "market_data_blocked",
            "market_data_tickers": ["BHP.AX"],
            "forecast_ready_tickers": [],
        }
    )

    assert orchestrator.after_market_data(state) == "error"


def test_market_data_with_ready_ticker_routes_to_forecasting():
    state = initial_state()

    state.update(
        {
            "current_stage": "market_data_complete",
            "market_data_tickers": ["BHP.AX"],
            "forecast_ready_tickers": ["BHP.AX"],
        }
    )

    assert (
        orchestrator.after_market_data(state)
        == "forecasting"
    )


def test_forecasting_without_results_routes_to_error():
    state = initial_state()
    state["current_stage"] = "forecasting_failed"

    assert orchestrator.after_forecasting(state) == "error"


def test_report_failure_routes_to_error():
    state = initial_state()
    state["current_stage"] = "report_failed"

    assert orchestrator.after_report_writer(state) == "error"


def test_successful_report_routes_to_completion():
    state = initial_state()
    state["current_stage"] = "report_complete"
    state["report_path"] = "reports/test.md"

    assert orchestrator.after_report_writer(state) == "complete"


def test_pipeline_blocks_forecasting_when_data_not_ready(
    monkeypatch,
):
    """Test the actual graph route, without network access."""

    class FakeMarketDataAgent:
        def __init__(self, db_path):
            self.db_path = db_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def run(self):
            return {
                "success": ["BHP.AX"],
                "failed": [],
                "forecast_ready": [],
                "forecast_blocked": ["BHP.AX"],
                "macro_failed": [],
                "fundamentals_failed": [],
                "warnings": [
                    "BHP.AX: benchmark dates not aligned."
                ],
                "errors": [],
            }

    def forbidden_forecast(*args, **kwargs):
        raise AssertionError(
            "Forecasting must not run when "
            "the benchmark fails readiness checks."
        )

    monkeypatch.setattr(
        orchestrator,
        "MarketDataAgent",
        FakeMarketDataAgent,
    )

    monkeypatch.setattr(
        orchestrator.ForecastingAgent,
        "run",
        forbidden_forecast,
    )

    result = orchestrator.run_pipeline(
        db_path=":memory:",
        use_llm=False,
    )

    assert result["success"] is False
    assert result["current_stage"] == "error"
    assert result["report_path"] is None
    assert any(
        "readiness" in error.lower()
        for error in result["errors"]
    )