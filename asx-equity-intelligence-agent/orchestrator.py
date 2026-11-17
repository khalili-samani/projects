"""Coordinate the ASX equity research pipeline with LangGraph."""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.forecasting_agent import ForecastingAgent, StockForecast
from agents.market_data_agent import MarketDataAgent
from agents.report_writer_agent import ReportWriterAgent
from agents.risk_agent import PortfolioRisk, RiskAgent, StockRisk
from agents.strategy_agent import StrategyAgent, StrategyOutput


logger = logging.getLogger(__name__)

SYDNEY = ZoneInfo("Australia/Sydney")
SCHEDULE_HOUR = 7
SCHEDULE_MINUTE = 0


class AgentState(TypedDict):
    """Shared state passed between the LangGraph nodes."""

    db_path: str
    report_dir: str
    use_llm: bool
    run_date: str

    market_data_success: bool
    market_data_tickers: list[str]
    forecast_ready_tickers: list[str]

    forecasts: list[StockForecast]
    stock_risks: list[StockRisk]
    portfolio_risk: PortfolioRisk | None
    strategy_output: StrategyOutput | None

    report_path: str | None
    errors: list[str]
    warnings: list[str]
    current_stage: str


def _failure(
    state: AgentState,
    stage: str,
    message: str,
) -> dict:
    """Return a consistent failed-stage update."""

    logger.error("%s: %s", stage, message)

    return {
        "current_stage": f"{stage}_failed",
        "errors": state["errors"] + [message],
    }


def run_market_data(state: AgentState) -> dict:
    """Ingest validated market data and assess forecast readiness."""

    logger.info("Stage 1/5: Market data ingestion")

    try:
        with MarketDataAgent(
            db_path=state["db_path"],
        ) as agent:
            results = agent.run()

        successful = list(results["success"])
        ready = list(results["forecast_ready"])

        errors = list(state["errors"])
        warnings = list(state["warnings"])

        warnings.extend(results.get("warnings", []))
        warnings.extend(results.get("errors", []))

        if results["failed"]:
            warnings.append(
                "Equity ingestion failed for: "
                + ", ".join(results["failed"])
            )

        if results["macro_failed"]:
            warnings.append(
                "Macroeconomic ingestion failed for: "
                + ", ".join(results["macro_failed"])
            )

        if results["fundamentals_failed"]:
            warnings.append(
                "Fundamental metadata unavailable for: "
                + ", ".join(results["fundamentals_failed"])
            )

        if not ready:
            errors.append(
                "No securities passed forecast-readiness "
                "validation. Forecasting has been blocked."
            )

            return {
                "market_data_success": bool(successful),
                "market_data_tickers": successful,
                "forecast_ready_tickers": [],
                "warnings": warnings,
                "errors": errors,
                "current_stage": "market_data_blocked",
            }

        return {
            "market_data_success": bool(successful),
            "market_data_tickers": successful,
            "forecast_ready_tickers": ready,
            "warnings": warnings,
            "errors": errors,
            "current_stage": "market_data_complete",
        }

    except Exception as exc:
        logger.exception("Market data stage failed")

        return {
            "market_data_success": False,
            "market_data_tickers": [],
            "forecast_ready_tickers": [],
            **_failure(
                state,
                "market_data",
                f"Market data stage failed: {exc}",
            ),
        }


def run_forecasting(state: AgentState) -> dict:
    """Generate forecasts only for readiness-approved securities."""

    logger.info("Stage 2/5: Forecasting")

    allowed = state["forecast_ready_tickers"]

    if not allowed:
        return _failure(
            state,
            "forecasting",
            "No securities are approved for forecasting.",
        )

    agent = None

    try:
        agent = ForecastingAgent(
            db_path=state["db_path"],
        )

        forecasts = agent.run(tickers=allowed)

        if not forecasts:
            return _failure(
                state,
                "forecasting",
                "Forecasting produced no valid forecasts.",
            )

        approved = set(allowed)
        unexpected = {
            forecast.ticker
            for forecast in forecasts
            if forecast.ticker not in approved
        }

        if unexpected:
            raise ValueError(
                "Forecasting returned unapproved tickers: "
                + ", ".join(sorted(unexpected))
            )

        expected_dates = {
            forecast.as_of_date
            for forecast in forecasts
        }

        if len(expected_dates) != 1:
            raise ValueError(
                "Forecast origins differ across securities."
            )

        missing = approved - {
            forecast.ticker
            for forecast in forecasts
        }

        warnings = list(state["warnings"])

        if missing:
            warnings.append(
                "No forecast generated for: "
                + ", ".join(sorted(missing))
            )

        return {
            "forecasts": forecasts,
            "warnings": warnings,
            "current_stage": "forecasting_complete",
        }

    except Exception as exc:
        logger.exception("Forecasting stage failed")

        return {
            "forecasts": [],
            **_failure(
                state,
                "forecasting",
                f"Forecasting stage failed: {exc}",
            ),
        }

    finally:
        if agent is not None:
            agent.close()


def run_risk(state: AgentState) -> dict:
    """Assess the risk of successfully forecasted securities."""

    logger.info("Stage 3/5: Quantitative risk assessment")

    agent = None

    try:
        agent = RiskAgent(
            db_path=state["db_path"],
        )

        stock_risks, portfolio_risk = agent.run(
            state["forecasts"]
        )

        if not stock_risks:
            return _failure(
                state,
                "risk",
                "Risk assessment produced no valid results.",
            )

        risk_tickers = {
            risk.ticker
            for risk in stock_risks
        }

        forecast_tickers = {
            forecast.ticker
            for forecast in state["forecasts"]
        }

        missing = forecast_tickers - risk_tickers
        warnings = list(state["warnings"])

        if missing:
            warnings.append(
                "Risk assessment unavailable for: "
                + ", ".join(sorted(missing))
            )

        return {
            "stock_risks": stock_risks,
            "portfolio_risk": portfolio_risk,
            "warnings": warnings,
            "current_stage": "risk_complete",
        }

    except Exception as exc:
        logger.exception("Risk assessment stage failed")

        return {
            "stock_risks": [],
            "portfolio_risk": None,
            **_failure(
                state,
                "risk",
                f"Risk assessment stage failed: {exc}",
            ),
        }

    finally:
        if agent is not None:
            agent.close()


def run_strategy(state: AgentState) -> dict:
    """Produce deterministic risk-adjusted recommendations."""

    logger.info("Stage 4/5: Recommendation strategy")

    try:
        if state["portfolio_risk"] is None:
            raise ValueError(
                "Portfolio risk assessment is unavailable."
            )

        agent = StrategyAgent(
            db_path=state["db_path"],
        )

        output = agent.run(
            state["forecasts"],
            state["stock_risks"],
            state["portfolio_risk"],
        )

        if not output.recommendations:
            raise ValueError(
                "Strategy generated no recommendations."
            )

        return {
            "strategy_output": output,
            "current_stage": "strategy_complete",
        }

    except Exception as exc:
        logger.exception("Strategy stage failed")

        return {
            "strategy_output": None,
            **_failure(
                state,
                "strategy",
                f"Strategy stage failed: {exc}",
            ),
        }


def run_report_writer(state: AgentState) -> dict:
    """Generate a factual report with optional Gemini commentary."""

    logger.info("Stage 5/5: Research report generation")

    agent = None

    try:
        if state["strategy_output"] is None:
            raise ValueError("Strategy output is unavailable.")

        agent = ReportWriterAgent(
            db_path=state["db_path"],
            output_dir=state["report_dir"],
        )

        report_path = agent.run(
            strategy_output=state["strategy_output"],
            use_llm=state["use_llm"],
        )

        if not Path(report_path).is_file():
            raise FileNotFoundError(
                f"Generated report does not exist: {report_path}"
            )

        return {
            "report_path": report_path,
            "current_stage": "report_complete",
        }

    except Exception as exc:
        logger.exception("Report generation failed")

        return {
            "report_path": None,
            **_failure(
                state,
                "report",
                f"Report generation failed: {exc}",
            ),
        }

    finally:
        if agent is not None:
            agent.close()


def handle_error(state: AgentState) -> dict:
    """Log pipeline failures without reporting false success."""

    logger.error(
        "Pipeline stopped at stage: %s",
        state["current_stage"],
    )

    for message in state["errors"]:
        logger.error("%s", message)

    return {
        "current_stage": "error",
    }


def after_market_data(state: AgentState) -> str:
    """Do not proceed with unapproved securities."""

    if (
        state["current_stage"] == "market_data_complete"
        and state["forecast_ready_tickers"]
    ):
        return "forecasting"

    return "error"


def after_forecasting(state: AgentState) -> str:
    """Require at least one successful forecast."""

    if (
        state["current_stage"] == "forecasting_complete"
        and state["forecasts"]
    ):
        return "risk"

    return "error"


def after_risk(state: AgentState) -> str:
    """Require valid security-level and aggregate risk."""

    if (
        state["current_stage"] == "risk_complete"
        and state["stock_risks"]
        and state["portfolio_risk"] is not None
    ):
        return "strategy"

    return "error"


def after_strategy(state: AgentState) -> str:
    """Require at least one recommendation."""

    output = state["strategy_output"]

    if (
        state["current_stage"] == "strategy_complete"
        and output is not None
        and output.recommendations
    ):
        return "report_writer"

    return "error"


def after_report_writer(state: AgentState) -> str:
    """Route report generation failures through error handling."""

    if (
        state["current_stage"] == "report_complete"
        and state["report_path"]
    ):
        return "complete"

    return "error"


def build_graph():
    """Compile the five-stage LangGraph workflow."""

    graph = StateGraph(AgentState)

    graph.add_node("market_data", run_market_data)
    graph.add_node("forecasting", run_forecasting)
    graph.add_node("risk", run_risk)
    graph.add_node("strategy", run_strategy)
    graph.add_node("report_writer", run_report_writer)
    graph.add_node("error", handle_error)

    graph.set_entry_point("market_data")

    graph.add_conditional_edges(
        "market_data",
        after_market_data,
        {
            "forecasting": "forecasting",
            "error": "error",
        },
    )

    graph.add_conditional_edges(
        "forecasting",
        after_forecasting,
        {
            "risk": "risk",
            "error": "error",
        },
    )

    graph.add_conditional_edges(
        "risk",
        after_risk,
        {
            "strategy": "strategy",
            "error": "error",
        },
    )

    graph.add_conditional_edges(
        "strategy",
        after_strategy,
        {
            "report_writer": "report_writer",
            "error": "error",
        },
    )

    graph.add_conditional_edges(
        "report_writer",
        after_report_writer,
        {
            "complete": END,
            "error": "error",
        },
    )

    graph.add_edge("error", END)

    return graph.compile()


def run_pipeline(
    db_path: str = "asx_market_data.db",
    report_dir: str = "reports",
    use_llm: bool = True,
) -> dict:
    """Execute the research pipeline once and return its result."""

    run_date = datetime.now(
        SYDNEY
    ).date().isoformat()

    logger.info(
        "ASX pipeline starting: %s",
        run_date,
    )

    initial_state: AgentState = {
        "db_path": db_path,
        "report_dir": report_dir,
        "use_llm": use_llm,
        "run_date": run_date,
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

    try:
        final_state = build_graph().invoke(initial_state)

    except Exception as exc:
        logger.exception(
            "Uncaught workflow execution error"
        )

        return {
            "success": False,
            "run_date": run_date,
            "current_stage": "error",
            "report_path": None,
            "errors": [
                f"Uncaught workflow error: {exc}"
            ],
            "warnings": [],
        }

    success = (
        final_state["current_stage"] == "report_complete"
        and bool(final_state["report_path"])
        and not final_state["errors"]
    )

    logger.info(
        "Pipeline %s | stage=%s | ingested=%d | "
        "ready=%d | forecasts=%d | risks=%d",
        "completed" if success else "stopped",
        final_state["current_stage"],
        len(final_state["market_data_tickers"]),
        len(final_state["forecast_ready_tickers"]),
        len(final_state["forecasts"]),
        len(final_state["stock_risks"]),
    )

    for warning in final_state["warnings"]:
        logger.warning("%s", warning)

    return {
        "success": success,
        "run_date": run_date,
        "current_stage": final_state["current_stage"],
        "report_path": final_state["report_path"],
        "errors": final_state["errors"],
        "warnings": final_state["warnings"],
    }


def start_scheduler(
    db_path: str = "asx_market_data.db",
    report_dir: str = "reports",
    use_llm: bool = True,
) -> None:
    """Run at 07:00 Sydney time, Monday to Friday.

    This is weekday scheduling, not exchange-holiday scheduling.
    The application must remain running.
    """

    logger.info(
        "Scheduler active: weekdays at 07:00 "
        "Australia/Sydney"
    )

    last_run_date = None

    while True:
        now = datetime.now(SYDNEY)
        today = now.date()

        scheduled_time_reached = (
            (now.hour, now.minute)
            >= (SCHEDULE_HOUR, SCHEDULE_MINUTE)
        )

        if (
            now.weekday() < 5
            and scheduled_time_reached
            and today != last_run_date
        ):
            # Record the attempted date before running to
            # prevent repeated submissions after failure.
            last_run_date = today

            result = run_pipeline(
                db_path=db_path,
                report_dir=report_dir,
                use_llm=use_llm,
            )

            if not result["success"]:
                logger.error(
                    "Scheduled pipeline failed on %s",
                    today,
                )

        time.sleep(30)


def main() -> int:
    """Parse command-line options and execute the workflow."""

    parser = argparse.ArgumentParser(
        description="ASX Equity Intelligence Agent"
    )

    parser.add_argument(
        "--mode",
        choices=("run", "schedule"),
        default="run",
    )

    parser.add_argument(
        "--db",
        default="asx_market_data.db",
        help="DuckDB database path.",
    )

    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for generated Markdown reports.",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Generate a deterministic report without Gemini.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(name)s | "
            "%(levelname)s | %(message)s"
        ),
    )

    load_dotenv()

    if args.mode == "schedule":
        try:
            start_scheduler(
                db_path=args.db,
                report_dir=args.report_dir,
                use_llm=not args.no_llm,
            )

        except KeyboardInterrupt:
            logger.info("Scheduler stopped.")

        return 0

    result = run_pipeline(
        db_path=args.db,
        report_dir=args.report_dir,
        use_llm=not args.no_llm,
    )

    if not result["success"]:
        print("\nPipeline stopped without generating a report.")

        for error in result["errors"]:
            print(f"ERROR: {error}")

        return 1

    report_path = result["report_path"]

    print(f"\nReport saved: {report_path}")

    with open(
        report_path,
        encoding="utf-8",
    ) as report_file:
        print("\n" + report_file.read())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())