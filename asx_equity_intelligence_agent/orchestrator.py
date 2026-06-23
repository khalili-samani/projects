"""
Orchestrator
============
Responsibility: Wire all five agents together into a single daily pipeline
using LangGraph. Manages state, handles failures gracefully, and schedules
the run automatically each morning.

Design decisions:
- LangGraph StateGraph defines the pipeline as a directed graph of nodes
- Each node is one agent — clean separation of concerns
- Shared AgentState passed between nodes — agents read inputs and write outputs
- Conditional edges route to error handling if a node fails
- Schedule module fires the pipeline at 7am AEST every trading day
- All errors accumulated in state rather than crashing — the run continues
  as far as possible and reports what succeeded and what failed
"""

import logging
import os
import sys
import schedule
import time
from datetime import datetime
from typing import TypedDict, Optional, Any
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END

load_dotenv()

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))

from market_data_agent import MarketDataAgent
from forecasting_agent import ForecastingAgent, StockForecast
from risk_agent import RiskAgent, StockRisk, PortfolioRisk
from strategy_agent import StrategyAgent, StrategyOutput
from report_writer_agent import ReportWriterAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("Orchestrator")


# ---------------------------------------------------------------------------
# Shared State
# Passed between every node in the graph. Each agent reads what it needs
# and writes its outputs back into the state.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # Configuration
    db_path: str
    run_date: str

    # Agent 1 output
    market_data_success: bool
    market_data_tickers: list[str]

    # Agent 2 output
    forecasts: list[Any]            # list[StockForecast]

    # Agent 3 output
    stock_risks: list[Any]          # list[StockRisk]
    portfolio_risk: Optional[Any]   # PortfolioRisk

    # Agent 4 output
    strategy_output: Optional[Any]  # StrategyOutput

    # Agent 5 output
    report_path: Optional[str]

    # Error tracking
    errors: list[str]
    current_stage: str


# ---------------------------------------------------------------------------
# Node functions — one per agent
# Each function receives the full state, does its work, and returns
# a dict of keys to update in the state.
# ---------------------------------------------------------------------------

def run_market_data(state: AgentState) -> dict:
    """Node 1: Fetch and store market data."""
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR | Stage 1: Market Data Agent")
    logger.info("=" * 60)

    try:
        agent = MarketDataAgent(db_path=state["db_path"])
        results = agent.run()

        return {
            "market_data_success": len(results["failed"]) == 0,
            "market_data_tickers": results["success"],
            "current_stage": "market_data_complete",
            "errors": state["errors"] + (
                [f"Market data failed for: {results['failed']}"]
                if results["failed"] else []
            ),
        }
    except Exception as e:
        logger.error(f"Market Data Agent failed: {e}")
        return {
            "market_data_success": False,
            "market_data_tickers": [],
            "current_stage": "market_data_failed",
            "errors": state["errors"] + [f"Market Data Agent error: {str(e)}"],
        }


def run_forecasting(state: AgentState) -> dict:
    """Node 2: Forecast next-day returns."""
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR | Stage 2: Forecasting Agent")
    logger.info("=" * 60)

    try:
        agent = ForecastingAgent(db_path=state["db_path"])
        forecasts = agent.run()
        agent.close()

        if not forecasts:
            return {
                "forecasts": [],
                "current_stage": "forecasting_failed",
                "errors": state["errors"] + ["Forecasting Agent returned no forecasts"],
            }

        return {
            "forecasts": forecasts,
            "current_stage": "forecasting_complete",
            "errors": state["errors"],
        }
    except Exception as e:
        logger.error(f"Forecasting Agent failed: {e}")
        return {
            "forecasts": [],
            "current_stage": "forecasting_failed",
            "errors": state["errors"] + [f"Forecasting Agent error: {str(e)}"],
        }


def run_risk(state: AgentState) -> dict:
    """Node 3: Assess risk."""
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR | Stage 3: Risk Agent")
    logger.info("=" * 60)

    try:
        agent = RiskAgent(db_path=state["db_path"])
        stock_risks, portfolio_risk = agent.run(state["forecasts"])
        agent.close()

        return {
            "stock_risks": stock_risks,
            "portfolio_risk": portfolio_risk,
            "current_stage": "risk_complete",
            "errors": state["errors"],
        }
    except Exception as e:
        logger.error(f"Risk Agent failed: {e}")
        return {
            "stock_risks": [],
            "portfolio_risk": None,
            "current_stage": "risk_failed",
            "errors": state["errors"] + [f"Risk Agent error: {str(e)}"],
        }


def run_strategy(state: AgentState) -> dict:
    """Node 4: Generate trading recommendations."""
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR | Stage 4: Strategy Agent")
    logger.info("=" * 60)

    try:
        agent = StrategyAgent(db_path=state["db_path"])
        strategy_output = agent.run(
            state["forecasts"],
            state["stock_risks"],
            state["portfolio_risk"],
        )

        return {
            "strategy_output": strategy_output,
            "current_stage": "strategy_complete",
            "errors": state["errors"],
        }
    except Exception as e:
        logger.error(f"Strategy Agent failed: {e}")
        return {
            "strategy_output": None,
            "current_stage": "strategy_failed",
            "errors": state["errors"] + [f"Strategy Agent error: {str(e)}"],
        }


def run_report_writer(state: AgentState) -> dict:
    """Node 5: Generate daily briefing."""
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR | Stage 5: Report Writer Agent")
    logger.info("=" * 60)

    try:
        agent = ReportWriterAgent(db_path=state["db_path"])
        report_path = agent.run(state["strategy_output"])
        agent.close()

        return {
            "report_path": report_path,
            "current_stage": "report_complete",
            "errors": state["errors"],
        }
    except Exception as e:
        logger.error(f"Report Writer Agent failed: {e}")
        return {
            "report_path": None,
            "current_stage": "report_failed",
            "errors": state["errors"] + [f"Report Writer Agent error: {str(e)}"],
        }


def handle_error(state: AgentState) -> dict:
    """Error node — logs what went wrong and exits gracefully."""
    logger.error("=" * 60)
    logger.error("ORCHESTRATOR | Pipeline halted due to critical error")
    for error in state["errors"]:
        logger.error(f"  {error}")
    logger.error("=" * 60)
    return {"current_stage": "error"}


# ---------------------------------------------------------------------------
# Conditional routing — decides what to do after each node
# ---------------------------------------------------------------------------

def after_market_data(state: AgentState) -> str:
    """
    After market data: only proceed if we got data for at least some tickers.
    If the whole fetch failed, there's nothing to forecast.
    """
    if not state["market_data_tickers"]:
        logger.error("No market data fetched — halting pipeline")
        return "error"
    return "forecasting"


def after_forecasting(state: AgentState) -> str:
    """After forecasting: only proceed if we have at least some forecasts."""
    if not state["forecasts"]:
        logger.error("No forecasts produced — halting pipeline")
        return "error"
    return "risk"


def after_risk(state: AgentState) -> str:
    """After risk: only proceed if we have risk data."""
    if not state["stock_risks"]:
        logger.error("No risk assessments produced — halting pipeline")
        return "error"
    return "strategy"


def after_strategy(state: AgentState) -> str:
    """After strategy: only proceed if we have recommendations."""
    if not state["strategy_output"]:
        logger.error("No strategy output produced — halting pipeline")
        return "error"
    return "report_writer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Define the LangGraph pipeline.

    Nodes are agents. Edges are the flow between them.
    Conditional edges route to error handling if something goes wrong.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("market_data",   run_market_data)
    graph.add_node("forecasting",   run_forecasting)
    graph.add_node("risk",          run_risk)
    graph.add_node("strategy",      run_strategy)
    graph.add_node("report_writer", run_report_writer)
    graph.add_node("error",         handle_error)

    # Entry point
    graph.set_entry_point("market_data")

    # Conditional edges — each agent decides whether to continue or error
    graph.add_conditional_edges(
        "market_data",
        after_market_data,
        {"forecasting": "forecasting", "error": "error"},
    )
    graph.add_conditional_edges(
        "forecasting",
        after_forecasting,
        {"risk": "risk", "error": "error"},
    )
    graph.add_conditional_edges(
        "risk",
        after_risk,
        {"strategy": "strategy", "error": "error"},
    )
    graph.add_conditional_edges(
        "strategy",
        after_strategy,
        {"report_writer": "report_writer", "error": "error"},
    )

    # Terminal edges
    graph.add_edge("report_writer", END)
    graph.add_edge("error", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

def run_pipeline(db_path: str = "asx_market_data.db") -> dict:
    """
    Execute the full agent pipeline for today's run.
    Returns a summary dict the scheduler can log.
    """
    run_date = datetime.today().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"ORCHESTRATOR | Daily run starting — {run_date}")
    logger.info("=" * 60)

    # Initial state
    initial_state: AgentState = {
        "db_path":               db_path,
        "run_date":              run_date,
        "market_data_success":   False,
        "market_data_tickers":   [],
        "forecasts":             [],
        "stock_risks":           [],
        "portfolio_risk":        None,
        "strategy_output":       None,
        "report_path":           None,
        "errors":                [],
        "current_stage":         "starting",
    }

    # Build and run graph
    pipeline = build_graph()
    final_state = pipeline.invoke(initial_state)

    # Summary
    success = final_state["current_stage"] == "report_complete"
    logger.info("=" * 60)
    logger.info(f"ORCHESTRATOR | Run {'complete' if success else 'failed'}")
    logger.info(f"  Stage reached : {final_state['current_stage']}")
    logger.info(f"  Tickers       : {len(final_state['market_data_tickers'])}")
    logger.info(f"  Forecasts     : {len(final_state['forecasts'])}")
    logger.info(f"  Risks assessed: {len(final_state['stock_risks'])}")
    logger.info(f"  Report        : {final_state['report_path'] or 'not generated'}")
    if final_state["errors"]:
        logger.warning(f"  Errors        : {len(final_state['errors'])}")
        for err in final_state["errors"]:
            logger.warning(f"    - {err}")
    logger.info("=" * 60)

    return {
        "success":      success,
        "run_date":     run_date,
        "report_path":  final_state["report_path"],
        "errors":       final_state["errors"],
    }


# ---------------------------------------------------------------------------
# Scheduler — runs the pipeline every weekday at 7am AEST
# ---------------------------------------------------------------------------

def start_scheduler(db_path: str = "asx_market_data.db"):
    """
    Schedule the pipeline to run at 7:00am every day.
    ASX opens at 10am AEST — 7am gives time for data to settle overnight
    and the report to be ready before the open.

    Run this in a terminal you leave open, or set it up as a Windows
    Task Scheduler / cron job for fully automated daily runs.
    """
    logger.info("Scheduler started — pipeline will run daily at 07:00")
    logger.info("Press Ctrl+C to stop")

    schedule.every().day.at("07:00").do(run_pipeline, db_path=db_path)

    while True:
        schedule.run_pending()
        time.sleep(60)  # check every minute


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASX Trading Agent Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["run", "schedule"],
        default="run",
        help="run: execute once now | schedule: run daily at 07:00"
    )
    parser.add_argument(
        "--db",
        default="asx_market_data.db",
        help="Path to DuckDB database file"
    )
    args = parser.parse_args()

    if args.mode == "schedule":
        start_scheduler(db_path=args.db)
    else:
        result = run_pipeline(db_path=args.db)
        if result["report_path"]:
            print(f"\n{'='*60}")
            print("GENERATED REPORT:")
            print(f"{'='*60}\n")
            with open(result["report_path"], encoding="utf-8") as f:
                print(f.read())