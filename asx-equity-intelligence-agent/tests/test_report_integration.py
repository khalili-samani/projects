"""
Offline integration tests for the report-writing agent.

Tests real Markdown file creation, filename collision handling,
deterministic execution without Gemini, and validation failures.

All database files and reports are isolated in pytest temporary
directories.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import agents.report_writer_agent as reporting
from agents.market_data_agent import MarketDatabase
from agents.report_writer_agent import ReportWriterAgent


FORECAST_DATE = "2026-11-17"
TICKER = "BHP.AX"


def make_strategy_output():
    """Construct a consistent, single-security strategy result."""
    recommendation = SimpleNamespace(
        ticker=TICKER,
        as_of_date=FORECAST_DATE,
        forecast_signal="BUY",
        risk_modifier="CONFIRM",
        final_signal="BUY",
        predicted_return=0.0121,
        confidence=0.51,
        risk_level="MEDIUM",
        sharpe_score=0.32,
        position_size=0.075,
        position_label="REDUCED",
        rationale="Synthetic integration test rationale.",
        priority=1,
    )

    portfolio_risk = SimpleNamespace(
        total_stocks=1,
        sector_weights={"Basic Materials": 1.0},
        most_concentrated_sector="Basic Materials",
        concentration_risk="HIGH",
        average_beta=1.20,
        average_volatility=0.28,
        watchlist_var_95=-0.026,
        risk_summary="Synthetic test watchlist.",
    )

    return SimpleNamespace(
        recommendations=[recommendation],
        actionable_signals=[recommendation],
        portfolio_risk=portfolio_risk,
        market_bias="NEUTRAL",
        strategy_summary="One synthetic BUY recommendation.",
    )


@pytest.fixture
def report_environment(tmp_path, monkeypatch):
    """
    Create a report writer with an isolated DuckDB database.

    Macro context is mocked because these tests focus on report
    creation and persistence rather than market-data ingestion.
    """
    database_path = tmp_path / "test_market_data.db"
    report_directory = tmp_path / "reports"

    with MarketDatabase(str(database_path)):
        pass

    writer = ReportWriterAgent(
        db_path=str(database_path),
        output_dir=str(report_directory),
    )

    monkeypatch.setattr(
        writer,
        "_load_macro_context",
        lambda *args, **kwargs: {},
    )

    try:
        yield {
            "writer": writer,
            "directory": report_directory,
            "database": database_path,
        }
    finally:
        writer.close()


def test_report_run_creates_markdown_file(
    report_environment,
):
    """The real run method must create a readable Markdown report."""
    writer = report_environment["writer"]

    report_path = Path(
        writer.run(
            strategy_output=make_strategy_output(),
            use_llm=False,
        )
    )

    assert report_path.is_file()
    assert report_path.suffix == ".md"

    content = report_path.read_text(encoding="utf-8")

    assert TICKER in content
    assert FORECAST_DATE in content
    assert "+1.21%" in content
    assert "BUY" in content
    assert "DISCLAIMER" in content


def test_repeated_run_preserves_original_report(
    report_environment,
):
    """Repeated generation must preserve previously written reports."""
    writer = report_environment["writer"]
    output = make_strategy_output()

    first_path = Path(
        writer.run(
            strategy_output=output,
            use_llm=False,
        )
    )

    original_bytes = first_path.read_bytes()

    second_path = Path(
        writer.run(
            strategy_output=output,
            use_llm=False,
        )
    )

    assert first_path != second_path

    assert first_path.is_file()
    assert second_path.is_file()

    assert first_path.read_bytes() == original_bytes

    assert first_path.name.startswith(
        "asx_brief_20261117"
    )
    assert second_path.name.startswith(
        "asx_brief_20261117"
    )


def test_deterministic_run_never_requests_gemini(
    report_environment,
    monkeypatch,
):
    """
    Disabling Gemini commentary must prevent the network boundary
    from being invoked, while still producing the factual report.
    """
    writer = report_environment["writer"]

    def forbidden_llm_call(*args, **kwargs):
        raise AssertionError(
            "Gemini must not be called when use_llm=False."
        )

    monkeypatch.setattr(
        reporting,
        "_call_llm",
        forbidden_llm_call,
    )

    report_path = Path(
        writer.run(
            strategy_output=make_strategy_output(),
            use_llm=False,
        )
    )

    assert report_path.is_file()

    content = report_path.read_text(encoding="utf-8")

    assert TICKER in content
    assert "DISCLAIMER" in content


def test_invalid_strategy_does_not_write_report(
    report_environment,
):
    """Inconsistent recommendation dates must prevent file creation."""
    writer = report_environment["writer"]
    output = make_strategy_output()

    second = SimpleNamespace(
        **vars(output.recommendations[0])
    )

    second.ticker = "RIO.AX"
    second.as_of_date = "2026-11-16"

    output.recommendations.append(second)

    with pytest.raises(ValueError):
        writer.run(
            strategy_output=output,
            use_llm=False,
        )

    assert not list(
        report_environment["directory"].glob("*.md")
    )