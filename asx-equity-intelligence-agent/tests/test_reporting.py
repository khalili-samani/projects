"""Regression tests for deterministic report construction."""

from types import SimpleNamespace

import pytest

from agents.report_writer_agent import ReportWriterAgent


def make_strategy_output():
    recommendation = SimpleNamespace(
        ticker="BHP.AX",
        as_of_date="2026-11-17",
        forecast_signal="BUY",
        risk_modifier="CONFIRM",
        final_signal="BUY",
        predicted_return=0.0121,
        confidence=0.51,
        risk_level="MEDIUM",
        position_size=0.075,
        position_label="REDUCED",
        rationale="Deterministic test rationale.",
        priority=1,
    )

    portfolio_risk = SimpleNamespace(
        total_stocks=1,
        average_beta=1.20,
        average_volatility=0.28,
        watchlist_var_95=-0.026,
        most_concentrated_sector="Basic Materials",
        concentration_risk="HIGH",
        sector_weights={"Basic Materials": 1.0},
    )

    return SimpleNamespace(
        recommendations=[recommendation],
        actionable_signals=[recommendation],
        portfolio_risk=portfolio_risk,
        market_bias="NEUTRAL",
    )


def make_writer_without_database():
    """Construct a writer for pure formatting tests only."""

    return ReportWriterAgent.__new__(
        ReportWriterAgent
    )


def test_forecast_origin_is_not_report_generation_date():
    output = make_strategy_output()
    writer = make_writer_without_database()

    assert writer._get_as_of_date(output) == "2026-11-17"

    content = writer._build_report(
        output=output,
        macro={},
        as_of_date="2026-11-17",
        generated_date="2026-11-18",
        commentary=None,
    )

    assert "Forecast origin:** 2026-11-17" in content
    assert "Report generated:** 2026-11-18" in content


def test_recommendations_are_rendered_deterministically():
    output = make_strategy_output()
    writer = make_writer_without_database()

    writer._validate_output(output)

    content = writer._build_report(
        output=output,
        macro={},
        as_of_date="2026-11-17",
        generated_date="2026-11-18",
        commentary=None,
    )

    assert "BHP.AX" in content
    assert "+1.21%" in content
    assert "**BUY**" in content
    assert "7.5%" in content
    assert "1 BUY, 0 HOLD, 0 REDUCE" in content


def test_macro_observation_date_is_displayed():
    writer = make_writer_without_database()

    content = writer._format_macro_context(
        {
            "ASX 200": {
                "ticker": "^AXJO",
                "date": "2026-11-16",
                "close": 9044.7,
                "daily_return": -0.004129,
            }
        }
    )

    assert "2026-11-16" in content
    assert "9044.7000" in content
    assert "-0.41%" in content


def test_inconsistent_recommendation_dates_are_rejected():
    output = make_strategy_output()

    second = SimpleNamespace(
        **vars(output.recommendations[0])
    )
    second.ticker = "RIO.AX"
    second.as_of_date = "2026-11-16"

    output.recommendations.append(second)

    writer = make_writer_without_database()

    with pytest.raises(ValueError, match="inconsistent"):
        writer._get_as_of_date(output)


def test_duplicate_tickers_are_rejected():
    output = make_strategy_output()

    duplicate = SimpleNamespace(
        **vars(output.recommendations[0])
    )
    output.recommendations.append(duplicate)

    output.portfolio_risk.total_stocks = 2

    writer = make_writer_without_database()

    with pytest.raises(ValueError, match="Duplicate"):
        writer._validate_output(output)


def test_llm_is_not_called_in_deterministic_mode(monkeypatch):
    import agents.report_writer_agent as reporting

    def forbidden_call(prompt):
        raise AssertionError(
            "Gemini should not be called."
        )

    monkeypatch.setattr(
        reporting,
        "_call_llm",
        forbidden_call,
    )

    writer = make_writer_without_database()

    commentary = writer._optional_commentary(
        make_strategy_output(),
        use_llm=False,
    )

    assert commentary is None


def test_unsafe_llm_commentary_is_rejected():
    writer = make_writer_without_database()

    assert not writer._validate_commentary(
        "Buy BHP.AX for a 12% return."
    )

    assert not writer._validate_commentary(
        "Visit https://example.com for advice."
    )


def test_missing_commentary_does_not_remove_disclaimer():
    output = make_strategy_output()
    writer = make_writer_without_database()

    content = writer._build_report(
        output=output,
        macro={},
        as_of_date="2026-11-17",
        generated_date="2026-11-18",
        commentary=None,
    )

    assert "## DISCLAIMER" in content
    assert "does not constitute financial advice" in content
    assert "AI-ASSISTED METHODOLOGY COMMENTARY" not in content