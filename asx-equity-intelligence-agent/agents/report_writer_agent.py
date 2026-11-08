"""
report_writer_agent.py
----------------------
Generates a daily ASX market briefing using structured outputs from the
forecasting, risk and strategy agents.

The LLM writes the report, but it does not make the trading decision.
All forecasts, risk metrics and recommendations are produced upstream.
"""

import logging
import os
import sys
from datetime import datetime

import duckdb
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forecasting_agent import ForecastingAgent
from risk_agent import RiskAgent
from strategy_agent import StrategyAgent, StrategyOutput, TradeRecommendation

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ReportWriterAgent")


def _call_llm(prompt: str) -> str:
    """
    Call Gemini 2.5 Flash Lite and return the generated report text.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    return response.text


class ReportWriterAgent:
    """
    Generates a daily ASX briefing from structured strategy output.

    The model is used only for communication and summarisation. It is not
    responsible for producing forecasts, risk calculations or trading signals.
    """

    def __init__(
        self,
        db_path: str = "asx_market_data.db",
        output_dir: str = "reports",
    ):
        self.db_path = db_path
        self.output_dir = output_dir
        self.conn = duckdb.connect(db_path)
        os.makedirs(output_dir, exist_ok=True)

    def _load_macro_context(self) -> dict:
        """
        Fetch latest macro data points for the report.

        Returns a dictionary containing exact numerical values supplied to the
        LLM. This reduces the risk of the model inventing market movements.
        """
        macro_tickers = {
            "^AXJO": "ASX 200",
            "AUDUSD=X": "AUD/USD",
            "GC=F": "Gold",
            "CL=F": "Crude Oil",
        }

        context = {}

        for ticker, name in macro_tickers.items():
            try:
                row = self.conn.execute(
                    """
                    SELECT close, daily_return
                    FROM macro
                    WHERE ticker = ?
                    ORDER BY date DESC
                    LIMIT 1
                    """,
                    (ticker,),
                ).df()

                if not row.empty:
                    context[name] = {
                        "close": round(float(row["close"].iloc[0]), 2),
                        "return_pct": round(
                            float(row["daily_return"].iloc[0]) * 100,
                            2,
                        ),
                    }

            except Exception as exc:
                logger.warning("Could not load macro context for %s: %s", ticker, exc)

        return context

    def _format_macro_context(self, macro: dict) -> str:
        """Format macro context using exact values from the database."""
        if not macro:
            return "- Macro context unavailable for this run."

        lines = []

        for name, data in macro.items():
            return_pct = data["return_pct"]
            direction = "up" if return_pct >= 0 else "down"

            lines.append(
                f"- {name}: close={data['close']}, "
                f"daily_return={return_pct:+.2f}% ({direction})"
            )

        return "\n".join(lines)

    def _format_recommendations(
        self,
        recommendations: list[TradeRecommendation],
    ) -> str:
        """Format all watchlist recommendations for the prompt."""
        lines = []

        for rec in recommendations:
            lines.append(
                f"- {rec.ticker}: final_signal={rec.final_signal}, "
                f"predicted_return={rec.predicted_return * 100:+.2f}%, "
                f"confidence={rec.confidence:.0%}, "
                f"risk_level={rec.risk_level}, "
                f"position={rec.position_label}"
            )

        return "\n".join(lines)

    def _format_actionable(
        self,
        actionable: list[TradeRecommendation],
    ) -> str:
        """Format actionable BUY and REDUCE signals with rationale."""
        if not actionable:
            return "- No actionable BUY or REDUCE signals today."

        lines = []

        for rec in actionable:
            lines.append(f"- {rec.ticker}: {rec.rationale}")

        return "\n".join(lines)

    def _build_prompt(
        self,
        strategy_output: StrategyOutput,
        macro: dict,
        report_date: str,
    ) -> str:
        """
        Assemble the LLM prompt.

        The prompt explicitly prevents the LLM from inventing market movements
        or changing recommendations.
        """
        macro_str = self._format_macro_context(macro)
        recs_str = self._format_recommendations(
            strategy_output.recommendations
        )
        actionable_str = self._format_actionable(
            strategy_output.actionable_signals
        )

        return f"""You are a professional financial analyst writing a daily ASX equity briefing.

IMPORTANT RULES:
1. Use ONLY the information supplied below.
2. Do NOT invent market movements, prices, returns, explanations or statistics.
3. If information is not supplied, do not mention it.
4. Do not infer macro trends beyond the provided values.
5. Quote numerical values exactly as provided.
6. Do not round or modify percentages.
7. All BUY, HOLD and REDUCE recommendations must come only from the supplied strategy output.
8. Do not change any recommendation, confidence value, risk level or position label.
9. The LLM is responsible for communication only, not decision-making.

DATE:
{report_date}

MARKET BIAS:
{strategy_output.market_bias}

MACRO DATA:
{macro_str}

PORTFOLIO RISK CONTEXT:
{strategy_output.portfolio_risk.risk_summary}

WATCHLIST RECOMMENDATIONS:
{recs_str}

ACTIONABLE SIGNALS:
{actionable_str}

Write a professional Markdown report using exactly these sections:

1. ASX MORNING BRIEF — {report_date}
2. MACRO OVERVIEW
3. PORTFOLIO RISK SNAPSHOT
4. WATCHLIST SUMMARY
5. ACTIONABLE RECOMMENDATIONS
6. DISCLAIMER

The disclaimer must state exactly:
"This report is generated by an automated system for educational purposes only and does not constitute financial advice."

Keep the tone professional, concise and factual."""

    def _save_report(self, content: str, report_date: str) -> str:
        """Save the report to the output directory and return the file path."""
        filename = f"asx_brief_{report_date.replace('-', '')}.md"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        logger.info("  ✓ Report saved to %s", filepath)
        return filepath

    def run(self, strategy_output: StrategyOutput) -> str:
        """
        Generate the daily briefing and return the saved report path.
        """
        if strategy_output is None:
            raise ValueError("strategy_output cannot be None.")

        logger.info("=" * 60)
        logger.info("Report Writer Agent — starting run")
        logger.info("=" * 60)

        report_date = datetime.today().strftime("%Y-%m-%d")

        logger.info("Loading macro context.")
        macro = self._load_macro_context()

        logger.info("Building prompt.")
        prompt = self._build_prompt(strategy_output, macro, report_date)
        logger.info("  Prompt length: %d characters", len(prompt))

        logger.info("Calling LLM.")
        report_content = _call_llm(prompt)
        logger.info("  ✓ LLM response received")

        filepath = self._save_report(report_content, report_date)

        logger.info("=" * 60)
        logger.info("Report Writer complete. Report saved to: %s", filepath)
        logger.info("=" * 60)

        return filepath

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("\nGEMINI_API_KEY not found in .env file.")
        print("Add it to your .env file: GEMINI_API_KEY=your_key_here\n")
        sys.exit(1)

    print("Running upstream agents.")

    forecasting_agent = ForecastingAgent()
    forecasts = forecasting_agent.run()
    forecasting_agent.close()

    risk_agent = RiskAgent()
    stock_risks, portfolio_risk = risk_agent.run(forecasts)
    risk_agent.close()

    strategy_agent = StrategyAgent()
    strategy_output = strategy_agent.run(forecasts, stock_risks, portfolio_risk)

    report_agent = ReportWriterAgent()
    filepath = report_agent.run(strategy_output)
    report_agent.close()

    print(f"\n{'=' * 60}")
    print("GENERATED REPORT:")
    print(f"{'=' * 60}\n")

    with open(filepath, encoding="utf-8") as file:
        print(file.read())