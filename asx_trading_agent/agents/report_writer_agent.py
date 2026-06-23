"""
Report Writer Agent
===================
Responsibility: Take the outputs of all four previous agents and generate
a professional daily trading briefing using an LLM.

Design decisions:
- Google Gemini Flash used as the LLM (free tier, no cost to run daily)
- Swappable LLM provider — see _call_llm() to switch to Claude or others
- Prompt structured with clear sections so the LLM produces consistent output
- Macro context fetched directly from the database for freshness
- Report saved as both .txt and .md for flexibility
- Disclaimer always appended — this is not financial advice
"""

import logging
import os
import duckdb
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forecasting_agent import ForecastingAgent, StockForecast
from risk_agent import RiskAgent, StockRisk, PortfolioRisk
from strategy_agent import StrategyAgent, StrategyOutput, TradeRecommendation

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ReportWriterAgent")


# ---------------------------------------------------------------------------
# LLM provider — swap this function to change providers
# ---------------------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    """
    Call the LLM with a prompt and return the response text.

    Currently configured for Google Gemini Flash (free tier).
    Uses the new google-genai package (replaces deprecated google-generativeai).
    Install with: pip install google-genai

    To switch to Claude, replace the body of this function with:

        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    To switch to Groq (also free):

        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    """
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return response.text
    except ImportError:
        raise ImportError(
            "google-genai not installed. "
            "Run: pip install google-genai"
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise



# ---------------------------------------------------------------------------
# Report Writer Agent
# ---------------------------------------------------------------------------
class ReportWriterAgent:
    """
    Generates a daily ASX trading briefing by prompting an LLM with
    structured context assembled from all four upstream agents.
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
        Fetch the latest macro data points for the report header.
        Gives the LLM real numbers to reference.
        """
        macro_tickers = {
            "^AXJO":    "ASX 200",
            "AUDUSD=X": "AUD/USD",
            "GC=F":     "Gold",
            "CL=F":     "Crude Oil",
        }
        context = {}
        for ticker, name in macro_tickers.items():
            try:
                row = self.conn.execute("""
                    SELECT close, daily_return
                    FROM macro
                    WHERE ticker = ?
                    ORDER BY date DESC
                    LIMIT 1
                """, (ticker,)).df()
                if not row.empty:
                    context[name] = {
                        "close":  round(float(row["close"].iloc[0]), 2),
                        "return": round(float(row["daily_return"].iloc[0]) * 100, 2),
                    }
            except Exception:
                pass
        return context

    def _format_macro_context(self, macro: dict) -> str:
        """Format macro context into a string for the prompt."""
        lines = []
        for name, data in macro.items():
            direction = "up" if data["return"] >= 0 else "down"
            lines.append(
                f"- {name}: {data['close']} ({direction} {abs(data['return'])}%)"
            )
        return "\n".join(lines)

    def _format_recommendations(
        self, recommendations: list[TradeRecommendation]
    ) -> str:
        """Format all recommendations into a structured string for the prompt."""
        lines = []
        for r in recommendations:
            emoji = {"BUY": "↑", "HOLD": "→", "REDUCE": "↓"}.get(r.final_signal, "→")
            lines.append(
                f"- {r.ticker}: {emoji} {r.final_signal} | "
                f"Predicted {r.predicted_return*100:+.2f}% | "
                f"Confidence {r.confidence:.0%} | "
                f"Risk {r.risk_level} | "
                f"Position: {r.position_label}"
            )
        return "\n".join(lines)

    def _format_actionable(
        self, actionable: list[TradeRecommendation]
    ) -> str:
        """Format actionable signals with full rationale for the prompt."""
        if not actionable:
            return "No actionable signals today. All stocks on HOLD."
        lines = []
        for r in actionable:
            lines.append(f"- {r.ticker}: {r.rationale}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        strategy_output: StrategyOutput,
        macro: dict,
        report_date: str,
    ) -> str:
        """
        Assemble the full prompt for the LLM.

        The prompt is structured to give the LLM:
        1. A clear role and output format
        2. All the data it needs to write the report
        3. Explicit instructions on tone and length
        4. A reminder that this is not financial advice
        """
        macro_str      = self._format_macro_context(macro)
        recs_str       = self._format_recommendations(
            strategy_output.recommendations
        )
        actionable_str = self._format_actionable(
            strategy_output.actionable_signals
        )

        prompt = f"""You are a professional financial analyst writing a daily ASX equity briefing for a portfolio management team.

Today's date: {report_date}
Market bias: {strategy_output.market_bias}

---
MACRO DATA (latest available):
{macro_str}

---
PORTFOLIO RISK CONTEXT:
{strategy_output.portfolio_risk.risk_summary}

---
WATCHLIST RECOMMENDATIONS ({len(strategy_output.recommendations)} stocks):
{recs_str}

---
ACTIONABLE SIGNALS:
{actionable_str}

---
INSTRUCTIONS:
Write a professional daily briefing using the data above. Structure it with these exact sections:

1. ASX MORNING BRIEF — {report_date}
2. MACRO OVERVIEW (2-3 sentences summarising market conditions from the macro data)
3. PORTFOLIO RISK SNAPSHOT (2-3 sentences on the risk context)
4. WATCHLIST SUMMARY (brief commentary on the overall signal mix and market bias)
5. ACTIONABLE RECOMMENDATIONS (one paragraph per actionable signal, explaining the rationale clearly)
6. DISCLAIMER

Keep the tone professional but direct. Write for someone who understands financial markets.
The disclaimer must state: "This report is generated by an automated system for educational purposes only and does not constitute financial advice."

If there are no actionable signals, say so clearly and explain what that means for the portfolio today."""

        return prompt

    def _save_report(self, content: str, report_date: str) -> str:
        """Save the report to the output directory and return the file path."""
        filename = f"asx_brief_{report_date.replace('-', '')}.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"  ✓ Report saved to {filepath}")
        return filepath

    def run(self, strategy_output: StrategyOutput) -> str:
        """
        Main entry point. Takes the Strategy Agent's output and returns
        a path to the generated report file.
        """
        logger.info("=" * 60)
        logger.info("Report Writer Agent — starting run")
        logger.info("=" * 60)

        report_date = datetime.today().strftime("%Y-%m-%d")

        # Load macro context
        logger.info("Loading macro context...")
        macro = self._load_macro_context()

        # Build prompt
        logger.info("Building prompt...")
        prompt = self._build_prompt(strategy_output, macro, report_date)
        logger.info(f"  Prompt length: {len(prompt)} characters")

        # Call LLM
        logger.info("Calling LLM...")
        report_content = _call_llm(prompt)
        logger.info("  ✓ LLM response received")

        # Save report
        filepath = self._save_report(report_content, report_date)

        logger.info("=" * 60)
        logger.info(f"Report Writer complete. Report saved to: {filepath}")
        logger.info("=" * 60)

        return filepath

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Run directly for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Check for API key before running
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  GEMINI_API_KEY not found in .env file.")
        print("Get a free key at: aistudio.google.com")
        print("Then add to your .env file: GEMINI_API_KEY=your_key_here\n")
        sys.exit(1)

    # Run Agents 1–4
    print("Running upstream agents...")

    forecasting_agent = ForecastingAgent()
    forecasts = forecasting_agent.run()
    forecasting_agent.close()

    risk_agent = RiskAgent()
    stock_risks, portfolio_risk = risk_agent.run(forecasts)
    risk_agent.close()

    strategy_agent = StrategyAgent()
    strategy_output = strategy_agent.run(forecasts, stock_risks, portfolio_risk)

    # Run Agent 5
    report_agent = ReportWriterAgent()
    filepath = report_agent.run(strategy_output)
    report_agent.close()

    # Print the report to terminal
    print(f"\n{'='*60}")
    print("GENERATED REPORT:")
    print(f"{'='*60}\n")
    with open(filepath, encoding="utf-8") as f:
        print(f.read())