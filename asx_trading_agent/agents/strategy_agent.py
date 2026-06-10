"""
Trading Strategy Agent
======================
Responsibility: Combine forecast signals and risk assessments into final
actionable trading recommendations with position sizing and rationale.

Design decisions:
- Signal resolution table maps forecast + risk modifier → final signal
- Volatility-scaled position sizing — higher volatility stocks get smaller
  recommended allocations to equalise risk contribution across the portfolio
- Sharpe score used to rank and prioritise recommendations
- Plain-English rationale generated for each recommendation so the
  Report Writer Agent has structured context to work with
- Conservative by default — when in doubt, output HOLD rather than
  a marginal BUY or REDUCE
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forecasting_agent import ForecastingAgent, StockForecast
from risk_agent import RiskAgent, StockRisk, PortfolioRisk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("StrategyAgent")


# ---------------------------------------------------------------------------
# Signal resolution table
# Forecast signal + risk modifier → final signal
# ---------------------------------------------------------------------------
SIGNAL_RESOLUTION = {
    ("BUY",    "CONFIRM"):   "BUY",
    ("BUY",    "DOWNGRADE"): "HOLD",
    ("BUY",    "BLOCK"):     "HOLD",
    ("HOLD",   "CONFIRM"):   "HOLD",
    ("HOLD",   "DOWNGRADE"): "HOLD",
    ("HOLD",   "BLOCK"):     "HOLD",
    ("REDUCE", "CONFIRM"):   "REDUCE",
    ("REDUCE", "DOWNGRADE"): "HOLD",
    ("REDUCE", "BLOCK"):     "HOLD",
}

# Emoji indicators for the report
SIGNAL_EMOJI = {
    "BUY":    "↑",
    "HOLD":   "→",
    "REDUCE": "↓",
}


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------
@dataclass
class TradeRecommendation:
    ticker: str
    forecast_date: str
    forecast_signal: str            # raw signal from Agent 2
    risk_modifier: str              # CONFIRM / DOWNGRADE / BLOCK from Agent 3
    final_signal: str               # resolved final signal
    predicted_return: float
    confidence: float
    risk_level: str
    sharpe_score: float
    position_size: float            # recommended allocation (0.0 to 1.0)
    position_label: str             # FULL / REDUCED / MINIMAL / NONE
    rationale: str                  # plain English — passed to Report Writer
    priority: int                   # 1 = highest priority recommendation


@dataclass
class StrategyOutput:
    recommendations: list[TradeRecommendation]
    portfolio_risk: PortfolioRisk
    actionable_signals: list[TradeRecommendation]   # BUY or REDUCE only
    market_bias: str                                # BULLISH / NEUTRAL / BEARISH
    strategy_summary: str                           # plain English for Report Writer


# ---------------------------------------------------------------------------
# Trading Strategy Agent
# ---------------------------------------------------------------------------
class StrategyAgent:
    """
    Combines forecast and risk outputs into final trading recommendations.

    This agent is where the system's decision-making logic lives. It applies
    the signal resolution table, sizes positions, ranks opportunities, and
    assembles rationale strings for the Report Writer.
    """

    def __init__(
        self,
        db_path: str = "asx_market_data.db",
        base_position_size: float = 0.10,   # 10% per stock in equal-weight portfolio
        min_confidence: float = 0.30,        # ignore forecasts below this threshold
    ):
        self.db_path = db_path
        self.base_position_size = base_position_size
        self.min_confidence = min_confidence

    # -----------------------------------------------------------------------
    # Signal resolution
    # -----------------------------------------------------------------------
    def _resolve_signal(
        self, forecast_signal: str, risk_modifier: str
    ) -> str:
        """
        Apply the signal resolution table.
        Falls back to HOLD for any unknown combination.
        """
        return SIGNAL_RESOLUTION.get((forecast_signal, risk_modifier), "HOLD")

    # -----------------------------------------------------------------------
    # Position sizing
    # -----------------------------------------------------------------------
    def _calculate_position_size(
        self,
        final_signal: str,
        confidence: float,
        risk_level: str,
        sharpe_score: float,
    ) -> tuple[float, str]:
        """
        Volatility-scaled position sizing.

        The idea: rather than putting the same dollar amount into every stock,
        we size positions so that each stock contributes roughly equal risk
        to the portfolio. High-volatility stocks get smaller allocations.

        Returns a tuple of (size as float 0–1, label string).

        HOLD signals always get 0 — this system is about adjustments,
        not building a portfolio from scratch.
        """
        if final_signal == "HOLD":
            return 0.0, "NONE"

        # Start from the base allocation
        size = self.base_position_size

        # Scale up for high confidence
        if confidence >= 0.60:
            size *= 1.5
        elif confidence >= 0.45:
            size *= 1.0
        else:
            size *= 0.5     # low confidence → smaller position

        # Scale down for high risk
        if risk_level == "HIGH":
            size *= 0.5
        elif risk_level == "MEDIUM":
            size *= 0.75

        # Cap at base position size for safety
        size = min(size, self.base_position_size * 1.5)
        size = round(size, 3)

        # Label
        if size >= self.base_position_size * 1.2:
            label = "FULL"
        elif size >= self.base_position_size * 0.7:
            label = "REDUCED"
        else:
            label = "MINIMAL"

        return size, label

    # -----------------------------------------------------------------------
    # Rationale generation
    # -----------------------------------------------------------------------
    def _generate_rationale(
        self,
        forecast: StockForecast,
        risk: StockRisk,
        final_signal: str,
        position_label: str,
    ) -> str:
        """
        Assemble a plain-English rationale for each recommendation.
        This is passed directly to the Report Writer Agent as context.
        """
        emoji = SIGNAL_EMOJI.get(final_signal, "→")
        ret_str = f"{forecast.predicted_return * 100:+.2f}%"
        conf_str = f"{forecast.confidence:.0%}"
        vol_str = f"{risk.volatility_annual:.1%}"
        var_str = f"{risk.var_95:.2%}"

        # Was the signal changed by the risk modifier?
        signal_changed = forecast.signal != final_signal
        modifier_note = ""
        if signal_changed:
            modifier_note = (
                f" Original {forecast.signal} signal downgraded to {final_signal} "
                f"due to {risk.risk_level.lower()} risk profile."
            )

        # Top feature context
        top_feat = list(forecast.feature_importance.keys())[0]
        feature_labels = {
            "return_5d":       "short-term momentum",
            "return_10d":      "medium-term momentum",
            "return_20d":      "trend direction",
            "volatility_20d":  "recent volatility",
            "rsi_14":          "RSI momentum signal",
            "volume_ratio":    "unusual trading volume",
            "dist_52w_high":   "proximity to 52-week high",
            "dist_52w_low":    "proximity to 52-week low",
            "ma_crossover":    "moving average crossover",
            "asx200_return":   "broad market momentum",
            "asx200_return_5d":"5-day market trend",
        }
        feature_desc = feature_labels.get(top_feat, top_feat)

        rationale = (
            f"{emoji} {final_signal} | Predicted return: {ret_str} "
            f"(confidence: {conf_str}). "
            f"Primary driver: {feature_desc}. "
            f"Annualised volatility: {vol_str}, 1-day VaR (95%): {var_str}. "
            f"Risk level: {risk.risk_level}. "
            f"Position sizing: {position_label}.{modifier_note}"
        )
        return rationale

    # -----------------------------------------------------------------------
    # Market bias
    # -----------------------------------------------------------------------
    def _derive_market_bias(
        self, recommendations: list[TradeRecommendation]
    ) -> str:
        """
        Derive an overall market bias from the balance of signals.
        Used by the Report Writer to set the tone of the briefing.
        """
        buys    = sum(1 for r in recommendations if r.final_signal == "BUY")
        reduces = sum(1 for r in recommendations if r.final_signal == "REDUCE")
        total   = len(recommendations)

        if total == 0:
            return "NEUTRAL"

        buy_ratio    = buys / total
        reduce_ratio = reduces / total

        if buy_ratio > 0.40:
            return "BULLISH"
        elif reduce_ratio > 0.40:
            return "BEARISH"
        else:
            return "NEUTRAL"

    # -----------------------------------------------------------------------
    # Main processing
    # -----------------------------------------------------------------------
    def _process_pair(
        self,
        forecast: StockForecast,
        risk: StockRisk,
        priority: int,
    ) -> TradeRecommendation:
        """Process a single forecast + risk pair into a recommendation."""

        # Filter out very low confidence forecasts
        if forecast.confidence < self.min_confidence:
            logger.info(
                f"  {forecast.ticker}: confidence {forecast.confidence:.0%} "
                f"below threshold — defaulting to HOLD"
            )
            final_signal = "HOLD"
            risk_modifier = risk.recommendation_modifier
        else:
            final_signal = self._resolve_signal(
                forecast.signal, risk.recommendation_modifier
            )
            risk_modifier = risk.recommendation_modifier

        position_size, position_label = self._calculate_position_size(
            final_signal,
            forecast.confidence,
            risk.risk_level,
            risk.sharpe_score,
        )

        rationale = self._generate_rationale(
            forecast, risk, final_signal, position_label
        )

        logger.info(
            f"  ✓ {forecast.ticker}: {forecast.signal} + {risk_modifier} "
            f"→ {final_signal} | size={position_label}"
        )

        return TradeRecommendation(
            ticker=forecast.ticker,
            forecast_date=forecast.forecast_date,
            forecast_signal=forecast.signal,
            risk_modifier=risk_modifier,
            final_signal=final_signal,
            predicted_return=forecast.predicted_return,
            confidence=forecast.confidence,
            risk_level=risk.risk_level,
            sharpe_score=risk.sharpe_score,
            position_size=position_size,
            position_label=position_label,
            rationale=rationale,
            priority=priority,
        )

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def run(
        self,
        forecasts: list[StockForecast],
        stock_risks: list[StockRisk],
        portfolio_risk: PortfolioRisk,
    ) -> StrategyOutput:
        """
        Main entry point. Takes outputs from Agents 2 and 3 and returns
        a full StrategyOutput with ranked recommendations.
        """
        logger.info("=" * 60)
        logger.info("Strategy Agent — starting run")
        logger.info(f"Processing {len(forecasts)} forecasts")
        logger.info("=" * 60)

        # Build lookup maps
        forecast_map = {f.ticker: f for f in forecasts}
        risk_map     = {r.ticker: r for r in stock_risks}

        # Find tickers present in both
        common_tickers = set(forecast_map.keys()) & set(risk_map.keys())
        logger.info(f"Tickers with both forecast and risk data: {len(common_tickers)}")

        # Sort by Sharpe score descending — highest opportunity first
        sorted_tickers = sorted(
            common_tickers,
            key=lambda t: risk_map[t].sharpe_score,
            reverse=True,
        )

        recommendations = []
        for priority, ticker in enumerate(sorted_tickers, start=1):
            rec = self._process_pair(
                forecast_map[ticker],
                risk_map[ticker],
                priority,
            )
            recommendations.append(rec)

        # Actionable signals only (BUY or REDUCE)
        actionable = [r for r in recommendations if r.final_signal != "HOLD"]

        # Market bias
        market_bias = self._derive_market_bias(recommendations)

        # Strategy summary for Report Writer
        buy_count    = sum(1 for r in recommendations if r.final_signal == "BUY")
        reduce_count = sum(1 for r in recommendations if r.final_signal == "REDUCE")
        hold_count   = sum(1 for r in recommendations if r.final_signal == "HOLD")

        strategy_summary = (
            f"Market bias: {market_bias}. "
            f"Signal breakdown: {buy_count} BUY, {hold_count} HOLD, {reduce_count} REDUCE "
            f"across {len(recommendations)} stocks. "
            f"{portfolio_risk.risk_summary}"
        )

        output = StrategyOutput(
            recommendations=recommendations,
            portfolio_risk=portfolio_risk,
            actionable_signals=actionable,
            market_bias=market_bias,
            strategy_summary=strategy_summary,
        )

        logger.info("=" * 60)
        logger.info(f"Strategy complete.")
        logger.info(f"  BUY: {buy_count} | HOLD: {hold_count} | REDUCE: {reduce_count}")
        logger.info(f"  Market bias: {market_bias}")
        logger.info("=" * 60)

        return output

    def get_recommendation_summary(
        self, output: StrategyOutput
    ) -> pd.DataFrame:
        """Clean summary DataFrame for display and logging."""
        rows = []
        for r in output.recommendations:
            rows.append({
                "ticker":          r.ticker,
                "forecast_signal": r.forecast_signal,
                "modifier":        r.risk_modifier,
                "final_signal":    r.final_signal,
                "predicted_ret":   f"{r.predicted_return*100:+.2f}%",
                "confidence":      f"{r.confidence:.0%}",
                "risk_level":      r.risk_level,
                "position":        r.position_label,
                "sharpe_score":    f"{r.sharpe_score:.3f}",
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run directly for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Run Agent 2
    forecasting_agent = ForecastingAgent()
    forecasts = forecasting_agent.run()
    forecasting_agent.close()

    # Run Agent 3
    risk_agent = RiskAgent()
    stock_risks, portfolio_risk = risk_agent.run(forecasts)
    risk_agent.close()

    # Run Agent 4
    strategy_agent = StrategyAgent()
    output = strategy_agent.run(forecasts, stock_risks, portfolio_risk)

    print("\n📋 Final Recommendations:")
    print(strategy_agent.get_recommendation_summary(output).to_string(index=False))

    print(f"\n🎯 Market Bias: {output.market_bias}")
    print(f"\n📝 Strategy Summary:")
    print(f"  {output.strategy_summary}")

    if output.actionable_signals:
        print(f"\n⚡ Actionable Signals ({len(output.actionable_signals)}):")
        for rec in output.actionable_signals:
            print(f"  {rec.ticker}: {rec.rationale}")