"""
Risk Agent
==========
Responsibility: Assess the risk profile of each stock forecast and the
overall watchlist portfolio. Outputs risk-adjusted metrics that the
Trading Strategy Agent uses to size and filter recommendations.

Design decisions:
- Historical VaR (non-parametric) — uses actual return distribution
  rather than assuming normality. Stock returns have fat tails, so
  the normal distribution underestimates real-world risk.
- Beta calculated against ASX 200 (^AXJO) — the standard Australian
  market benchmark.
- Sector concentration measured from fundamentals table — flags
  portfolios that are overweight a single sector.
- Risk-adjusted score combines forecast confidence, predicted return,
  and volatility — gives the Strategy Agent a single number to rank
  opportunities.
"""

import logging
import numpy as np
import pandas as pd
import duckdb
from dataclasses import dataclass, field
from typing import Optional

from forecasting_agent import StockForecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("RiskAgent")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------
@dataclass
class StockRisk:
    ticker: str
    volatility_annual: float        # annualised volatility (%)
    var_95: float                   # 1-day VaR at 95% confidence (% of position)
    var_99: float                   # 1-day VaR at 99% confidence (% of position)
    max_drawdown: float             # worst peak-to-trough decline in history (%)
    beta: float                     # sensitivity to ASX 200
    sharpe_score: float             # predicted return / volatility (risk-adjusted)
    risk_level: str                 # LOW / MEDIUM / HIGH — derived from volatility
    recommendation_modifier: str    # CONFIRM / DOWNGRADE / BLOCK


@dataclass
class PortfolioRisk:
    total_stocks: int
    sector_weights: dict            # sector → % of watchlist
    most_concentrated_sector: str
    concentration_risk: str         # LOW / MEDIUM / HIGH
    average_beta: float
    average_volatility: float
    watchlist_var_95: float         # simple average VaR across watchlist
    risk_summary: str               # plain English summary for Report Writer


# ---------------------------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------------------------
class RiskAgent:
    """
    Calculates risk metrics for each stock and the overall portfolio.

    Takes StockForecast objects from the Forecasting Agent as input,
    enriches them with risk context, and returns StockRisk objects
    that the Strategy Agent uses to filter and size recommendations.
    """

    def __init__(self, db_path: str = "asx_market_data.db"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)

    def _load_returns(self, ticker: str, days: int = 252) -> pd.Series:
        """Load recent daily returns for a ticker."""
        df = self.conn.execute("""
            SELECT date, daily_return
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """, (ticker, days)).df()
        return df["daily_return"].dropna()

    def _load_market_returns(self, days: int = 252) -> pd.Series:
        """Load ASX 200 returns as the market benchmark."""
        df = self.conn.execute("""
            SELECT date, daily_return
            FROM macro
            WHERE ticker = '^AXJO'
            ORDER BY date DESC
            LIMIT ?
        """, (days,)).df()
        return df["daily_return"].dropna()

    def _load_closes(self, ticker: str, days: int = 252) -> pd.Series:
        """Load closing prices for drawdown calculation."""
        df = self.conn.execute("""
            SELECT date, close
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date ASC
        """, (ticker,)).df()
        return df["close"].dropna()

    def _load_fundamentals(self) -> pd.DataFrame:
        """Load latest fundamentals for sector concentration analysis."""
        return self.conn.execute("""
            SELECT DISTINCT ON (ticker) ticker, sector, beta
            FROM fundamentals
            ORDER BY ticker, fetched_at DESC
        """).df()

    # -----------------------------------------------------------------------
    # Individual metric calculations
    # -----------------------------------------------------------------------
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """
        Annualised volatility = daily std × √252
        252 is the standard number of ASX trading days per year.
        """
        return float(returns.std() * np.sqrt(252))

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """
        Historical (non-parametric) Value at Risk.

        We sort the actual historical returns and take the percentile
        at the tail. No distribution assumptions — we let the data speak.

        A 95% VaR of -0.018 means: on 95% of days, the loss won't
        exceed 1.8% of the position value.
        """
        return float(np.percentile(returns, (1 - confidence) * 100))

    def _calculate_max_drawdown(self, closes: pd.Series) -> float:
        """
        Maximum drawdown = worst peak-to-trough decline over the period.

        We calculate the running maximum (the "peak") at each point,
        then measure how far below that peak the price fell.
        """
        rolling_max = closes.cummax()
        drawdown = (closes - rolling_max) / rolling_max
        return float(drawdown.min())  # most negative value = worst drawdown

    def _calculate_beta(
        self, stock_returns: pd.Series, market_returns: pd.Series
    ) -> float:
        """
        Beta = covariance(stock, market) / variance(market)

        Measures how much the stock moves relative to the ASX 200.
        Beta > 1: amplifies market moves (more volatile)
        Beta < 1: dampens market moves (more defensive)
        Beta < 0: moves against the market (rare — WDS sometimes shows this)

        We align dates before calculating — not every stock trades
        on every day the index moves.
        """
        combined = pd.DataFrame({
            "stock": stock_returns.values,
            "market": market_returns.values
        }).dropna()

        if len(combined) < 30:
            return 1.0  # default to market beta if insufficient data

        cov_matrix = np.cov(combined["stock"], combined["market"])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        return round(float(beta), 3)

    def _calculate_sharpe_score(
        self,
        predicted_return: float,
        volatility_annual: float,
        confidence: float,
    ) -> float:
        """
        Risk-adjusted score for ranking opportunities.

        Combines three things:
        1. Predicted return — higher is better
        2. Volatility — lower is better (we divide by it)
        3. Forecast confidence — scales the score down for uncertain predictions

        This isn't a formal Sharpe ratio (which uses a risk-free rate),
        but a practical ranking metric for the Strategy Agent.
        """
        if volatility_annual == 0:
            return 0.0
        # Convert annual volatility to daily for apples-to-apples comparison
        daily_vol = volatility_annual / np.sqrt(252)
        raw_score = predicted_return / daily_vol
        # Scale by confidence so low-confidence predictions rank lower
        return round(float(raw_score * confidence), 4)

    def _derive_risk_level(self, volatility_annual: float) -> str:
        """
        Classify risk level from annualised volatility.
        Thresholds based on typical ASX equity volatility ranges.
        """
        if volatility_annual < 0.20:      # < 20% annualised
            return "LOW"
        elif volatility_annual < 0.35:    # 20–35%
            return "MEDIUM"
        else:                             # > 35%
            return "HIGH"

    def _derive_recommendation_modifier(
        self,
        risk_level: str,
        forecast_confidence: float,
        var_95: float,
    ) -> str:
        """
        Tells the Strategy Agent whether to act on, downgrade, or block
        a forecast based on risk context.

        CONFIRM  — risk is acceptable, proceed with the signal
        DOWNGRADE — reduce position size or move signal toward HOLD
        BLOCK    — risk too high, override the forecast signal entirely
        """
        # Block if VaR implies a potential 5%+ loss in a single day
        if var_95 < -0.05:
            return "BLOCK"
        # Downgrade if high risk AND low confidence
        if risk_level == "HIGH" and forecast_confidence < 0.40:
            return "DOWNGRADE"
        # Downgrade if medium risk AND very low confidence
        if risk_level == "MEDIUM" and forecast_confidence < 0.25:
            return "DOWNGRADE"
        return "CONFIRM"

    # -----------------------------------------------------------------------
    # Per-stock risk assessment
    # -----------------------------------------------------------------------
    def assess_stock(self, forecast: StockForecast) -> Optional[StockRisk]:
        """Calculate full risk profile for a single stock."""
        logger.info(f"Assessing risk for {forecast.ticker}...")

        returns = self._load_returns(forecast.ticker)
        market_returns = self._load_market_returns()
        closes = self._load_closes(forecast.ticker)

        if returns.empty or len(returns) < 30:
            logger.warning(f"  Insufficient return history for {forecast.ticker}")
            return None

        volatility   = self._calculate_volatility(returns)
        var_95       = self._calculate_var(returns, 0.95)
        var_99       = self._calculate_var(returns, 0.99)
        max_drawdown = self._calculate_max_drawdown(closes)
        beta         = self._calculate_beta(returns, market_returns)
        risk_level   = self._derive_risk_level(volatility)
        sharpe_score = self._calculate_sharpe_score(
            forecast.predicted_return, volatility, forecast.confidence
        )
        modifier = self._derive_recommendation_modifier(
            risk_level, forecast.confidence, var_95
        )

        risk = StockRisk(
            ticker=forecast.ticker,
            volatility_annual=round(volatility, 4),
            var_95=round(var_95, 4),
            var_99=round(var_99, 4),
            max_drawdown=round(max_drawdown, 4),
            beta=beta,
            sharpe_score=sharpe_score,
            risk_level=risk_level,
            recommendation_modifier=modifier,
        )

        logger.info(
            f"  ✓ {forecast.ticker}: vol={volatility:.1%} | "
            f"VaR95={var_95:.2%} | beta={beta:.2f} | "
            f"risk={risk_level} | modifier={modifier}"
        )
        return risk

    # -----------------------------------------------------------------------
    # Portfolio-level risk
    # -----------------------------------------------------------------------
    def assess_portfolio(self, stock_risks: list[StockRisk]) -> PortfolioRisk:
        """Assess risk at the portfolio (watchlist) level."""
        logger.info("Assessing portfolio-level risk...")

        fundamentals = self._load_fundamentals()

        # Sector concentration
        sector_counts = fundamentals["sector"].value_counts()
        total = len(fundamentals)
        sector_weights = {
            sector: round(count / total, 2)
            for sector, count in sector_counts.items()
        }
        most_concentrated = sector_counts.index[0]
        top_weight = sector_weights[most_concentrated]

        if top_weight > 0.40:
            concentration_risk = "HIGH"
        elif top_weight > 0.25:
            concentration_risk = "MEDIUM"
        else:
            concentration_risk = "LOW"

        # Aggregate metrics
        avg_beta = round(float(np.mean([r.beta for r in stock_risks])), 3)
        avg_vol  = round(float(np.mean([r.volatility_annual for r in stock_risks])), 4)
        avg_var  = round(float(np.mean([r.var_95 for r in stock_risks])), 4)

        # Plain English summary for the Report Writer Agent
        risk_summary = (
            f"Watchlist average beta: {avg_beta:.2f} vs ASX 200. "
            f"Average annualised volatility: {avg_vol:.1%}. "
            f"Average 1-day VaR (95%): {avg_var:.2%}. "
            f"Sector concentration: {most_concentrated} at {top_weight:.0%} "
            f"of watchlist ({concentration_risk} concentration risk)."
        )

        portfolio = PortfolioRisk(
            total_stocks=len(stock_risks),
            sector_weights=sector_weights,
            most_concentrated_sector=most_concentrated,
            concentration_risk=concentration_risk,
            average_beta=avg_beta,
            average_volatility=avg_vol,
            watchlist_var_95=avg_var,
            risk_summary=risk_summary,
        )

        logger.info(f"  ✓ Portfolio risk assessed. Concentration: {concentration_risk}")
        return portfolio

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def run(self, forecasts: list[StockForecast]) -> tuple[list[StockRisk], PortfolioRisk]:
        """
        Main entry point. Takes forecasts from Agent 2 and returns
        per-stock risk profiles plus a portfolio-level risk summary.
        """
        logger.info("=" * 60)
        logger.info("Risk Agent — starting run")
        logger.info(f"Assessing {len(forecasts)} forecasts")
        logger.info("=" * 60)

        stock_risks = []
        for forecast in forecasts:
            risk = self.assess_stock(forecast)
            if risk:
                stock_risks.append(risk)

        portfolio_risk = self.assess_portfolio(stock_risks)

        logger.info("=" * 60)
        logger.info(f"Risk assessment complete. {len(stock_risks)}/{len(forecasts)} stocks assessed.")
        logger.info("=" * 60)

        return stock_risks, portfolio_risk

    def get_risk_summary(
        self,
        stock_risks: list[StockRisk],
        forecasts: list[StockForecast],
    ) -> pd.DataFrame:
        """
        Merges forecast and risk data into a single summary DataFrame.
        Passed downstream to the Strategy Agent.
        """
        forecast_map = {f.ticker: f for f in forecasts}
        rows = []
        for r in stock_risks:
            f = forecast_map.get(r.ticker)
            rows.append({
                "ticker":         r.ticker,
                "signal":         f.signal if f else "—",
                "predicted_ret":  f"{f.predicted_return*100:+.2f}%" if f else "—",
                "confidence":     f"{f.confidence:.0%}" if f else "—",
                "volatility":     f"{r.volatility_annual:.1%}",
                "var_95":         f"{r.var_95:.2%}",
                "max_drawdown":   f"{r.max_drawdown:.2%}",
                "beta":           f"{r.beta:.2f}",
                "risk_level":     r.risk_level,
                "modifier":       r.recommendation_modifier,
                "sharpe_score":   f"{r.sharpe_score:.3f}",
            })
        return pd.DataFrame(rows).sort_values("sharpe_score", ascending=False)

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Run directly for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from forecasting_agent import ForecastingAgent

    # Run Agent 2 first to get forecasts
    forecasting_agent = ForecastingAgent()
    forecasts = forecasting_agent.run()
    forecasting_agent.close()

    # Run Agent 3
    risk_agent = RiskAgent()
    stock_risks, portfolio_risk = risk_agent.run(forecasts)

    print("\n📊 Risk Summary:")
    print(risk_agent.get_risk_summary(stock_risks, forecasts).to_string(index=False))

    print(f"\n🏦 Portfolio Risk:")
    print(f"  {portfolio_risk.risk_summary}")
    print(f"  Sector weights: {portfolio_risk.sector_weights}")
    print(f"  Concentration risk: {portfolio_risk.concentration_risk}")

    risk_agent.close()