"""Quantitative risk assessment for individual securities and watchlists."""

import logging
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from agents.forecasting_agent import StockForecast


logger = logging.getLogger(__name__)


TRADING_SESSIONS = 252
MIN_RISK_OBSERVATIONS = 30

LOW_RISK_THRESHOLD = 0.20
HIGH_RISK_THRESHOLD = 0.35


@dataclass
class StockRisk:
    """Historical risk metrics and forecast risk controls."""

    ticker: str
    volatility_annual: float
    var_95: float
    var_99: float
    max_drawdown: float
    beta: float | None
    sharpe_score: float
    risk_level: str
    recommendation_modifier: str

    @property
    def risk_adjusted_forecast_score(self) -> float:
        """Return the custom forecast-to-volatility ranking score."""
        return self.sharpe_score


@dataclass
class PortfolioRisk:
    """Aggregate watchlist statistics, not capital-weighted portfolio risk."""

    total_stocks: int
    sector_weights: dict[str, float]
    most_concentrated_sector: str
    concentration_risk: str
    average_beta: float | None
    average_volatility: float
    watchlist_var_95: float
    risk_summary: str


class RiskAgent:
    """Calculate historical risk metrics for forecasted securities."""

    def __init__(
        self,
        db_path: str = "asx_market_data.db",
    ):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)

    def _load_returns(
        self,
        ticker: str,
        days: int = TRADING_SESSIONS,
        as_of_date: str | None = None,
    ) -> pd.Series:
        """Load recent equity returns indexed by observation date."""

        df = self.conn.execute(
            """
            SELECT date, daily_return
            FROM ohlcv
            WHERE ticker = ?
              AND date <= COALESCE(
                  CAST(? AS DATE),
                  CURRENT_DATE
              )
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, as_of_date, days),
        ).df()

        return self._prepare_series(
            df,
            "daily_return",
        )

    def _load_market_returns(
        self,
        days: int = TRADING_SESSIONS,
        as_of_date: str | None = None,
    ) -> pd.Series:
        """Load benchmark returns indexed by observation date."""

        df = self.conn.execute(
            """
            SELECT date, daily_return
            FROM macro
            WHERE ticker = '^AXJO'
              AND date <= COALESCE(
                  CAST(? AS DATE),
                  CURRENT_DATE
              )
            ORDER BY date DESC
            LIMIT ?
            """,
            (as_of_date, days),
        ).df()

        return self._prepare_series(
            df,
            "daily_return",
        )

    def _load_closes(
        self,
        ticker: str,
        days: int = TRADING_SESSIONS,
        as_of_date: str | None = None,
    ) -> pd.Series:
        """Load recent closing prices in chronological order."""

        df = self.conn.execute(
            """
            SELECT date, close
            FROM ohlcv
            WHERE ticker = ?
              AND date <= COALESCE(
                  CAST(? AS DATE),
                  CURRENT_DATE
              )
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, as_of_date, days),
        ).df()

        return self._prepare_series(
            df,
            "close",
        )

    def _load_fundamentals(self) -> pd.DataFrame:
        """Retrieve the most recent sector metadata per security."""

        return self.conn.execute(
            """
            SELECT DISTINCT ON (ticker)
                ticker,
                sector
            FROM fundamentals
            ORDER BY ticker, fetched_at DESC
            """
        ).df()

    @staticmethod
    def _prepare_series(
        df: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        """Prepare finite numerical observations indexed by date."""

        if df.empty:
            return pd.Series(dtype=float)

        df = df.copy()

        df["date"] = pd.to_datetime(df["date"])

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        series = (
            df.drop_duplicates("date", keep="last")
            .set_index("date")[column]
            .sort_index()
        )

        return (
            series.replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )

    @staticmethod
    def _calculate_volatility(
        returns: pd.Series,
    ) -> float:
        """Annualise the sample standard deviation of daily returns."""

        if len(returns) < 2:
            raise ValueError(
                "At least two returns are required."
            )

        return float(
            returns.std(ddof=1)
            * np.sqrt(TRADING_SESSIONS)
        )

    @staticmethod
    def _calculate_var(
        returns: pd.Series,
        confidence: float,
    ) -> float:
        """Calculate a signed historical downside return quantile."""

        if returns.empty:
            raise ValueError("Return history is empty.")

        if not 0 < confidence < 1:
            raise ValueError(
                "Confidence must be between zero and one."
            )

        return float(
            np.quantile(
                returns.to_numpy(),
                1 - confidence,
            )
        )

    @staticmethod
    def _calculate_max_drawdown(
        closes: pd.Series,
    ) -> float:
        """Calculate the worst historical peak-to-trough decline."""

        if closes.empty or (closes <= 0).any():
            raise ValueError(
                "Drawdown requires positive closing prices."
            )

        closes = closes.sort_index()

        running_peak = closes.cummax()

        drawdown = closes / running_peak - 1

        return float(drawdown.min())

    @staticmethod
    def _calculate_beta(
        stock_returns: pd.Series,
        market_returns: pd.Series,
    ) -> float | None:
        """Estimate beta using date-aligned stock and market returns."""

        aligned = pd.concat(
            [
                stock_returns.rename("stock"),
                market_returns.rename("market"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if len(aligned) < MIN_RISK_OBSERVATIONS:
            return None

        market_variance = aligned["market"].var(ddof=1)

        if (
            not np.isfinite(market_variance)
            or market_variance <= 0
        ):
            return None

        covariance = aligned["stock"].cov(
            aligned["market"]
        )

        beta = covariance / market_variance

        if not np.isfinite(beta):
            return None

        return float(beta)

    @staticmethod
    def _calculate_sharpe_score(
        predicted_return: float,
        volatility_annual: float,
        confidence: float,
    ) -> float:
        """Calculate the custom confidence-weighted forecast score.

        The method name is retained for compatibility. The result is
        not a conventional Sharpe ratio.
        """

        if (
            not np.isfinite(volatility_annual)
            or volatility_annual <= 0
        ):
            return 0.0

        daily_volatility = (
            volatility_annual / np.sqrt(TRADING_SESSIONS)
        )

        return float(
            predicted_return
            / daily_volatility
            * confidence
        )

    @staticmethod
    def _derive_risk_level(
        volatility_annual: float,
    ) -> str:
        """Classify risk using annualised volatility thresholds."""

        if (
            not np.isfinite(volatility_annual)
            or volatility_annual < 0
        ):
            raise ValueError("Invalid annualised volatility.")

        if volatility_annual < LOW_RISK_THRESHOLD:
            return "LOW"

        if volatility_annual < HIGH_RISK_THRESHOLD:
            return "MEDIUM"

        return "HIGH"

    @staticmethod
    def _derive_recommendation_modifier(
        risk_level: str,
        forecast_confidence: float,
        var_95: float,
    ) -> str:
        """Apply the existing deterministic risk policy."""

        if var_95 < -0.05:
            return "BLOCK"

        if (
            risk_level == "HIGH"
            and forecast_confidence < 0.40
        ):
            return "DOWNGRADE"

        if (
            risk_level == "MEDIUM"
            and forecast_confidence < 0.25
        ):
            return "DOWNGRADE"

        return "CONFIRM"

    def assess_stock(
        self,
        forecast: StockForecast,
    ) -> StockRisk | None:
        """Calculate risk using data available at the forecast origin."""

        ticker = forecast.ticker
        as_of_date = forecast.as_of_date

        logger.info("Assessing risk for %s", ticker)

        returns = self._load_returns(
            ticker,
            as_of_date=as_of_date,
        )

        if len(returns) < MIN_RISK_OBSERVATIONS:
            logger.warning(
                "Insufficient return history for %s",
                ticker,
            )
            return None

        market_returns = self._load_market_returns(
            as_of_date=as_of_date,
        )

        closes = self._load_closes(
            ticker,
            as_of_date=as_of_date,
        )

        if closes.empty:
            logger.warning(
                "Closing-price history unavailable for %s",
                ticker,
            )
            return None

        volatility = self._calculate_volatility(returns)

        var_95 = self._calculate_var(
            returns,
            0.95,
        )

        var_99 = self._calculate_var(
            returns,
            0.99,
        )

        max_drawdown = self._calculate_max_drawdown(
            closes
        )

        beta = self._calculate_beta(
            returns,
            market_returns,
        )

        risk_level = self._derive_risk_level(
            volatility
        )

        score = self._calculate_sharpe_score(
            forecast.predicted_return,
            volatility,
            forecast.confidence,
        )

        modifier = self._derive_recommendation_modifier(
            risk_level,
            forecast.confidence,
            var_95,
        )

        risk = StockRisk(
            ticker=ticker,
            volatility_annual=volatility,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            beta=beta,
            sharpe_score=score,
            risk_level=risk_level,
            recommendation_modifier=modifier,
        )

        beta_text = (
            f"{beta:.2f}"
            if beta is not None
            else "N/A"
        )

        logger.info(
            "%s | volatility %.1f%% | VaR95 %.2f%% | "
            "beta %s | risk %s | modifier %s",
            ticker,
            volatility * 100,
            var_95 * 100,
            beta_text,
            risk_level,
            modifier,
        )

        return risk

    def assess_portfolio(
        self,
        stock_risks: list[StockRisk],
    ) -> PortfolioRisk:
        """Summarise risk across the assessed watchlist."""

        if not stock_risks:
            raise ValueError(
                "Cannot assess an empty watchlist."
            )

        tickers = sorted(
            {risk.ticker for risk in stock_risks}
        )

        fundamentals = self._load_fundamentals()

        sector_map = dict(
            zip(
                fundamentals["ticker"],
                fundamentals["sector"],
            )
        )

        sectors = pd.Series(
            [sector_map.get(ticker) for ticker in tickers],
            dtype="object",
        )

        sectors = (
            sectors.fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

        sector_counts = sectors.value_counts()

        total = len(tickers)

        sector_weights = {
            sector: float(count / total)
            for sector, count in sector_counts.items()
        }

        known_weights = {
            sector: weight
            for sector, weight in sector_weights.items()
            if sector != "Unknown"
        }

        if not known_weights:
            most_concentrated = "Unknown"
            top_weight = 0.0
            concentration_risk = "UNAVAILABLE"

        else:
            most_concentrated = sorted(
                known_weights,
                key=lambda sector: (
                    -known_weights[sector],
                    sector,
                ),
            )[0]

            top_weight = known_weights[
                most_concentrated
            ]

            if top_weight > 0.40:
                concentration_risk = "HIGH"

            elif sector_weights.get("Unknown", 0) > 0:
                concentration_risk = "INCOMPLETE"

            elif top_weight > 0.25:
                concentration_risk = "MEDIUM"

            else:
                concentration_risk = "LOW"

        valid_betas = [
            risk.beta
            for risk in stock_risks
            if risk.beta is not None
            and np.isfinite(risk.beta)
        ]

        average_beta = (
            float(np.mean(valid_betas))
            if valid_betas
            else None
        )

        average_volatility = float(
            np.mean(
                [
                    risk.volatility_annual
                    for risk in stock_risks
                ]
            )
        )

        average_var = float(
            np.mean(
                [risk.var_95 for risk in stock_risks]
            )
        )

        beta_text = (
            f"{average_beta:.2f}"
            if average_beta is not None
            else "Unavailable"
        )

        if known_weights:
            concentration_text = (
                f"Largest identified sector: "
                f"{most_concentrated} at {top_weight:.0%}. "
                f"Concentration: {concentration_risk}."
            )

        else:
            concentration_text = (
                "Sector concentration is unavailable."
            )

        risk_summary = (
            f"Mean individual-security beta: {beta_text} "
            f"({len(valid_betas)}/{total} securities). "
            f"Mean annualised volatility: "
            f"{average_volatility:.1%}. "
            f"Mean individual-security 95% historical VaR: "
            f"{average_var:.2%}. "
            f"{concentration_text} "
            f"Unknown sector coverage: "
            f"{sector_weights.get('Unknown', 0):.0%}."
        )

        return PortfolioRisk(
            total_stocks=total,
            sector_weights=sector_weights,
            most_concentrated_sector=most_concentrated,
            concentration_risk=concentration_risk,
            average_beta=average_beta,
            average_volatility=average_volatility,
            watchlist_var_95=average_var,
            risk_summary=risk_summary,
        )

    def run(
        self,
        forecasts: list[StockForecast],
    ) -> tuple[list[StockRisk], PortfolioRisk]:
        """Assess securities and aggregate their risk statistics."""

        stock_risks: list[StockRisk] = []

        for forecast in forecasts:
            try:
                risk = self.assess_stock(forecast)

                if risk is not None:
                    stock_risks.append(risk)

            except Exception:
                logger.exception(
                    "Risk assessment failed for %s",
                    forecast.ticker,
                )

        portfolio_risk = self.assess_portfolio(
            stock_risks
        )

        logger.info(
            "Risk assessment complete: %d/%d successful",
            len(stock_risks),
            len(forecasts),
        )

        return stock_risks, portfolio_risk

    def get_risk_summary(
        self,
        stock_risks: list[StockRisk],
        forecasts: list[StockForecast],
    ) -> pd.DataFrame:
        """Return formatted security-level risk diagnostics."""

        forecast_map = {
            forecast.ticker: forecast
            for forecast in forecasts
        }

        rows = []

        sorted_risks = sorted(
            stock_risks,
            key=lambda risk: (
                -risk.risk_adjusted_forecast_score,
                risk.ticker,
            ),
        )

        for risk in sorted_risks:
            forecast = forecast_map.get(risk.ticker)

            rows.append(
                {
                    "ticker": risk.ticker,
                    "signal": (
                        forecast.signal
                        if forecast
                        else "N/A"
                    ),
                    "predicted_return": (
                        f"{forecast.predicted_return:+.2%}"
                        if forecast
                        else "N/A"
                    ),
                    "volatility": (
                        f"{risk.volatility_annual:.1%}"
                    ),
                    "var_95": f"{risk.var_95:.2%}",
                    "var_99": f"{risk.var_99:.2%}",
                    "max_drawdown": (
                        f"{risk.max_drawdown:.2%}"
                    ),
                    "beta": (
                        f"{risk.beta:.2f}"
                        if risk.beta is not None
                        else "N/A"
                    ),
                    "risk_level": risk.risk_level,
                    "modifier": (
                        risk.recommendation_modifier
                    ),
                    "risk_adjusted_score": (
                        f"{risk.risk_adjusted_forecast_score:.3f}"
                    ),
                }
            )

        return pd.DataFrame(rows)

    def close(self) -> None:
        """Close the database connection."""

        self.conn.close()


if __name__ == "__main__":
    from agents.forecasting_agent import ForecastingAgent

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(name)s | "
            "%(levelname)s | %(message)s"
        ),
    )

    forecasting_agent = ForecastingAgent()

    try:
        forecasts = forecasting_agent.run()

    finally:
        forecasting_agent.close()

    risk_agent = RiskAgent()

    try:
        stock_risks, watchlist_risk = risk_agent.run(
            forecasts
        )

        print("\nRisk Summary")
        print(
            risk_agent.get_risk_summary(
                stock_risks,
                forecasts,
            ).to_string(index=False)
        )

        print("\nWatchlist Risk")
        print(watchlist_risk.risk_summary)

    finally:
        risk_agent.close()