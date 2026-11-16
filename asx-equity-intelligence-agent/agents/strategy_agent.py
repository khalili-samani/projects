"""Deterministic equity recommendation and position-adjustment rules."""

import logging
import math
from dataclasses import dataclass

import pandas as pd

from agents.forecasting_agent import StockForecast
from agents.risk_agent import PortfolioRisk, StockRisk


logger = logging.getLogger(__name__)


SIGNAL_RESOLUTION = {
    ("BUY", "CONFIRM"): "BUY",
    ("BUY", "DOWNGRADE"): "HOLD",
    ("BUY", "BLOCK"): "HOLD",
    ("HOLD", "CONFIRM"): "HOLD",
    ("HOLD", "DOWNGRADE"): "HOLD",
    ("HOLD", "BLOCK"): "HOLD",
    ("REDUCE", "CONFIRM"): "REDUCE",
    ("REDUCE", "DOWNGRADE"): "HOLD",
    ("REDUCE", "BLOCK"): "HOLD",
}


FEATURE_LABELS = {
    "return_5d": "short-term momentum",
    "return_10d": "medium-term momentum",
    "return_20d": "20-session momentum",
    "volatility_20d": "recent volatility",
    "rsi_14": "Relative Strength Index",
    "volume_ratio": "relative trading volume",
    "dist_52w_high": "distance from the rolling price high",
    "dist_52w_low": "distance from the rolling price low",
    "ma_crossover": "moving-average relationship",
    "asx200_return": "ASX 200 daily return",
    "asx200_return_5d": "ASX 200 mean five-session return",
}


@dataclass
class TradeRecommendation:
    """Final recommendation and its supporting analytical context."""

    ticker: str
    forecast_date: str
    forecast_signal: str
    risk_modifier: str
    final_signal: str
    predicted_return: float
    confidence: float
    risk_level: str
    sharpe_score: float
    position_size: float
    position_label: str
    rationale: str
    priority: int

    @property
    def as_of_date(self) -> str:
        """Return the observation date associated with the forecast."""

        return self.forecast_date

    @property
    def risk_adjusted_forecast_score(self) -> float:
        """Return the custom forecast-to-volatility ranking score."""

        return self.sharpe_score


@dataclass
class StrategyOutput:
    """Complete recommendation output for an assessed watchlist."""

    recommendations: list[TradeRecommendation]
    portfolio_risk: PortfolioRisk
    actionable_signals: list[TradeRecommendation]
    market_bias: str
    strategy_summary: str


class StrategyAgent:
    """Combine numerical forecasts with deterministic risk controls."""

    def __init__(
        self,
        db_path: str = "asx_market_data.db",
        base_position_size: float = 0.10,
        min_confidence: float = 0.30,
    ):
        if not 0 < base_position_size <= (1 / 1.5):
            raise ValueError(
                "Invalid base position size."
            )

        if not 0 <= min_confidence <= 1:
            raise ValueError(
                "Minimum confidence must be between zero and one."
            )

        self.db_path = db_path
        self.base_position_size = base_position_size
        self.min_confidence = min_confidence

    @staticmethod
    def _resolve_signal(
        forecast_signal: str,
        risk_modifier: str,
    ) -> str:
        """Resolve a forecast and risk modifier into a final signal."""

        return SIGNAL_RESOLUTION.get(
            (forecast_signal, risk_modifier),
            "HOLD",
        )

    def _calculate_position_size(
        self,
        final_signal: str,
        confidence: float,
        risk_level: str,
        sharpe_score: float = 0.0,
    ) -> tuple[float, str]:
        """Calculate a heuristic adjustment magnitude.

        The optional score argument is retained for compatibility.
        Position adjustments depend on confidence and risk level,
        not the ranking score.
        """

        if final_signal not in {"BUY", "REDUCE"}:
            return 0.0, "NONE"

        if (
            not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or confidence < self.min_confidence
        ):
            return 0.0, "NONE"

        if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
            return 0.0, "NONE"

        size = self.base_position_size

        if confidence >= 0.60:
            size *= 1.50

        elif confidence < 0.45:
            size *= 0.50

        risk_multiplier = {
            "LOW": 1.00,
            "MEDIUM": 0.75,
            "HIGH": 0.50,
        }

        size *= risk_multiplier[risk_level]

        maximum_size = self.base_position_size * 1.50

        size = min(size, maximum_size)

        size = round(size, 3)

        if size >= self.base_position_size * 1.20:
            label = "FULL"

        elif size >= self.base_position_size * 0.70:
            label = "REDUCED"

        else:
            label = "MINIMAL"

        return size, label

    def _generate_rationale(
        self,
        forecast: StockForecast,
        risk: StockRisk,
        final_signal: str,
        position_label: str,
    ) -> str:
        """Construct a factual rationale without causal model claims."""

        rationale = [
            f"Final signal: {final_signal}.",
            (
                f"Predicted next-session return: "
                f"{forecast.predicted_return:+.2%}."
            ),
            (
                f"Model-agreement score: "
                f"{forecast.confidence:.0%}."
            ),
            (
                f"Annualised volatility: "
                f"{risk.volatility_annual:.1%}."
            ),
            (
                f"One-day historical VaR (95%): "
                f"{risk.var_95:.2%}."
            ),
            f"Risk classification: {risk.risk_level}.",
            f"Position adjustment category: {position_label}.",
        ]

        if forecast.confidence < self.min_confidence:
            rationale.append(
                "The forecast did not meet the minimum "
                "model-agreement threshold."
            )

        elif forecast.signal != final_signal:
            rationale.append(
                f"The original {forecast.signal} signal "
                f"was changed to {final_signal} under the "
                f"{risk.recommendation_modifier} risk rule."
            )

        if forecast.feature_importance:
            top_feature = max(
                forecast.feature_importance,
                key=forecast.feature_importance.get,
            )

            feature_label = FEATURE_LABELS.get(
                top_feature,
                top_feature,
            )

            rationale.append(
                f"Highest-ranked global model feature: "
                f"{feature_label}. "
                "This does not establish the cause of "
                "the individual prediction."
            )

        return " ".join(rationale)

    @staticmethod
    def _derive_market_bias(
        recommendations: list[TradeRecommendation],
    ) -> str:
        """Summarise the distribution of watchlist recommendations."""

        if not recommendations:
            return "NEUTRAL"

        total = len(recommendations)

        buys = sum(
            rec.final_signal == "BUY"
            for rec in recommendations
        )

        reduces = sum(
            rec.final_signal == "REDUCE"
            for rec in recommendations
        )

        if buys / total > 0.40:
            return "BULLISH"

        if reduces / total > 0.40:
            return "BEARISH"

        return "NEUTRAL"

    def _process_pair(
        self,
        forecast: StockForecast,
        risk: StockRisk,
        priority: int,
    ) -> TradeRecommendation:
        """Combine an individual forecast with its risk assessment."""

        if forecast.ticker != risk.ticker:
            raise ValueError(
                "Forecast and risk assessment tickers do not match."
            )

        if not math.isfinite(forecast.predicted_return):
            raise ValueError(
                "Predicted return must be finite."
            )

        if (
            not math.isfinite(forecast.confidence)
            or not 0 <= forecast.confidence <= 1
        ):
            raise ValueError(
                "Invalid forecast confidence."
            )

        if forecast.confidence < self.min_confidence:
            final_signal = "HOLD"

        elif risk.risk_level not in {"LOW", "MEDIUM", "HIGH"}:
            final_signal = "HOLD"

        else:
            final_signal = self._resolve_signal(
                forecast.signal,
                risk.recommendation_modifier,
            )

        position_size, position_label = (
            self._calculate_position_size(
                final_signal,
                forecast.confidence,
                risk.risk_level,
            )
        )

        rationale = self._generate_rationale(
            forecast,
            risk,
            final_signal,
            position_label,
        )

        recommendation = TradeRecommendation(
            ticker=forecast.ticker,
            forecast_date=forecast.as_of_date,
            forecast_signal=forecast.signal,
            risk_modifier=risk.recommendation_modifier,
            final_signal=final_signal,
            predicted_return=forecast.predicted_return,
            confidence=forecast.confidence,
            risk_level=risk.risk_level,
            sharpe_score=risk.risk_adjusted_forecast_score,
            position_size=position_size,
            position_label=position_label,
            rationale=rationale,
            priority=priority,
        )

        logger.info(
            "%s | %s + %s -> %s | adjustment %s",
            forecast.ticker,
            forecast.signal,
            risk.recommendation_modifier,
            final_signal,
            position_label,
        )

        return recommendation

    def run(
        self,
        forecasts: list[StockForecast],
        stock_risks: list[StockRisk],
        portfolio_risk: PortfolioRisk,
    ) -> StrategyOutput:
        """Generate and rank recommendations for assessed securities."""

        forecast_map = {
            forecast.ticker: forecast
            for forecast in forecasts
        }

        risk_map = {
            risk.ticker: risk
            for risk in stock_risks
        }

        common_tickers = (
            forecast_map.keys()
            & risk_map.keys()
        )

        if not common_tickers:
            raise ValueError(
                "No securities have both forecast and risk data."
            )

        if portfolio_risk.total_stocks != len(common_tickers):
            raise ValueError(
                "Watchlist risk coverage does not match "
                "the recommendation universe."
            )

        sorted_tickers = sorted(
            common_tickers,
            key=lambda ticker: (
                -risk_map[ticker].risk_adjusted_forecast_score,
                ticker,
            ),
        )

        recommendations = []

        for priority, ticker in enumerate(
            sorted_tickers,
            start=1,
        ):
            recommendation = self._process_pair(
                forecast_map[ticker],
                risk_map[ticker],
                priority,
            )

            recommendations.append(recommendation)

        actionable = [
            rec
            for rec in recommendations
            if rec.final_signal in {"BUY", "REDUCE"}
        ]

        market_bias = self._derive_market_bias(
            recommendations
        )

        buy_count = sum(
            rec.final_signal == "BUY"
            for rec in recommendations
        )

        hold_count = sum(
            rec.final_signal == "HOLD"
            for rec in recommendations
        )

        reduce_count = sum(
            rec.final_signal == "REDUCE"
            for rec in recommendations
        )

        strategy_summary = (
            f"Watchlist signal bias: {market_bias}. "
            f"Signal distribution: {buy_count} BUY, "
            f"{hold_count} HOLD, {reduce_count} REDUCE "
            f"across {len(recommendations)} securities. "
            f"{portfolio_risk.risk_summary}"
        )

        logger.info(
            "Strategy complete | BUY: %d | HOLD: %d | "
            "REDUCE: %d | bias: %s",
            buy_count,
            hold_count,
            reduce_count,
            market_bias,
        )

        return StrategyOutput(
            recommendations=recommendations,
            portfolio_risk=portfolio_risk,
            actionable_signals=actionable,
            market_bias=market_bias,
            strategy_summary=strategy_summary,
        )

    def get_recommendation_summary(
        self,
        output: StrategyOutput,
    ) -> pd.DataFrame:
        """Return formatted recommendations for inspection."""

        rows = []

        for recommendation in output.recommendations:
            rows.append(
                {
                    "ticker": recommendation.ticker,
                    "as_of_date": recommendation.as_of_date,
                    "forecast_signal": (
                        recommendation.forecast_signal
                    ),
                    "modifier": (
                        recommendation.risk_modifier
                    ),
                    "final_signal": recommendation.final_signal,
                    "predicted_return": (
                        f"{recommendation.predicted_return:+.2%}"
                    ),
                    "confidence": (
                        f"{recommendation.confidence:.0%}"
                    ),
                    "risk_level": recommendation.risk_level,
                    "adjustment": (
                        f"{recommendation.position_size:.1%}"
                    ),
                    "position_label": (
                        recommendation.position_label
                    ),
                    "risk_adjusted_score": (
                        f"{recommendation.risk_adjusted_forecast_score:.3f}"
                    ),
                }
            )

        return pd.DataFrame(rows)


if __name__ == "__main__":
    from agents.forecasting_agent import ForecastingAgent
    from agents.risk_agent import RiskAgent

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
        stock_risks, portfolio_risk = risk_agent.run(
            forecasts
        )

    finally:
        risk_agent.close()

    strategy_agent = StrategyAgent()

    output = strategy_agent.run(
        forecasts,
        stock_risks,
        portfolio_risk,
    )

    print("\nRecommendation Summary")
    print(
        strategy_agent.get_recommendation_summary(
            output
        ).to_string(index=False)
    )

    print("\nStrategy Summary")
    print(output.strategy_summary)