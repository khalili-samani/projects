"""Financial feature engineering and next-session return forecasting."""

import logging
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_20d",
    "rsi_14",
    "volume_ratio",
    "dist_52w_high",
    "dist_52w_low",
    "ma_crossover",
    "asx200_return",
    "asx200_return_5d",
]

BUY_THRESHOLD = 0.005
REDUCE_THRESHOLD = -0.005
CONFIDENCE_SCALE = 0.02


@dataclass
class StockForecast:
    """Forecast and model diagnostics for an individual security."""

    ticker: str
    forecast_date: str
    predicted_return: float
    confidence: float
    mae: float
    signal: str
    feature_importance: dict[str, float]
    baseline_mae: float | None = None

    @property
    def as_of_date(self) -> str:
        """Return the observation date used for inference."""
        return self.forecast_date


def engineer_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Construct trailing market features and forward return targets."""

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    # Momentum features
    for window in (5, 10, 20):
        df[f"return_{window}d"] = df["close"].pct_change(
            periods=window,
            fill_method=None,
        )

    # Historical volatility
    df["volatility_20d"] = (
        df["daily_return"].rolling(20).std()
    )

    # Relative Strength Index using simple rolling averages
    delta = df["close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    relative_strength = gain / loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (loss == 0) & (gain > 0),
        100.0,
    )

    rsi = rsi.mask(
        (gain == 0) & (loss == 0),
        50.0,
    )

    df["rsi_14"] = rsi

    # Relative trading volume
    average_volume = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] / average_volume.replace(0, np.nan)
    )

    # Distance from historical closing-price extremes
    rolling_high = df["close"].rolling(
        252,
        min_periods=1,
    ).max()

    rolling_low = df["close"].rolling(
        252,
        min_periods=1,
    ).min()

    df["dist_52w_high"] = df["close"] / rolling_high - 1
    df["dist_52w_low"] = df["close"] / rolling_low - 1

    # Moving-average relationship
    ma_10 = df["close"].rolling(
        10,
        min_periods=1,
    ).mean()

    ma_50 = df["close"].rolling(
        50,
        min_periods=1,
    ).mean()

    df["ma_crossover"] = ma_10 / ma_50 - 1

    # Align benchmark observations by trading date
    if macro is None or macro.empty:
        df["asx200_return"] = np.nan
        df["asx200_return_5d"] = np.nan

    else:
        benchmark = macro[
            ["date", "daily_return"]
        ].copy()

        benchmark["date"] = pd.to_datetime(
            benchmark["date"]
        )

        benchmark = (
            benchmark.sort_values("date")
            .drop_duplicates("date", keep="last")
        )

        benchmark = benchmark.rename(
            columns={"daily_return": "asx200_return"}
        )

        benchmark["asx200_return_5d"] = (
            benchmark["asx200_return"].rolling(5).mean()
        )

        df = df.merge(
            benchmark,
            on="date",
            how="left",
            validate="one_to_one",
        )

    df = df.sort_values("date").reset_index(drop=True)

    # Forward target: return of the following stored observation
    df["target"] = df["daily_return"].shift(-1)

    df[FEATURE_COLS + ["target"]] = (
        df[FEATURE_COLS + ["target"]]
        .replace([np.inf, -np.inf], np.nan)
    )

    return df


def derive_signal(predicted_return: float) -> str:
    """Convert a predicted return into a preliminary signal."""

    if not np.isfinite(predicted_return):
        raise ValueError("Predicted return must be finite.")

    if predicted_return > BUY_THRESHOLD:
        return "BUY"

    if predicted_return < REDUCE_THRESHOLD:
        return "REDUCE"

    return "HOLD"


def derive_confidence(
    model: RandomForestRegressor,
    features: np.ndarray,
) -> float:
    """Calculate an uncalibrated model-agreement score."""

    predictions = np.array(
        [
            tree.predict(features)
            for tree in model.estimators_
        ]
    )

    spread = predictions.std()

    confidence = np.clip(
        1 - spread / CONFIDENCE_SCALE,
        0,
        1,
    )

    return round(float(confidence), 3)


def build_model() -> RandomForestRegressor:
    """Construct the baseline Random Forest model."""

    return RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )


class ForecastingAgent:
    """Train and evaluate an independent model for each security."""

    def __init__(
        self,
        db_path: str = "asx_market_data.db",
        min_training_rows: int = 100,
    ):
        if min_training_rows < 10:
            raise ValueError(
                "min_training_rows must be at least 10."
            )

        self.db_path = db_path
        self.min_training_rows = min_training_rows
        self.conn = duckdb.connect(db_path)

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        """Retrieve historical equity observations."""

        return self.conn.execute(
            """
            SELECT
                date,
                open,
                high,
                low,
                close,
                volume,
                daily_return
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date ASC
            """,
            (ticker,),
        ).df()

    def _load_macro(self) -> pd.DataFrame:
        """Retrieve historical ASX 200 observations."""

        return self.conn.execute(
            """
            SELECT date, close, daily_return
            FROM macro
            WHERE ticker = '^AXJO'
            ORDER BY date ASC
            """
        ).df()

    def _load_watchlist(self) -> list[str]:
        """Retrieve securities available in the database."""

        rows = self.conn.execute(
            """
            SELECT DISTINCT ticker
            FROM ohlcv
            ORDER BY ticker
            """
        ).fetchall()

        return [row[0] for row in rows]

    def _train_and_predict(
        self,
        ticker: str,
        macro: pd.DataFrame,
    ) -> StockForecast | None:
        """Evaluate a model and forecast the next observation."""

        logger.info("Training model for %s", ticker)

        prices = self._load_prices(ticker)

        if len(prices) < self.min_training_rows:
            logger.warning(
                "Insufficient price history for %s",
                ticker,
            )
            return None

        features = engineer_features(prices, macro)

        # Preserve the latest observation before filtering targets.
        latest = features.iloc[[-1]].copy()

        if latest[FEATURE_COLS].isna().any().any():
            logger.warning(
                "Incomplete latest inference features for %s",
                ticker,
            )
            return None

        labelled = features.dropna(
            subset=FEATURE_COLS + ["target"]
        )

        if len(labelled) < self.min_training_rows:
            logger.warning(
                "Insufficient labelled observations for %s: %d",
                ticker,
                len(labelled),
            )
            return None

        X = labelled[FEATURE_COLS].to_numpy(dtype=float)
        y = labelled["target"].to_numpy(dtype=float)

        # Chronological holdout with a one-observation purge
        split = int(len(labelled) * 0.8)

        X_train = X[:split - 1]
        y_train = y[:split - 1]

        X_validation = X[split:]
        y_validation = y[split:]

        if len(X_train) == 0 or len(X_validation) == 0:
            logger.warning(
                "Insufficient validation data for %s",
                ticker,
            )
            return None

        validation_model = build_model()

        validation_model.fit(
            X_train,
            y_train,
        )

        validation_predictions = validation_model.predict(
            X_validation
        )

        mae = float(
            mean_absolute_error(
                y_validation,
                validation_predictions,
            )
        )

        baseline_mae = float(
            mean_absolute_error(
                y_validation,
                np.zeros_like(y_validation),
            )
        )

        logger.info(
            "%s validation MAE: %.2f%% | "
            "zero-return baseline MAE: %.2f%%",
            ticker,
            mae * 100,
            baseline_mae * 100,
        )

        # Retrain using all available labelled observations.
        final_model = build_model()
        final_model.fit(X, y)

        # Predict from the latest unlabelled observation.
        X_latest = latest[
            FEATURE_COLS
        ].to_numpy(dtype=float)

        predicted_return = float(
            final_model.predict(X_latest)[0]
        )

        confidence = derive_confidence(
            final_model,
            X_latest,
        )

        signal = derive_signal(predicted_return)

        importance = {
            feature: float(value)
            for feature, value in zip(
                FEATURE_COLS,
                final_model.feature_importances_,
            )
        }

        top_features = dict(
            sorted(
                importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        )

        observation_date = (
            pd.Timestamp(latest["date"].iloc[0])
            .date()
            .isoformat()
        )

        forecast = StockForecast(
            ticker=ticker,
            forecast_date=observation_date,
            predicted_return=predicted_return,
            confidence=confidence,
            mae=mae,
            signal=signal,
            feature_importance=top_features,
            baseline_mae=baseline_mae,
        )

        logger.info(
            "%s | as of %s | predicted return %+.2f%% | "
            "confidence %.0f%% | signal %s",
            ticker,
            observation_date,
            predicted_return * 100,
            confidence * 100,
            signal,
        )

        return forecast

    def run(
        self,
        tickers: list[str] | None = None,
    ) -> list[StockForecast]:
        """Generate forecasts for the selected securities."""

        macro = self._load_macro()

        if macro.empty:
            logger.error(
                "ASX 200 benchmark history is unavailable."
            )
            return []

        watchlist = (
            self._load_watchlist()
            if tickers is None
            else sorted(set(tickers))
        )

        forecasts: list[StockForecast] = []

        for ticker in watchlist:
            try:
                forecast = self._train_and_predict(
                    ticker,
                    macro,
                )

                if forecast is not None:
                    forecasts.append(forecast)

            except Exception:
                logger.exception(
                    "Forecasting failed for %s",
                    ticker,
                )

        logger.info(
            "Forecasting complete: %d/%d successful",
            len(forecasts),
            len(watchlist),
        )

        return forecasts

    def get_forecast_summary(
        self,
        forecasts: list[StockForecast],
    ) -> pd.DataFrame:
        """Return formatted forecast and validation diagnostics."""

        rows = []

        for forecast in forecasts:
            baseline = forecast.baseline_mae

            rows.append(
                {
                    "ticker": forecast.ticker,
                    "as_of_date": forecast.as_of_date,
                    "predicted_return": (
                        f"{forecast.predicted_return:+.2%}"
                    ),
                    "confidence": (
                        f"{forecast.confidence:.0%}"
                    ),
                    "mae": f"{forecast.mae:.2%}",
                    "baseline_mae": (
                        f"{baseline:.2%}"
                        if baseline is not None
                        else "N/A"
                    ),
                    "signal": forecast.signal,
                    "top_feature": next(
                        iter(forecast.feature_importance),
                        None,
                    ),
                }
            )

        return pd.DataFrame(rows)

    def close(self) -> None:
        """Close the database connection."""

        self.conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(name)s | "
            "%(levelname)s | %(message)s"
        ),
    )

    agent = ForecastingAgent()

    try:
        forecasts = agent.run()

        print("\nForecast Summary")
        print(
            agent.get_forecast_summary(forecasts)
            .to_string(index=False)
        )

    finally:
        agent.close()