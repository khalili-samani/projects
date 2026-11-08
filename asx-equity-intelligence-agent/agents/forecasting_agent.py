"""
Forecasting Agent
=================
Responsibility: Read OHLCV data from the database, engineer features,
train a model per stock, and predict next-day returns.

Design decisions:
- Random Forest Regressor as baseline (robust, interpretable, handles
  non-linear relationships common in financial time series)
- Features engineered from price/volume history — no raw prices fed
  directly into the model (non-stationary and misleading for ML)
- One model trained per stock (each stock has different dynamics)
- Walk-forward validation to avoid lookahead bias — a common mistake
  in financial ML where future data leaks into training
- Confidence score derived from the spread of individual tree predictions
"""

import logging
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ForecastingAgent")


# ---------------------------------------------------------------------------
# Output model — what this agent returns to the orchestrator
# ---------------------------------------------------------------------------
@dataclass
class StockForecast:
    ticker: str
    forecast_date: str          # the date we're predicting FOR (tomorrow)
    predicted_return: float     # e.g. 0.008 means +0.8%
    confidence: float           # 0 to 1 — derived from tree prediction spread
    mae: float                  # mean absolute error on validation set
    signal: str                 # BUY / HOLD / REDUCE — derived from predicted return
    feature_importance: dict    # which features drove the prediction


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw OHLCV data into ML features.

    Key principle: all features are derived from data available BEFORE
    the prediction date. No future data can touch the feature set —
    this is called avoiding lookahead bias.

    prices: DataFrame with columns [date, open, high, low, close, volume, daily_return]
    macro:  DataFrame with columns [date, close, daily_return] for ASX 200
    """
    df = prices.copy().sort_values("date").reset_index(drop=True)

    # --- Momentum features ---
    # How much has the stock moved over different windows?
    df["return_5d"]  = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    # --- Volatility ---
    # Rolling standard deviation of daily returns — how choppy is the stock?
    df["volatility_20d"] = df["daily_return"].rolling(20).std()

    # --- RSI (Relative Strength Index) ---
    # Classic momentum oscillator. Above 70 = overbought, below 30 = oversold.
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Volume ratio ---
    # Today's volume vs 20-day average. High ratio = unusual activity.
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # --- Distance from 52-week high and low ---
    # Mean reversion signal — stocks far from highs may revert.
    df["dist_52w_high"] = df["close"] / df["close"].rolling(252, min_periods=1).max() - 1
    df["dist_52w_low"]  = df["close"] / df["close"].rolling(252, min_periods=1).min() - 1

    # --- Moving average crossover ---
    # When short MA crosses above long MA, often a bullish signal.
    df["ma_10"] = df["close"].rolling(10, min_periods=1).mean()
    df["ma_50"] = df["close"].rolling(50, min_periods=1).mean()
    df["ma_crossover"] = df["ma_10"] / df["ma_50"] - 1

    # --- Macro context ---
    # Merge ASX 200 returns so the model knows what the broad market is doing
    if macro is not None and not macro.empty:
        macro_df = macro[["date", "daily_return"]].rename(
            columns={"daily_return": "asx200_return"}
        )
        macro_df["asx200_return_5d"] = macro_df["asx200_return"].rolling(5).mean()
        df = df.merge(macro_df, on="date", how="left")
    else:
        df["asx200_return"]    = 0.0
        df["asx200_return_5d"] = 0.0

    # --- Target variable ---
    # What we're predicting: tomorrow's return.
    # We shift by -1 so each row's target is the NEXT row's daily_return.
    df["target"] = df["daily_return"].shift(-1)

    return df


# ---------------------------------------------------------------------------
# Feature columns — what gets fed into the model
# ---------------------------------------------------------------------------
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


def derive_signal(predicted_return: float) -> str:
    """
    Convert a predicted return into a trading signal.

    Thresholds are deliberately conservative — we only signal
    BUY or REDUCE when the predicted move is meaningful.
    """
    if predicted_return > 0.005:    # > +0.5%
        return "BUY"
    elif predicted_return < -0.005: # < -0.5%
        return "REDUCE"
    else:
        return "HOLD"


def derive_confidence(model: RandomForestRegressor, X: np.ndarray) -> float:
    """
    Estimate confidence from the spread of individual tree predictions.

    A Random Forest is an ensemble of many decision trees. If all trees
    agree on the prediction, confidence is high. If they disagree widely,
    confidence is low.

    We invert and normalise the standard deviation so that:
    - Low spread → high confidence (closer to 1.0)
    - High spread → low confidence (closer to 0.0)
    """
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    std = tree_predictions.std()
    # Normalise: std of ~0 → confidence 1.0, std of ~0.02 → confidence ~0.0
    confidence = float(np.clip(1 - (std / 0.02), 0, 1))
    return round(confidence, 3)


# ---------------------------------------------------------------------------
# Forecasting Agent
# ---------------------------------------------------------------------------
class ForecastingAgent:
    """
    Trains one Random Forest model per stock and predicts tomorrow's return.

    Walk-forward validation:
    - We train on the first 80% of history
    - Validate on the remaining 20%
    - Then retrain on ALL data to make the final prediction

    This mimics how the model would actually be used in production —
    you can only train on data you had at the time.
    """

    def __init__(
        self,
        db_path: str = "asx_market_data.db",
        min_training_rows: int = 100,   # need at least 100 days to train
    ):
        self.db_path = db_path
        self.min_training_rows = min_training_rows
        self.conn = duckdb.connect(db_path)

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT date, open, high, low, close, volume, daily_return
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date ASC
        """, (ticker,)).df()

    def _load_macro(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT date, close, daily_return
            FROM macro
            WHERE ticker = '^AXJO'
            ORDER BY date ASC
        """).df()

    def _load_watchlist(self) -> list[str]:
        result = self.conn.execute("""
            SELECT DISTINCT ticker FROM ohlcv
        """).df()
        return result["ticker"].tolist()

    def _train_and_predict(self, ticker: str, macro: pd.DataFrame) -> StockForecast | None:
        """
        For a single ticker:
        1. Load and engineer features
        2. Walk-forward train/validate split
        3. Retrain on full dataset
        4. Predict tomorrow's return
        """
        logger.info(f"Training model for {ticker}...")

        prices = self._load_prices(ticker)
        if len(prices) < self.min_training_rows:
            logger.warning(f"  Insufficient data for {ticker} ({len(prices)} rows)")
            return None

        # Engineer features
        df = engineer_features(prices, macro)

        # Drop rows with NaN features or target
        df = df.dropna(subset=FEATURE_COLS + ["target"])

        if len(df) < self.min_training_rows:
            logger.warning(f"  Insufficient clean rows for {ticker} after feature engineering")
            return None

        X = df[FEATURE_COLS].values
        y = df["target"].values

        # --- Walk-forward split ---
        # Train on first 80%, validate on last 20%
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Scale features — Random Forest doesn't strictly need this but
        # it helps with numerical stability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)

        # --- Train on 80% ---
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,            # prevent overfitting
            min_samples_leaf=5,     # each leaf needs at least 5 samples
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # --- Validate ---
        val_predictions = model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, val_predictions)
        logger.info(f"  Validation MAE: {mae:.4f} ({mae*100:.2f}%)")

        # --- Retrain on full dataset for final prediction ---
        X_all_scaled = scaler.fit_transform(X)
        model.fit(X_all_scaled, y)

        # --- Predict tomorrow ---
        # The last row of features represents today — we predict its target (tomorrow)
        X_today = X_all_scaled[-1].reshape(1, -1)
        predicted_return = float(model.predict(X_today)[0])
        confidence = derive_confidence(model, X_today)
        signal = derive_signal(predicted_return)

        # --- Feature importance ---
        importance = dict(zip(
            FEATURE_COLS,
            [round(v, 4) for v in model.feature_importances_]
        ))
        top_features = dict(sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )[:3])

        forecast = StockForecast(
            ticker=ticker,
            forecast_date=str(
                pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
            )[:10],
            predicted_return=round(predicted_return, 6),
            confidence=confidence,
            mae=round(mae, 6),
            signal=signal,
            feature_importance=top_features,
        )

        logger.info(
            f"  ✓ {ticker}: predicted {predicted_return*100:+.2f}% | "
            f"confidence {confidence:.0%} | signal {signal}"
        )
        return forecast

    def run(self) -> list[StockForecast]:
        """
        Main entry point. Trains and predicts for every ticker in the DB.
        Returns a list of StockForecast objects for the orchestrator.
        """
        logger.info("=" * 60)
        logger.info("Forecasting Agent — starting run")
        logger.info("=" * 60)

        macro = self._load_macro()
        tickers = self._load_watchlist()
        logger.info(f"Tickers to forecast: {len(tickers)}")

        forecasts = []
        for ticker in tickers:
            forecast = self._train_and_predict(ticker, macro)
            if forecast:
                forecasts.append(forecast)

        logger.info("=" * 60)
        logger.info(f"Forecasting complete. {len(forecasts)}/{len(tickers)} successful.")
        logger.info("=" * 60)

        return forecasts

    def get_forecast_summary(self, forecasts: list[StockForecast]) -> pd.DataFrame:
        """
        Returns a clean summary DataFrame for the orchestrator to log
        and pass downstream to the Risk and Strategy agents.
        """
        rows = []
        for f in forecasts:
            rows.append({
                "ticker":           f.ticker,
                "forecast_date":    f.forecast_date,
                "predicted_return": f"{f.predicted_return*100:+.2f}%",
                "confidence":       f"{f.confidence:.0%}",
                "mae":              f"{f.mae*100:.2f}%",
                "signal":           f.signal,
                "top_feature":      list(f.feature_importance.keys())[0],
            })
        return pd.DataFrame(rows)

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Run directly for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = ForecastingAgent()
    forecasts = agent.run()

    print("\n📈 Forecast Summary:")
    print(agent.get_forecast_summary(forecasts).to_string(index=False))
    agent.close()