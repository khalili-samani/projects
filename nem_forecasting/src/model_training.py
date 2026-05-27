"""
Model training and evaluation.

Trains an XGBoost regressor on the engineered feature table using a
chronological 70/15/15 train/validation/test split to avoid look-ahead bias.
"""

import joblib
import pandas as pd
import mysql.connector

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.db_utils import get_engine
from src.config import (
    DB_CONFIG,
    FORECAST_TARGET,
    MODEL_FEATURES,
    MIN_TRAINING_ROWS,
    XGB_PARAMS,
)

MODEL_OUTPUT_PATH = Path("models/xgboost_nem_price_model.joblib")


def train_and_export_model():
    print("[Training] Loading engineered features...")

    query = """
        SELECT *
        FROM engineered_features
        ORDER BY settlement_date, region_id;
    """

    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(query, get_engine())
    conn.close()

    if len(df) < MIN_TRAINING_ROWS:
        print(
            f"[Abort] Only {len(df)} rows available — need at least "
            f"{MIN_TRAINING_ROWS} to train. Run the ingestion pipeline first."
        )
        return

    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df.sort_values(["settlement_date", "region_id"], inplace=True)
    df.dropna(subset=MODEL_FEATURES + [FORECAST_TARGET], inplace=True)

    X = df[MODEL_FEATURES]
    y = df[FORECAST_TARGET]

    # Chronological split — never shuffle time-series data
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
    X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

    print(
        f"[Training] Split — train: {len(X_train)}, "
        f"valid: {len(X_valid)}, test: {len(X_test)} rows."
    )

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    # root_mean_squared_error is available from sklearn >= 1.4;
    # fall back to the squared=False kwarg for older versions.
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(y_test, predictions)
    except ImportError:
        rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print("\n===== Forecast Evaluation (hold-out test set) =====")
    print(f"  MAE:  {mae:.2f} $/MWh")
    print(f"  RMSE: {rmse:.2f} $/MWh")
    print(f"  R²:   {r2:.4f}")
    print("====================================================\n")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": model,
        "features": MODEL_FEATURES,
        "target": FORECAST_TARGET,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
    }

    joblib.dump(model_bundle, MODEL_OUTPUT_PATH)
    print(f"[Training] Model saved to {MODEL_OUTPUT_PATH}.")