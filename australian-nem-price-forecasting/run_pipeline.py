"""
Unified forecasting pipeline runner.

Run from the project root:
    python run_pipeline.py

For a fresh environment with no historical data, run with --historical first:
    python run_pipeline.py --historical

This pulls the last 3 months of archive data (~500+ rows) before training.
Subsequent runs can use the default (current live files only).
"""

import sys

from src.db_initialiser import initialise_database
from src.data_ingestion import (
    ingest_aemo_current_month,
    ingest_aemo_historical,
    ingest_bom_weather,
)
from src.data_cleaning import clean_raw_observations
from src.feature_engineering import run_feature_pipeline
from src.model_training import train_and_export_model


def execute_system_pipeline(historical=False):
    print("\n====================================")
    print("   NEM Forecasting Pipeline Start")
    print("====================================\n")

    initialise_database()

    if historical:
        print("[Pipeline] Running historical archive ingestion...")
        ingest_aemo_historical()
    else:
        ingest_aemo_current_month()

    ingest_bom_weather()
    clean_raw_observations()
    run_feature_pipeline()
    train_and_export_model()

    print("\n====================================")
    print("   Pipeline Completed Successfully")
    print("====================================\n")


if __name__ == "__main__":
    historical = "--historical" in sys.argv
    execute_system_pipeline(historical=historical)