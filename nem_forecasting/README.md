# Australian NEM Price Forecasting

An end-to-end machine learning pipeline that ingests live and historical data from the Australian Energy Market Operator (AEMO) and the Bureau of Meteorology (BOM), engineers leakage-safe features, and trains an XGBoost model to forecast National Electricity Market (NEM) dispatch prices 30 minutes ahead. A Streamlit dashboard lets you explore historical market data and run live what-if scenarios.

---

## Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Dashboard](#running-the-dashboard)
- [Feature Engineering](#feature-engineering)
- [Model](#model)
- [Extending to Other Regions](#extending-to-other-regions)
- [Known Limitations](#known-limitations)

---

## Architecture

```
NEMWeb Archive (AEMO)      NEMWeb CURRENT (AEMO)      BOM Weather JSON
        │                          │                          │
        └──────────────────────────┴──────────────────────────┘
                                   │
                          data_ingestion.py
                                   │
                          data_cleaning.py
                    (price filter · merge_asof · climate fill)
                                   │
                       feature_engineering.py
                   (lags · rolling means · cyclical encodings)
                                   │
                          model_training.py
                    (XGBoost · chronological 70/15/15 split)
                                   │
                  models/xgboost_nem_price_model.joblib
                                   │
                               app.py
                         (Streamlit dashboard)
```

All intermediate data lives in a MySQL database. The pipeline is fully idempotent — every insert uses `ON DUPLICATE KEY UPDATE`, so re-runs never create duplicates.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| MySQL | ≥ 8.0 |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/nem-forecasting.git
cd nem-forecasting

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your MySQL credentials

# 5. First run — pull 3 months of historical archive data and train
python run_pipeline.py --historical

# 6. Subsequent runs — pull latest live data and retrain
python run_pipeline.py

# 7. Launch the dashboard
streamlit run app.py
```

---

## Project Structure

```
nem-forecasting/
├── app.py                      ← Streamlit dashboard (run from root)
├── run_pipeline.py             ← Pipeline runner (run from root)
├── setup.py
├── requirements.txt
├── .env.example                ← Template for environment variables
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py               ← All constants and configuration
│   ├── db_utils.py             ← SQLAlchemy engine for pandas reads
│   ├── db_initialiser.py       ← Creates MySQL schema
│   ├── data_ingestion.py       ← Downloads AEMO + BOM data
│   ├── data_cleaning.py        ← Filters and aligns raw data
│   ├── feature_engineering.py  ← Builds ML feature matrix
│   └── model_training.py       ← Trains and exports XGBoost model
│
├── sql/
│   └── create_tables.sql       ← MySQL schema definition
│
└── models/
    └── .gitkeep                ← Trained model saved here (gitignored)
```

---

## Configuration

All configuration lives in `src/config.py` and is driven by environment variables loaded from `.env`.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | MySQL host |
| `DB_PORT` | `3306` | MySQL port |
| `DB_USER` | `root` | MySQL user |
| `DB_PASSWORD` | _(empty)_ | MySQL password |
| `DB_NAME` | `nem_forecasting` | Database name |
| `AEMO_ARCHIVE_MONTHS` | Last 3 completed months | Comma-separated YYYYMM list, e.g. `202602,202603,202604` |
| `AEMO_ARCHIVE_DAYS_PER_MONTH` | `30` | Days to sample per archive month |
| `WEATHER_MERGE_TOLERANCE` | `24h` | Max gap between NEM and BOM timestamps |

### Key constants in `config.py`

| Constant | Default | Description |
|---|---|---|
| `FORECAST_HORIZON_STEPS` | `6` | Intervals ahead to forecast (6 × 5 min = 30 min) |
| `MIN_TRAINING_ROWS` | `500` | Minimum rows required before training |
| `BOM_STATIONS` | VIC1 / Melbourne Olympic Park | BOM station → NEM region mapping |
| `XGB_PARAMS` | See file | XGBoost hyperparameters |

---

## Running the Pipeline

### First run (fresh database)

```bash
python run_pipeline.py --historical
```

This pulls the last 3 completed months of AEMO archive data (~130,000 rows) before training. The archive lives at `https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/` as daily zip files, each containing 288 five-minute interval zips.

### Subsequent runs

```bash
python run_pipeline.py
```

Pulls the latest ~100 live dispatch files from NEMWeb CURRENT and retrains the model on the accumulated database.

### Pipeline stages

1. **Initialise** — creates the MySQL database and tables if they don't exist
2. **Ingest AEMO** — downloads and parses DispatchIS zip files (archive or live)
3. **Ingest BOM** — fetches current observations from the configured BOM JSON endpoint
4. **Clean** — filters prices to the valid dispatch range (−$1,000 to $20,000/MWh), aligns NEM and weather data via `merge_asof`, and fills any missing temperature values using Melbourne monthly climate averages
5. **Feature engineering** — builds lag features, rolling means, and cyclical time encodings; creates the forecast target
6. **Train** — fits XGBoost on a chronological 70/15/15 split and saves the model bundle

---

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard provides:

- **Market history chart** — demand and dispatch price over time, filterable by NEM region
- **Recent feature rows** — raw table view of the latest engineered data
- **Scenario forecasting** — adjust temperature, demand, and lagged price via sidebar controls and see the model's price prediction update instantly
- **Spike risk indicator** — flags forecasts above $300/MWh

Model hold-out metrics (MAE, RMSE, R²) are displayed alongside the forecast so you always know how much to trust it.

---

## Feature Engineering

Features are engineered in `src/feature_engineering.py`. All lags are shifted by at least `FORECAST_HORIZON_STEPS` intervals (6 × 5 min = 30 min) to guarantee no future information leaks into training.

| Feature | Description |
|---|---|
| `temperature_c` | BOM air temperature at the mapped station (or Melbourne monthly mean as fallback) |
| `hour`, `day_of_week`, `month` | Calendar features |
| `is_weekend` | Binary weekend flag |
| `hour_sin`, `hour_cos` | Cyclical hour encoding |
| `day_sin`, `day_cos` | Cyclical day-of-week encoding |
| `demand_lag_1h` | Demand 30 min ago (leakage-safe) |
| `demand_lag_24h` | Demand 24 hours ago |
| `price_lag_1h` | Dispatch price 30 min ago (leakage-safe) |
| `price_lag_24h` | Dispatch price 24 hours ago |
| `demand_rolling_mean_4h` | 4-hour rolling mean of demand |
| `price_rolling_mean_4h` | 4-hour rolling mean of price |

**Target:** `target_price_mwh` — dispatch price `FORECAST_HORIZON_STEPS` intervals (30 min) in the future.

---

## Model

- **Algorithm:** XGBoost (`reg:squarederror`)
- **Split:** Chronological 70 % train / 15 % validation / 15 % test — data is never shuffled
- **Validation:** Validation set passed to `eval_set` for early stopping
- **Output:** A `joblib` bundle containing the trained model, feature list, target name, and hold-out metrics

### Baseline results (3 months of archive data, VIC1 + 4 other regions)

| Metric | Value |
|---|---|
| MAE | $13.19 /MWh |
| RMSE | $23.57 /MWh |
| R² | 0.727 |

The model bundle is loaded by the dashboard at startup and cached with `@st.cache_resource`.

---

## Extending to Other Regions

The pipeline runs across all 5 NEM regions (NSW1, QLD1, SA1, TAS1, VIC1) automatically from the AEMO data. To add real weather for additional regions:

1. Find the relevant BOM station and JSON endpoint at [bom.gov.au](https://www.bom.gov.au)
2. Add an entry to `BOM_STATIONS` in `src/config.py`:

```python
BOM_STATIONS = {
    "VIC1": {
        "station_name": "Melbourne Olympic Park",
        "url": "https://www.bom.gov.au/fwo/IDV60901/IDV60901.95936.json",
    },
    "NSW1": {
        "station_name": "Sydney Olympic Park",
        "url": "https://www.bom.gov.au/fwo/IDN60901/IDN60901.95765.json",
    },
}
```

3. Add the station → region mapping to `sql/create_tables.sql`:

```sql
INSERT INTO weather_station_region_map (station_name, region_id)
VALUES ('Sydney Olympic Park', 'NSW1')
ON DUPLICATE KEY UPDATE region_id = VALUES(region_id);
```

4. Re-run `python run_pipeline.py --historical`.

Regions without a mapped BOM station fall back to Melbourne monthly climate averages for temperature, so they still train and forecast correctly.

---

## Known Limitations

- **Weather coverage:** Only VIC1 has a real BOM station configured. All other regions use Melbourne monthly climate means as a temperature proxy, which reduces feature accuracy for those regions.
- **Price spikes:** Extreme price events (e.g. $15,000+/MWh during heatwaves) are rare and hard to forecast with a single-model approach. Consider adding renewable generation and interconnector flow features for better spike capture.
- **Model artefacts:** The `.joblib` file is gitignored due to size. Re-train by running the pipeline, or use Git LFS / DVC if you want to version model files.
- **AEMO format changes:** The DispatchIS CSV parser targets `DISPATCH/PRICE` and `DISPATCH/REGIONSUM` tables. If AEMO changes the MMS export format, `_extract_dispatch_region_rows` in `data_ingestion.py` will need updating. Run `debug_aemo.py` to inspect the current format.

---

## Data Sources

- **AEMO NEMWeb** — [nemweb.com.au](https://nemweb.com.au) (open access, no API key required)
- **Bureau of Meteorology** — [bom.gov.au](https://www.bom.gov.au) (open access JSON observations)

---

## Licence

MIT