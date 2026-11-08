# Australian NEM Price Forecasting

A Python-based forecasting project that predicts Australian National Electricity Market dispatch prices 30 minutes ahead using AEMO market data, weather inputs, time-series feature engineering, XGBoost, MySQL, and Streamlit.

The repository demonstrates how raw electricity-market files can be ingested, validated, transformed into leakage-aware forecasting features, used to train a chronological hold-out model, and exposed through an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![MySQL](https://img.shields.io/badge/Database-MySQL-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project summary

Electricity dispatch prices in the National Electricity Market can change rapidly in response to demand, generation availability, network constraints, weather, and other market conditions.

This project investigates whether recent demand and price behaviour, calendar patterns, and temperature information can provide a useful 30-minute-ahead price estimate.

The implemented workflow:

1. downloads current or historical AEMO DispatchIS files;
2. extracts regional dispatch prices and total demand;
3. stores raw observations in MySQL;
4. retrieves available Bureau of Meteorology temperature observations;
5. cleans and aligns market and weather data;
6. creates lagged, rolling, calendar, and cyclical features;
7. trains an XGBoost regression model using chronological data splits;
8. evaluates the model on a hold-out test set; and
9. serves forecasts and market history through Streamlit.

This is a portfolio and analytical prototype. It is not intended for live bidding, trading, or operational decision-making without further validation.

---

## Verified model results

The supplied model artefact contains the following hold-out test metrics:

| Metric | Result |
|---|---:|
| Forecast horizon | 30 minutes |
| Model | XGBoost regressor |
| Mean absolute error | $19.02/MWh |
| Root mean squared error | $33.43/MWh |
| R² | 0.5085 |

These metrics were calculated after a chronological 70% training, 15% validation, and 15% test split.

The positive R² indicates that the model captures part of the variation in the hold-out target. The difference between MAE and RMSE also suggests that some observations have substantially larger errors than the typical forecast.

The repository does not currently preserve the dates, row counts, regional breakdown, target distribution, predictions, or baseline results associated with the saved test run. As a result, the metrics should not be interpreted as a complete assessment of production forecasting performance.

---

## What this project demonstrates

### Data engineering

- Parsing nested AEMO NEMWeb ZIP archives and MMS-format CSV files
- Combining dispatch price and regional demand records
- MySQL schema design with composite primary keys and supporting indexes
- Idempotent database writes using upsert operations
- Historical and current-file ingestion modes
- Pipeline orchestration across ingestion, cleaning, feature engineering, and training

### Time-series machine learning

- Chronological train, validation, and test splitting
- Region-specific lag and rolling-window calculations
- Forward forecasting targets
- Leakage-aware feature shifting
- Cyclical encoding of calendar variables
- XGBoost model training and hold-out evaluation
- Persisting a model together with feature names and evaluation metrics

### Application development

- Loading a persisted model into Streamlit
- Reading recent feature data from MySQL
- Interactive scenario inputs
- Presenting market history, model metrics, forecasts, and a simple price-risk classification

---

## System architecture

```text
AEMO NEMWeb DispatchIS files
              │
              ▼
     src/data_ingestion.py
              │
              ├──────────────► raw_nem_data
              │
BOM observations
              │
              └──────────────► raw_weather_data
                                      │
                                      ▼
                           src/data_cleaning.py
                                      │
                                      ▼
                         cleaned_nem_weather_data
                                      │
                                      ▼
                        src/feature_engineering.py
                                      │
                                      ▼
                           engineered_features
                                      │
                                      ▼
                          src/model_training.py
                                      │
                                      ▼
                 models/xgboost_nem_price_model.joblib
                                      │
                                      ▼
                                   app.py
````

MySQL is used as the shared storage layer between pipeline stages.

---

## Repository structure

```text
nem_forecasting/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── db_utils.py
│   ├── db_initialiser.py
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── model_training.py
│
├── sql/
│   └── create_tables.sql
│
├── models/
│   └── xgboost_nem_price_model.joblib
│
└── outputs/
    └── screenshots/
```

| Path                         | Responsibility                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| `run_pipeline.py`            | Runs database initialisation, ingestion, cleaning, feature engineering, and model training |
| `src/config.py`              | Database settings, source URLs, feature definitions, archive options, and model parameters |
| `src/data_ingestion.py`      | Downloads and parses AEMO and BOM data                                                     |
| `src/data_cleaning.py`       | Filters prices and aligns weather observations to market intervals                         |
| `src/feature_engineering.py` | Builds model features and the forward price target                                         |
| `src/model_training.py`      | Trains, evaluates, and serialises the XGBoost model                                        |
| `sql/create_tables.sql`      | Defines the MySQL database schema                                                          |
| `app.py`                     | Runs the Streamlit forecasting interface                                                   |
| `models/`                    | Stores the serialised model bundle                                                         |

---

## Data sources

### AEMO DispatchIS

The pipeline retrieves DispatchIS reports from the Australian Energy Market Operator's NEMWeb service.

Two ingestion modes are implemented:

* **Current ingestion:** processes up to the latest 100 ZIP files listed in the current DispatchIS directory.
* **Historical ingestion:** downloads selected daily archives from configured months.

The parser handles AEMO's nested archive structure and extracts:

| Field                | Description                 |
| -------------------- | --------------------------- |
| `settlement_date`    | Dispatch interval timestamp |
| `region_id`          | NEM region identifier       |
| `demand_mw`          | Regional total demand       |
| `dispatch_price_mwh` | Regional reference price    |

Rows are stored in `raw_nem_data` using `(settlement_date, region_id)` as the primary key.

### Bureau of Meteorology observations

The current configuration retrieves air-temperature observations for:

| NEM region | Weather station        |
| ---------- | ---------------------- |
| `VIC1`     | Melbourne Olympic Park |

The live BOM feed has limited historical coverage. During cleaning:

* a backward time-based join is attempted within the configured tolerance;
* short gaps may be forward-filled; and
* remaining missing temperatures are replaced with hard-coded Melbourne monthly average temperatures.

This means temperature is a limited proxy rather than complete region-specific historical weather coverage. In particular, the current implementation does not provide dedicated live stations for every NEM region.

---

## Data cleaning

The cleaning stage:

1. loads raw market and weather records from MySQL;
2. removes market prices outside the configured range of `-$1,000/MWh` to `$20,000/MWh`;
3. groups market data by region;
4. aligns observations to the most recent available weather record;
5. fills unresolved temperature gaps using Melbourne monthly averages; and
6. writes the results to `cleaned_nem_weather_data`.

The price bounds are implemented as validation filters in the project. They should not be assumed to represent the applicable market price floor and cap for every historical period without separate verification.

---

## Forecast target and features

### Target

The target is the regional dispatch price six five-minute intervals into the future:

```text
target_price_mwh(t) = dispatch_price_mwh(t + 30 minutes)
```

The target is calculated separately within each region.

### Model features

| Category               | Features                                          |
| ---------------------- | ------------------------------------------------- |
| Weather                | `temperature_c`                                   |
| Calendar               | `hour`, `day_of_week`, `month`, `is_weekend`      |
| Cyclical time encoding | `hour_sin`, `hour_cos`, `day_sin`, `day_cos`      |
| Recent demand          | `demand_lag_1h`, `demand_lag_24h`                 |
| Recent price           | `price_lag_1h`, `price_lag_24h`                   |
| Rolling statistics     | `demand_rolling_mean_4h`, `price_rolling_mean_4h` |

Despite their current column names, `demand_lag_1h` and `price_lag_1h` are shifted by six five-minute intervals in the implementation. They therefore represent values from approximately **30 minutes earlier**, not one hour earlier.

The 24-hour features use a 48-row shift. This corresponds to 24 hours only when the observations are interpreted at a 30-minute frequency. The ingested DispatchIS source is described in the code as five-minute data, so this naming and interval assumption should be corrected before relying on those features as true 24-hour lags.

Rolling means are calculated over eight rows after applying the short lag. The current names describe them as four-hour means, but their actual time duration depends on the effective interval frequency of the stored observations.

These naming inconsistencies do not prevent the code from running, but they make the temporal meaning of several features ambiguous. A future revision should define all lag and rolling windows from explicit timedeltas rather than fixed row counts.

### Leakage controls

Feature calculations are performed separately for each region and shifted before the target row is evaluated. This prevents contemporaneous or future price and demand observations from being inserted directly into the lagged features.

The model-training dataframe is then sorted chronologically and is not shuffled.

---

## Model training

The model is an `XGBRegressor` configured with:

| Parameter            |              Value |
| -------------------- | -----------------: |
| Objective            | `reg:squarederror` |
| Evaluation metric    |               RMSE |
| Number of estimators |                700 |
| Learning rate        |               0.03 |
| Maximum tree depth   |                  6 |
| Row subsampling      |                0.8 |
| Column subsampling   |                0.8 |
| Tree method          |             `hist` |
| Random seed          |                 42 |

The prepared observations are split by row order:

```text
70% training
15% validation
15% test
```

The validation set is passed to XGBoost during fitting. Early stopping is not currently configured, so the validation data does not determine the final number of trees.

One pooled model is trained across all regions. Although lag features are calculated within each region, `region_id` is not included as a model feature. The model therefore cannot directly learn a region-specific intercept or categorical regional effect.

The saved Joblib bundle contains:

```python
{
    "model": model,
    "features": MODEL_FEATURES,
    "target": "target_price_mwh",
    "metrics": {
        "mae": ...,
        "rmse": ...,
        "r2": ...
    }
}
```

---

## Dashboard

The Streamlit application:

* loads up to 3,000 recent engineered-feature rows from MySQL;
* allows the user to select an available NEM region;
* plots recent demand and dispatch-price history;
* displays recent feature records;
* shows the saved hold-out metrics;
* lets the user modify temperature, recent demand, and recent price inputs;
* generates a forecast using the saved model; and
* labels forecasts at or above `$300/MWh` as high spike risk.

The scenario tool changes selected inputs while retaining the remaining values from the latest database row. It is therefore a local sensitivity interface, not a complete simulation of future market conditions.

The spike-risk threshold is a dashboard rule rather than a separately trained classification model.

---

## Installation

### Prerequisites

* Python 3
* MySQL
* Internet access to the configured AEMO and BOM endpoints
* A MySQL user that can create the `nem_forecasting` database and its tables

### Clone the repository

```bash
git clone <repository-url>
cd nem_forecasting
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the database

Create a `.env` file in the project root:

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=nem_forecasting
```

Optional historical-ingestion settings:

```dotenv
AEMO_ARCHIVE_MONTHS=202603,202604,202605
AEMO_ARCHIVE_DAYS_PER_MONTH=30
WEATHER_MERGE_TOLERANCE=24h
```

When `AEMO_ARCHIVE_MONTHS` is not supplied, the application selects the previous three completed calendar months at runtime.

---

## Running the pipeline

### Initial historical run

```bash
python run_pipeline.py --historical
```

This command:

1. creates the database and tables if they do not exist;
2. downloads configured AEMO historical archives;
3. retrieves the configured BOM observations;
4. cleans and joins the data;
5. rebuilds engineered features;
6. trains the model; and
7. writes the model bundle to:

```text
models/xgboost_nem_price_model.joblib
```

Historical ingestion can require substantial download and processing time because each daily AEMO archive may contain many nested interval files.

### Current-data run

```bash
python run_pipeline.py
```

This processes recent files from the AEMO current DispatchIS directory before rerunning the downstream stages.

Database writes use upsert logic, so records with existing primary keys are updated rather than duplicated.

### Launch the dashboard

```bash
streamlit run app.py
```

The dashboard requires:

* a populated `engineered_features` table; and
* a compatible model bundle at `models/xgboost_nem_price_model.joblib`.

---

## Reproducibility

The repository provides:

* declared Python dependencies;
* centralised environment-driven configuration;
* a SQL schema;
* deterministic XGBoost configuration through `random_state=42`;
* a single pipeline entry point;
* fixed feature definitions; and
* a serialised model bundle containing feature names and metrics.

Current reproducibility gaps include:

* dependencies use minimum versions rather than pinned versions;
* the raw training data is not included;
* the database snapshot used for the supplied model is not included;
* the exact training and test dates are not stored in the model bundle;
* dataset counts are not stored alongside the model;
* predictions and actual test targets are not exported;
* the model is serialised with Joblib, which can be sensitive to XGBoost and Python version changes; and
* there are no automated tests or data-quality reports.

For stronger experiment traceability, future model bundles should include dataset dates, row counts, source configuration, package versions, training parameters, baseline metrics, and test predictions.

---

## Limitations

### Limited weather representation

Only one live weather station is configured, and Melbourne monthly averages are used as a fallback. The weather feature does not yet represent conditions across all NEM regions.

### Ambiguous interval-based feature names

Several lag and rolling columns do not match the row shifts used in the code. The project should establish one canonical data frequency and calculate features using timestamp-based windows.

### No benchmark model

The XGBoost result is not compared with simple alternatives such as:

* last observed price;
* same time on the previous day;
* rolling mean;
* linear regression; or
* a region-specific baseline.

Without a benchmark, it is not possible to determine how much value XGBoost adds over a simple forecasting rule.

### Limited evaluation detail

The current evaluation reports aggregate MAE, RMSE, and R² only. It does not include:

* performance by region;
* performance by forecast-price band;
* spike-event recall or error;
* seasonal performance;
* actual-versus-predicted plots;
* residual analysis;
* prediction intervals; or
* repeated backtesting windows.

### Pooled regional model without region identity

The model combines observations from multiple regions but does not receive `region_id` as an input. Regional price dynamics may therefore be underrepresented.

### No exogenous market variables

The feature set does not currently include:

* generation availability;
* unit outages;
* renewable output;
* interconnector flows;
* network constraints;
* bidding data;
* forecast demand;
* fuel conditions; or
* market notices.

### No deployment or monitoring layer

The application runs locally and does not implement model versioning, automated retraining, drift detection, forecast logging, alerting, authentication, or production service monitoring.

---

## Suggested next steps

1. Correct the feature names and calculate lag windows from timestamps.
2. Save training dates, test dates, row counts, source settings, and package versions with each model.
3. Add naive and statistical baseline forecasts.
4. Report metrics separately for each NEM region.
5. Evaluate performance on high-price and extreme-price intervals.
6. Add region-specific weather stations and historical weather data.
7. Include region identity or train separate regional models.
8. Add renewable generation, interconnector, outage, and constraint features.
9. Export actual and predicted test values for diagnostic analysis.
10. Replace or supplement Joblib persistence with XGBoost's native model format.
11. Add automated tests for parsing, feature timing, and leakage prevention.
12. Implement rolling-origin backtesting across multiple evaluation periods.

---

## Responsible use

This project is an analytical prototype. Electricity-market forecasts can be sensitive to data quality, changing market conditions, rare events, and modelling assumptions.

The outputs should not be treated as financial, trading, bidding, or operational advice.

---

## Licence

This project is available under the MIT License. See [`LICENSE`](LICENSE) for details.