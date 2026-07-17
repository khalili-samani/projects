# Australian NEM Price Forecasting

An end-to-end analytics and machine learning platform that forecasts Australian National Electricity Market (NEM) dispatch prices 30 minutes ahead using electricity market and weather data.

The project combines data engineering, database design, feature engineering, machine learning and dashboard development to transform raw operational data into an interactive forecasting and decision-support tool.

![Dashboard Overview](outputs/screenshots/executive_summary.jpeg)

---

## Business Problem

Electricity prices in the Australian National Electricity Market are highly volatile and can change dramatically within short periods due to fluctuations in demand, weather conditions, generation availability and market dynamics.

Accurate short-term forecasts can help market participants monitor risk, understand emerging market conditions and support operational planning. However, forecasting electricity prices is challenging because of non-linear relationships, seasonality and rare but extreme price spike events.

This project explores how historical market and weather data can be used to forecast short-term electricity prices through a reproducible analytics pipeline.

---

## Why I Built This Project

I built this project to demonstrate how modern analytics workflows extend beyond machine learning models.

Rather than focusing solely on prediction accuracy, the project was designed as a complete analytical solution that:

* Collects and stores real-world operational data
* Cleans and prepares data for analysis
* Engineers predictive features
* Trains and evaluates forecasting models
* Delivers forecasts through an interactive dashboard

The objective was to simulate the type of end-to-end workflow commonly found in analytics, data engineering and market intelligence teams.

---

## Key Results

| Metric                     | Result                |
| -------------------------- | --------------------- |
| Forecast Horizon           | 30 Minutes Ahead      |
| Historical Period          | March 2026 – May 2026 |
| Raw AEMO Records Processed | 129,495               |
| Cleaned Records            | 129,495               |
| Engineered Feature Records | 129,225               |
| Training Rows              | 90,457                |
| Validation Rows            | 19,384                |
| Test Rows                  | 19,384                |
| Forecasting Model          | XGBoost Regressor     |
| Hold-Out MAE               | $19.02/MWh            |
| Hold-Out RMSE              | $33.43/MWh            |
| Hold-Out R²                | 0.5085                |

The model captures a meaningful proportion of short-term price variation while highlighting the inherent difficulty of forecasting extreme electricity price spikes.

---

## Skills Demonstrated

### Analytics & Forecasting

* Time-series forecasting
* Statistical analysis
* Model evaluation
* Performance measurement

### Data Engineering

* ETL pipelines
* Data cleaning
* Data transformation
* Database design

### Machine Learning

* XGBoost
* Feature engineering
* Leakage-safe forecasting
* Chronological model validation

### Data Visualisation & Reporting

* Streamlit dashboards
* Interactive scenario analysis
* Data storytelling
* Analytical reporting

### Technology Stack

Python • SQL • MySQL • Pandas • SQLAlchemy • XGBoost • Streamlit

---

## Dashboard Screenshots

The dashboard serves as the user-facing component of the forecasting pipeline and enables users to explore market conditions, evaluate forecasts and perform scenario analysis.

### Executive Summary

![Executive Summary](outputs/screenshots/executive_summary.jpeg)

Displays forecast outputs, model performance metrics and market summary information.

### Scenario-Based Forecasting

![Scenario Forecast](outputs/screenshots/scenario_based_forecasting.jpeg)

Allows users to adjust market inputs and observe how forecasted prices respond to changing conditions.

### Market History

![Market History](outputs/screenshots/market_history.jpeg)

Visualises historical dispatch prices and demand trends for selected NEM regions.

### Data Coverage

![Data Coverage](outputs/screenshots/data_coverage.jpeg)

Summarises regional coverage, record counts and key market statistics.

---

## Solution Overview

The project was designed as a complete forecasting workflow rather than a standalone machine learning model.

| Stage               | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
| Data Ingestion      | Collect market and weather data from external sources |
| Data Cleaning       | Validate, standardise and align datasets              |
| Feature Engineering | Transform raw observations into predictive features   |
| Model Training      | Train and evaluate forecasting models                 |
| Dashboard Reporting | Deliver forecasts through an interactive interface    |

Together, these components demonstrate the full lifecycle of an analytics solution.

---

## Architecture

The diagram below shows how data moves through the system from acquisition to forecasting and reporting.

```text
AEMO NEMWeb Archive / Current Files
                    │
                    ▼
          data_ingestion.py
                    │
                    ▼
             MySQL Database
                    │
                    ▼
           data_cleaning.py
                    │
                    ▼
       Cleaned Market Dataset
                    │
                    ▼
      feature_engineering.py
                    │
                    ▼
      Engineered Feature Table
                    │
                    ▼
        model_training.py
                    │
                    ▼
      XGBoost Model Bundle
                    │
                    ▼
              Streamlit App
```

---

## Repository Structure

The repository is organised to separate data engineering, modelling and reporting responsibilities.

```text
nem-forecasting/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
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

### Component Responsibilities

| Component       | Purpose                                              |
| --------------- | ---------------------------------------------------- |
| app.py          | Interactive forecasting dashboard                    |
| run_pipeline.py | Pipeline orchestration                               |
| src/            | Core analytics, data engineering and modelling logic |
| sql/            | Database schema definitions                          |
| models/         | Trained model artefacts                              |
| outputs/        | Screenshots and generated outputs                    |

---

## Data Sources

Forecast quality depends heavily on data quality.

This project combines electricity market data with weather observations because electricity demand and pricing are strongly influenced by environmental conditions.

| Source                 | Purpose                                     |
| ---------------------- | ------------------------------------------- |
| AEMO NEMWeb DispatchIS | Dispatch prices and electricity demand      |
| Bureau of Meteorology  | Weather observations                        |
| MySQL                  | Storage and retrieval of processed datasets |

---

## Feature Engineering

Raw market data rarely contains enough information for effective forecasting.

This stage transforms historical observations into predictive features that capture seasonality, temporal behaviour and recent market conditions while preventing information leakage.

| Feature Category  | Examples                                   |
| ----------------- | ------------------------------------------ |
| Weather Features  | Temperature                                |
| Calendar Features | Hour, weekday, month, weekend              |
| Cyclical Features | Hour sine/cosine, day sine/cosine          |
| Demand Lags       | Demand 30 minutes ago, demand 24 hours ago |
| Price Lags        | Price 30 minutes ago, price 24 hours ago   |
| Rolling Features  | Four-hour demand and price averages        |

**Target Variable**

```text
Dispatch price 30 minutes into the future
```

---

## Model Development

The forecasting model predicts dispatch prices 30 minutes ahead using historical market and weather information.

A chronological train-validation-test split was used to simulate real forecasting conditions and prevent look-ahead bias.

| Item              | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| Algorithm         | XGBoost Regressor                                          |
| Split Strategy    | 70% Train / 15% Validation / 15% Test                      |
| Evaluation Method | Hold-Out Test Set                                          |
| Output            | Trained model bundle with metadata and performance metrics |

---

## Pipeline Scale

A successful historical run processed:

```text
129,495 raw market records
129,495 cleaned records
129,225 engineered feature records

90,457 training rows
19,384 validation rows
19,384 testing rows
```

Model Performance:

```text
MAE:  $19.02/MWh
RMSE: $33.43/MWh
R²:   0.5085
```

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Database Environment Variables

```bash
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=nem_forecasting
```

### Run Historical Pipeline

```bash
python run_pipeline.py --historical
```

### Launch Dashboard

```bash
streamlit run app.py
```

---

## Dashboard Features

| Feature           | Purpose                                           |
| ----------------- | ------------------------------------------------- |
| Executive Summary | Forecast outputs and model metrics                |
| Scenario Analysis | Interactive forecasting under changing conditions |
| Market History    | Historical price and demand trends                |
| Data Coverage     | Regional coverage and market statistics           |
| Methodology       | Explanation of forecasting workflow               |
| Model Information | Forecast inputs and assumptions                   |

---

## Limitations

Understanding limitations is an important part of responsible analytics.

| Area                 | Limitation                                         |
| -------------------- | -------------------------------------------------- |
| Weather Coverage     | Limited live weather observations                  |
| Regional Granularity | Additional weather stations would improve coverage |
| Extreme Price Events | Rare spikes remain difficult to forecast           |
| Market Variables     | Renewable generation and outages are not included  |
| Training Window      | Longer historical periods would improve robustness |

---

## Future Improvements

Potential future enhancements include:

* Region-specific weather coverage across all NEM regions
* Renewable generation and interconnector flow features
* Feature importance analysis
* Actual-versus-predicted diagnostics
* Automated retraining schedules
* Extended historical training periods
* Continuous model monitoring

---

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) for details.