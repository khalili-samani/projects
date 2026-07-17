# Australian Energy Market Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![MySQL](https://img.shields.io/badge/Database-MySQL-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end ETL and analytics pipeline that ingests, cleans, stores and reports on wholesale electricity market data from the Australian National Electricity Market (NEM). Built to demonstrate the full data analyst/analytics engineer workflow: from raw AEMO files through to SQL reporting views and automated visual reports.

![Daily Average Wholesale Price](outputs/charts/daily_avg_rrp_by_region_202603.png)

---

## Business Problem

Electricity markets generate large volumes of operational data every day, but raw data on its own tells market analysts and energy professionals very little. It has to be collected, validated, structured and modelled before it can support pricing or demand decisions.

This project builds a repeatable pipeline that turns raw AEMO price and demand data into business-ready reporting, without manual intervention at any stage.

---

## Key Results

| Metric | Result |
|---|---|
| Market | Australian National Electricity Market (NEM) |
| Regions analysed | NSW1, QLD1, VIC1 |
| Data frequency | 5-minute settlement intervals |
| Reporting period | March 2026 |
| Database | MySQL |
| Main fact table | `fact_nem_price_demand` |
| SQL reporting views | 2 |
| Automated charts | 4 |

---

## Tech Stack

**Python** · **Pandas** · **SQL** · **MySQL** · **SQLAlchemy** · **PyMySQL** · **Matplotlib**

---

## Skills Demonstrated

- **Data engineering:** ETL pipeline design, ingestion automation, data cleaning and validation, sequential pipeline orchestration via `run_pipeline.py`
- **SQL & database design:** relational schema design, fact table modelling, reporting view creation, aggregation queries
- **Analytics:** price and demand analysis, volatility analysis, regional comparison, insight generation
- **Visualisation:** automated chart generation covering trend, distribution and relationship analysis

---

## Pipeline Architecture

```text
AEMO Price & Demand Files
        │
        ▼
 fetch_energy.py  →  Raw CSV files
        │
        ▼
 clean_energy.py  →  Cleaned dataset
        │
        ▼
 load_energy.py   →  MySQL database
        │
        ▼
 SQL reporting views
        │
        ▼
 analyse_energy.py → Visual reports
```

---

## Repository Structure

```text
aus_energy_automation/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   └── charts/
│
├── scripts/
│   ├── config.py
│   ├── fetch_energy.py
│   ├── clean_energy.py
│   ├── load_energy.py
│   ├── analyse_energy.py
│   └── run_pipeline.py
│
└── sql/
    └── aus_energy.sql
```

| Component | Purpose |
|---|---|
| `fetch_energy.py` | Downloads AEMO market data |
| `clean_energy.py` | Cleans and validates raw data |
| `load_energy.py` | Loads processed data into MySQL |
| `analyse_energy.py` | Generates visual reports |
| `run_pipeline.py` | Runs the full pipeline end to end |
| `aus_energy.sql` | Database schema and reporting SQL |

---

## Database Design

Central fact table built for analytical queries:

**`fact_nem_price_demand`**

| Column | Description |
|---|---|
| region_code | NEM region identifier |
| settlement_datetime | 5-minute settlement timestamp |
| trading_date | Trading date |
| rrp_aud_mwh | Regional Reference Price |
| total_demand_mw | Electricity demand |
| period_type | Settlement period type |

Two reporting views sit on top of the fact table:

- **`vw_nem_daily_summary`**: daily average, max and min price, price volatility, average demand and interval counts by region
- **`vw_nem_regional_summary`**: regional averages, volatility statistics, demand statistics, negative pricing frequency and reporting period coverage

Negative prices are retained intentionally, as they reflect genuine NEM market behaviour rather than data errors.

---

## Setup and Installation

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables** (PowerShell example)

```powershell
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_HOST="localhost"
$env:MYSQL_DATABASE="aus_energy_automation"
```

**3. Run the pipeline**

```bash
python scripts/run_pipeline.py
```

This downloads market data, cleans and validates it, loads it into MySQL, builds the reporting views and generates the charts, in that order.

---

## Sample Outputs and Insights

### Daily average wholesale price by region
![Daily Average Wholesale Price](outputs/charts/daily_avg_rrp_by_region_202603.png)

### Average daily price volatility by region
![Price Volatility](outputs/charts/avg_daily_price_volatility_by_region_202603.png)

### Price vs demand relationship
![Price vs Demand](outputs/charts/price_vs_demand_202603.png)

### Electricity price distribution (VIC1)
![Price Distribution](outputs/charts/price_distribution_vic1_202603.png)

**Key findings from the March 2026 dataset:**

- **NSW1** ran the highest average prices with relatively stable demand and few negative pricing events
- **QLD1** showed moderate volatility with several high-price periods but consistent demand
- **VIC1** had the highest price volatility, the largest share of negative pricing intervals, and the widest price distribution of the three regions

These differences show how much regional dynamics can vary within a single national market.

---

## Limitations

- Covers NSW1, QLD1 and VIC1 only, not the full NEM
- Single-month reporting period (no multi-month trend analysis yet)
- No FCAS market data
- No renewable generation or interconnector flow data

---

## Future Improvements

- Extend coverage to all NEM regions
- Add renewable generation and FCAS market data
- Build an interactive Streamlit or Power BI dashboard
- Support multi-month trend analysis
- Add automated scheduling and monitoring

---

## License

Licensed under the MIT License, see [LICENSE](LICENSE) for details.