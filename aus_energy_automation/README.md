# Australian Energy Market Analysis Pipeline

## Overview

This project builds an **automated data pipeline** for collecting, processing, and analysing electricity market data from the Australian National Electricity Market (NEM).

The pipeline retrieves wholesale electricity price and demand data, stores it in a MySQL database, and generates analytical insights and visualisations across multiple regions.

---

## Objectives

* Automate data ingestion from AEMO
* Clean and standardise raw energy market data
* Store structured data in MySQL
* Analyse price behaviour, volatility, and demand patterns
* Generate reproducible charts for market insights

---

## Pipeline Architecture

```text
User Input (YYYYMM)
        ↓
Fetch Data (AEMO)
        ↓
Clean & Transform
        ↓
Load into MySQL
        ↓
Create Analytical Views
        ↓
Generate Charts
```

---

## Features

* Automated data pipeline (end-to-end)
* Multi-region analysis (NSW, QLD, VIC)
* SQL-based aggregation and metrics
* Time-series and distribution visualisations
* Modular and reusable Python scripts

---

## Project Structure

```text
aus_energy_automation/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── charts/
├── scripts/
│   ├── config.py
│   ├── fetch_energy.py
│   ├── clean_energy.py
│   ├── load_energy.py
│   ├── analyse_energy.py
│   └── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Data Source

Data is sourced from the **Australian Energy Market Operator (AEMO)**:

* Regional Reference Price (RRP)
* Electricity demand (MW)
* 5-minute settlement intervals

---

## Database Design

### Main Table

`fact_nem_price_demand`

| Column              | Description                      |
| ------------------- | -------------------------------- |
| region_code         | Market region (NSW1, QLD1, VIC1) |
| settlement_datetime | 5-minute interval timestamp      |
| trading_date        | Date                             |
| rrp_aud_mwh         | Electricity price (AUD/MWh)      |
| total_demand_mw     | Demand (MW)                      |
| period_type         | Settlement type                  |

---

### Analytical View

`vw_nem_daily_summary`

* Daily average price
* Maximum and minimum prices
* Price volatility (standard deviation)
* Average demand

---

## Visualisations

The pipeline generates the following charts:

* Daily average price by region
* Price vs demand relationship
* Price distribution (overall and by region)
* Volatility comparison by region

All outputs are saved in:

```text
outputs/charts/
```

---

## Key Insights

* Electricity prices are typically concentrated in lower ranges with occasional spikes
* Price distributions are right-skewed due to rare high-price events
* Regional differences exist in both price levels and volatility
* Demand influences price but is not the only driver (supply constraints also matter)

---

## Technologies Used

* Python
* pandas
* matplotlib
* MySQL
* SQLAlchemy
* PyMySQL

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_HOST=localhost
export MYSQL_DATABASE=aus_energy_automation
```

---

## Usage

Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

You will be prompted to enter:

```text
Year (e.g. 2026)
Month (e.g. 03)
```

---

## Future Improvements

* Integrate financial market data (ASX energy stocks)
* Add correlation analysis between electricity prices and stock returns
* Automate scheduling (cron jobs)
* Add logging and monitoring
* Extend to additional NEM regions (SA, TAS)

---

## Summary

This project demonstrates:

* Data pipeline automation
* Real-world data ingestion
* SQL-based analytics
* Visual storytelling of energy markets

It provides a strong foundation for combining **energy market analysis with financial market insights**.

---
