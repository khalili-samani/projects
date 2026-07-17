# Sector-Based Stock Market Analysis

A data analytics project that investigates the relationship between risk and return across major US companies in the Technology, Finance, Energy and Healthcare sectors.

The project combines data engineering, SQL analytics and visualisation to build a reusable market analysis pipeline. Historical stock price data is cleaned, loaded into a MySQL data warehouse, transformed into analytical views and visualised through automated reporting charts.

---

## Business Problem

Investors and analysts often compare sectors based on return potential and risk exposure.

However, raw market data alone does not provide meaningful insights. Analysts need a structured process for:

* Cleaning historical market data
* Calculating daily returns
* Measuring volatility
* Comparing companies across sectors
* Identifying risk-return trade-offs

This project demonstrates how a data warehouse and analytical workflow can transform raw stock market data into actionable insights.

---

## Why I Built This Project

I built this project to demonstrate practical data engineering and analytics skills using financial market data.

The objective was to create an end-to-end workflow that:

* Ingests historical stock price data
* Cleans and standardises datasets
* Stores data in a relational database
* Creates analytical SQL views
* Produces business-focused visualisations
* Supports sector-level investment analysis

---

## Key Results

| Area               | Result                     |
| ------------------ | -------------------------- |
| Companies Analysed | 12                         |
| Sectors Analysed   | 4                          |
| Database           | MySQL                      |
| Data Source        | Stooq                      |
| Analytical Views   | 3                          |
| Charts Produced    | 2                          |
| Time Period        | March 2023 – February 2026 |

---

## Example Outputs

### Risk vs Return Analysis

![Risk vs Return](outputs/charts/risk_vs_return.png)

The scatter plot compares average daily return against volatility for all companies.

Key observations:

* Nvidia delivered the highest average return but also exhibited the highest volatility.
* Healthcare companies clustered in the lower-risk, lower-return region.
* Financial institutions generated moderate returns with moderate volatility.
* Energy companies occupied the middle of the risk-return spectrum.

This visualisation highlights the classic relationship between risk and reward in equity markets.

---

### Average Daily Return by Company

![Average Daily Return](outputs/charts/avg_daily_return_by_company.png)

Nvidia achieved the strongest average daily performance during the analysis period.

Pfizer was the only company to generate a negative average daily return.

The chart illustrates substantial performance differences both across and within sectors.

---

## Skills Demonstrated

### Data Engineering

* ETL pipeline development
* Data cleaning and validation
* Relational database design
* Star schema modelling
* Automated data loading

### SQL Analytics

* Window functions (`LAG`)
* Aggregations
* Analytical views
* Dimensional modelling
* Data warehousing concepts

### Data Analysis

* Return calculations
* Volatility analysis
* Sector benchmarking
* Risk-return assessment
* Comparative company analysis

### Data Visualisation

* Business-focused chart design
* Financial reporting visuals
* Automated chart generation
* Insight communication

---

## Solution Architecture

```text
Raw Stock CSV Files
        │
        ▼
Data Cleaning Pipeline
        │
        ▼
MySQL Data Warehouse
        │
        ▼
Analytical SQL Views
        │
        ▼
Business Analysis Charts
```

The workflow automates the process of transforming raw market data into analytical outputs suitable for decision-making and reporting.

---

## Data Warehouse Design

The project uses a dimensional model consisting of two dimension tables and one fact table.

```text
dim_sector
      │
      ▼
dim_company
      │
      ▼
fact_stock_prices
```

This design separates descriptive business attributes from transactional stock price data and supports efficient analytical querying.

### Dimension Tables

#### dim_sector

Stores sector classifications:

* Technology
* Finance
* Energy
* Healthcare

#### dim_company

Stores:

* Company name
* Ticker symbol
* Sector assignment

#### fact_stock_prices

Stores daily market data including:

* Open price
* High price
* Low price
* Close price
* Trading volume
* Trading date

---

## Analytical Views

The database contains three analytical SQL views that support reporting and visualisation.

### vw_daily_returns

Calculates daily percentage returns for every company using the SQL `LAG()` window function.

```sql
(close_price - previous_close) / previous_close * 100
```

### vw_company_summary

Aggregates company-level metrics:

* Average daily return
* Volatility
* Total return
* Number of trading days

### vw_sector_summary

Aggregates performance metrics at sector level.

Used to compare:

* Average returns
* Volatility
* Relative sector performance

across Technology, Finance, Energy and Healthcare.

---

## Methodology

### Data Cleaning

The cleaning pipeline:

* Standardises column names
* Converts data types
* Validates records
* Removes invalid rows
* Assigns company identifiers

### Return Calculation

Daily return is calculated using the percentage change in closing price between consecutive trading days.

### Volatility Calculation

Volatility is measured as the standard deviation of daily returns.

### Risk vs Return Analysis

The final visualisation compares:

* Expected return (average daily return)
* Risk (volatility)

to identify companies with stronger risk-adjusted characteristics.

---

## Repository Structure

```text
market_analysis/
├── data_raw/
├── data_cleaned/
├── outputs/
│   └── charts/
│       ├── avg_daily_return_by_company.png
│       └── risk_vs_return.png
├── scripts/
│   ├── config.py
│   ├── clean_data.py
│   ├── load_data.py
│   ├── avg_daily_return.py
│   └── risk_vs_return.py
├── sql/
│   └── market.sql
├── requirements.txt
└── README.md
```

---

## Technologies Used

| Category        | Technology |
| --------------- | ---------- |
| Programming     | Python     |
| Database        | MySQL      |
| Data Processing | Pandas     |
| Database Access | SQLAlchemy |
| Visualisation   | Matplotlib |
| Data Source     | Stooq      |

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database Credentials

Windows:

```cmd
set MYSQL_USER=root
set MYSQL_PASSWORD=your_password
set MYSQL_HOST=localhost
set MYSQL_DATABASE=market_analysis
```

macOS / Linux:

```bash
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_HOST=localhost
export MYSQL_DATABASE=market_analysis
```

### 3. Prepare Raw Data

Download daily historical CSV files from:

https://stooq.com

Place the files inside:

```text
data_raw/
```

### 4. Clean Data

```bash
python scripts/clean_data.py
```

### 5. Load Data into MySQL

```bash
python scripts/load_data.py
```

### 6. Generate Visualisations

```bash
python scripts/avg_daily_return.py

python scripts/risk_vs_return.py
```

Generated charts are saved in:

```text
outputs/charts/
```

---

## Key Findings

| Finding                | Observation               |
| ---------------------- | ------------------------- |
| Highest Average Return | Nvidia                    |
| Lowest Average Return  | Pfizer                    |
| Highest Volatility     | Nvidia                    |
| Lowest Volatility      | Johnson & Johnson / Merck |
| Strongest Sector       | Technology                |
| Most Stable Sector     | Healthcare                |

---

## Business Value

This project demonstrates how raw financial data can be transformed into a structured analytical asset through data engineering, SQL modelling and visual analytics.

The workflow reflects common practices used in:

* Business Intelligence
* Financial Analytics
* Investment Research
* Data Warehousing
* Reporting and Dashboarding

---

## Limitations

Current limitations include:

* Dividend-adjusted returns are not included
* Transaction costs are ignored
* Tax considerations are excluded
* Analysis is limited to 12 companies
* Sector representation is intentionally simplified

---

## Future Improvements

Potential future enhancements include:

* Correlation heatmaps
* Rolling volatility analysis
* Sector performance dashboards
* Portfolio optimisation scenarios
* Power BI dashboards
* Streamlit applications
* Interactive Plotly visualisations
* Dividend-adjusted return calculations

---

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) for details.
