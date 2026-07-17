# Sector-Based Stock Market Analysis

An end-to-end data pipeline and analytics project comparing risk and return across 12 major US companies in Technology, Finance, Energy and Healthcare. Raw price data is cleaned, loaded into a MySQL warehouse, modelled with a star schema, and analysed through SQL views and automated charting.

---

## Overview

Comparing sectors on risk and return sounds simple, but raw market data doesn't get you there on its own. This project builds the structured workflow analysts actually need: cleaning historical price data, calculating daily returns and volatility, and surfacing risk-return trade-offs at both the company and sector level.

The end result is a repeatable pipeline: point it at new CSVs, and it cleans, loads, models and visualises the data with no manual intervention.

---

## Features

- **ETL pipeline**: cleans, validates and standardises raw CSV price data before loading
- **Dimensional data warehouse**: star schema in MySQL (2 dimension tables, 1 fact table)
- **SQL analytics layer**: 3 analytical views calculating returns, volatility and sector-level aggregates, including a window function (`LAG`) for day-over-day return calculations
- **Automated visual reporting**: risk-vs-return scatter plot and average daily return chart generated directly from the warehouse

---

## Tech Stack

| Category        | Technology |
| ---------------- | ---------- |
| Language          | Python |
| Database          | MySQL |
| Data processing   | Pandas |
| Database access   | SQLAlchemy |
| Visualisation     | Matplotlib |
| Data source       | Stooq |

---

## Architecture

```text
Raw Stock CSV Files
        │
        ▼
Data Cleaning Pipeline
        │
        ▼
MySQL Data Warehouse (star schema)
        │
        ▼
Analytical SQL Views
        │
        ▼
Business Analysis Charts
```

**Warehouse design:**

```text
dim_sector ──┐
             ├──> fact_stock_prices
dim_company ─┘
```

- `dim_sector`: Technology, Finance, Energy, Healthcare
- `dim_company`: company name, ticker, sector assignment
- `fact_stock_prices`: open, high, low, close, volume, trading date

**Analytical views:**

- `vw_daily_returns`: daily % return per company using `LAG()`
```sql
  (close_price - previous_close) / previous_close * 100
```
- `vw_company_summary`: average daily return, volatility, total return, trading days per company
- `vw_sector_summary`: same metrics aggregated to sector level

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure database credentials**

Windows:
```cmd
set MYSQL_USER=root
set MYSQL_PASSWORD=your_password
set MYSQL_HOST=localhost
set MYSQL_DATABASE=market_analysis
```

macOS/Linux:
```bash
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_HOST=localhost
export MYSQL_DATABASE=market_analysis
```

**3. Get the data**

Download daily historical CSVs from [Stooq](https://stooq.com) and place them in `data_raw/`.

**4. Run the pipeline**

```bash
python scripts/clean_data.py
python scripts/load_data.py
python scripts/avg_daily_return.py
python scripts/risk_vs_return.py
```

Charts are saved to `outputs/charts/`.

---

## Results

| Metric              | Value                       |
| -------------------- | ---------------------------- |
| Companies analysed   | 12 |
| Sectors analysed      | 4 (Technology, Finance, Energy, Healthcare) |
| Time period           | March 2023 – February 2026 |
| Analytical views       | 3 |
| Charts produced        | 2 |

**Risk vs return**

![Risk vs Return](outputs/charts/risk_vs_return.png)

- Nvidia had both the highest average daily return and the highest volatility of any company in the dataset
- Healthcare companies clustered in the low-risk, low-return region
- Financial and Energy companies sat in the middle of the risk-return spectrum, with moderate returns and moderate volatility

**Average daily return by company**

![Average Daily Return](outputs/charts/avg_daily_return_by_company.png)

- Nvidia posted the strongest average daily return of the 12 companies
- Pfizer was the only company with a negative average daily return
- Performance varied meaningfully both across and within sectors

**Summary**

| Finding                | Result |
| ------------------------ | -------- |
| Highest average return   | Nvidia |
| Lowest average return    | Pfizer |
| Highest volatility        | Nvidia |
| Lowest volatility          | Johnson & Johnson / Merck |
| Strongest sector           | Technology |
| Most stable sector         | Healthcare |

---

## Skills Demonstrated

**Data engineering**: ETL pipeline design, data cleaning and validation, relational schema design, star schema modelling, automated loading

**SQL**: window functions (`LAG`), aggregations, view design, dimensional modelling, data warehousing

**Analysis**: return and volatility calculations, sector benchmarking, risk-return assessment, comparative company analysis

**Visualisation**: chart design for business reporting, automated chart generation from a live data source

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

## Limitations

- Returns are not dividend-adjusted
- Transaction costs and tax are not factored in
- Limited to 12 companies across 4 sectors, so findings aren't representative of the broader market

---

## Future Improvements

- Correlation heatmaps across companies/sectors
- Rolling volatility analysis
- Interactive dashboard (Power BI or Streamlit)
- Portfolio optimisation scenarios
- Dividend-adjusted return calculations

---

## License

MIT License, see [LICENSE](LICENSE) for details.