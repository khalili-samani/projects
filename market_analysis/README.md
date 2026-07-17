# Sector-Based Stock Market Analysis

A Python and MySQL analytics project comparing daily returns and price volatility across 12 selected US-listed companies in four sectors.

The project takes daily OHLCV market data from CSV files, standardises it with Pandas, loads it into a dimensional MySQL schema, calculates return and volatility metrics with SQL, and generates two comparison charts with Matplotlib.

## Project Summary

| Item | Details |
| --- | --- |
| Companies | 12 |
| Sectors | Technology, Finance, Energy, Healthcare |
| Companies per sector | 3 |
| Observation period | 1 March 2023 to 27 February 2026 |
| Price records | 9,024 |
| Trading dates per company | 752 |
| Data source | Stooq CSV files |
| Database | MySQL |
| Main outputs | Company return chart and risk-versus-return chart |

### What the project demonstrates

- Cleaning and standardising multiple market-data files with Python and Pandas
- Loading relational data into a MySQL dimensional model
- Making repeated loads idempotent by replacing each company’s existing fact rows
- Calculating daily percentage returns with the SQL `LAG()` window function
- Aggregating return and volatility metrics at company and sector level
- Querying MySQL through SQLAlchemy
- Generating reproducible analytical charts with Matplotlib
- Separating configuration, ingestion, database loading, analysis, and output code

## Business and Analytical Question

Risk and return cannot be compared directly from raw closing prices because companies trade at different price levels.

This project converts closing-price movements into daily percentage returns and compares:

1. Which selected companies produced the highest average daily returns?
2. Which companies experienced the greatest daily price variability?
3. How did the selected companies group by sector?
4. Which observations sit in relatively high-return, high-volatility or low-return, low-volatility positions?

The analysis is descriptive. It is intended to demonstrate a data pipeline and comparative analytics workflow, not to provide investment recommendations.

## Dataset

The repository contains daily market data for the following companies:

| Sector | Companies |
| --- | --- |
| Technology | Apple, Microsoft, Nvidia |
| Finance | JPMorgan Chase, Goldman Sachs, Bank of America |
| Energy | ExxonMobil, Chevron, ConocoPhillips |
| Healthcare | Johnson & Johnson, Pfizer, Merck |

Each source file contains:

- Trading date
- Opening price
- Daily high
- Daily low
- Closing price
- Trading volume

The supplied dataset contains 752 observations for each company, covering 1 March 2023 through 27 February 2026.

> The companies are a deliberately limited sample. Results describe these 12 companies during this period and should not be interpreted as representative of the full US market or of each sector as a whole.

## Architecture

```text
Stooq CSV files
      |
      v
Python cleaning and standardisation
scripts/clean_data.py
      |
      v
Cleaned company CSV files
data_cleaned/
      |
      v
MySQL dimensional model
dim_sector
dim_company
fact_stock_prices
      |
      v
SQL analytical views
vw_daily_returns
vw_company_summary
vw_sector_summary
      |
      v
Python visualisation scripts
      |
      v
PNG charts
outputs/charts/
````

## Data Model

The MySQL database uses two dimensions and one fact table:

```text
dim_sector
    |
    | 1-to-many
    v
dim_company
    |
    | 1-to-many
    v
fact_stock_prices
```

### `dim_sector`

Stores the four sector classifications used in the analysis.

| Field         | Description       |
| ------------- | ----------------- |
| `sector_id`   | Sector identifier |
| `sector_name` | Sector label      |

### `dim_company`

Stores company reference information and sector assignments.

| Field          | Description                 |
| -------------- | --------------------------- |
| `company_id`   | Company identifier          |
| `company_name` | Company name                |
| `ticker`       | Market ticker               |
| `sector_id`    | Foreign key to `dim_sector` |

### `fact_stock_prices`

Stores one daily OHLCV record per company and trading date.

| Field         | Description                  |
| ------------- | ---------------------------- |
| `price_id`    | Surrogate primary key        |
| `company_id`  | Foreign key to `dim_company` |
| `date`        | Trading date                 |
| `open_price`  | Opening price                |
| `high_price`  | Daily high                   |
| `low_price`   | Daily low                    |
| `close_price` | Closing price                |
| `volume`      | Trading volume               |

A unique constraint on `(company_id, date)` prevents duplicate daily observations for the same company.

## Pipeline

### 1. Clean and standardise the source files

`scripts/clean_data.py`:

* Reads one raw CSV for each configured company
* Renames Stooq columns to a consistent snake_case schema
* Adds the corresponding `company_id`
* Parses dates
* Coerces price and volume fields to numeric values
* Removes rows containing unparseable values
* Sorts observations chronologically
* Writes one cleaned CSV per company to `data_cleaned/`

Company metadata and filesystem paths are maintained in `scripts/config.py`.

### 2. Create and populate MySQL

`scripts/load_data.py`:

* Creates the database when it does not already exist
* Creates the sector, company, and price tables
* Inserts sector and company reference records
* Deletes existing fact rows for each company before reloading them
* Loads the cleaned CSV files into `fact_stock_prices`
* Creates the analytical SQL views

Deleting and replacing each company’s rows allows the loading step to be rerun without accumulating duplicate records.

### 3. Calculate analytical metrics

The database exposes three views.

#### `vw_daily_returns`

Calculates close-to-close daily percentage return using `LAG()`:

```sql
(close_price - previous_close) / previous_close * 100
```

The first observation for each company is excluded because it has no previous closing price.

#### `vw_company_summary`

Aggregates the daily-return observations by company:

* Average daily return
* Standard deviation of daily returns
* Number of return observations

#### `vw_sector_summary`

Aggregates daily returns across the three selected companies assigned to each sector:

* Company count
* Average daily return
* Standard deviation of daily returns

The sector statistics are pooled descriptive statistics for the selected observations. They are not market-cap weighted sector-index returns.

### 4. Generate charts

Two scripts query `vw_company_summary` through SQLAlchemy:

```text
scripts/avg_daily_return.py
scripts/risk_vs_return.py
```

The generated PNG files are saved under:

```text
outputs/charts/
```

## Technology

| Area                  | Technology                                   |
| --------------------- | -------------------------------------------- |
| Programming           | Python                                       |
| Data manipulation     | Pandas                                       |
| Database              | MySQL                                        |
| Database connectivity | SQLAlchemy and PyMySQL                       |
| SQL analysis          | Window functions, views, joins, aggregations |
| Visualisation         | Matplotlib                                   |
| Data source           | Stooq                                        |

## Setup

### Prerequisites

* Python
* A running MySQL server
* Access to create a MySQL database
* Daily market CSVs in the expected Stooq format

The SQL uses window functions, so the MySQL installation must support `LAG()`.

### 1. Clone the repository

```bash
git clone [Add repository URL]
cd market_analysis
```

### 2. Create a virtual environment

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure MySQL credentials

macOS or Linux:

```bash
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_HOST=localhost
export MYSQL_DATABASE=market_analysis
```

Windows Command Prompt:

```cmd
set MYSQL_USER=root
set MYSQL_PASSWORD=your_password
set MYSQL_HOST=localhost
set MYSQL_DATABASE=market_analysis
```

Windows PowerShell:

```powershell
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_HOST="localhost"
$env:MYSQL_DATABASE="market_analysis"
```

`MYSQL_USER`, `MYSQL_HOST`, and `MYSQL_DATABASE` have defaults in `scripts/config.py`. `MYSQL_PASSWORD` must be supplied through the environment.

### 5. Add the source data

Place the Stooq CSV files in `data_raw/` using the filenames configured in `scripts/config.py`.

Examples:

```text
data_raw/aapl_us.csv
data_raw/msft_us.csv
data_raw/nvda_us.csv
```

Each CSV must contain these columns:

```text
Date, Open, High, Low, Close, Volume
```

The repository currently expects all 12 configured files to be present. The cleaning command exits with an error when one or more required files cannot be processed.

### 6. Run the pipeline

Run each command from the repository root:

```bash
python scripts/clean_data.py
python scripts/load_data.py
python scripts/avg_daily_return.py
python scripts/risk_vs_return.py
```

The loading script creates the database objects automatically. The SQL file under `sql/market.sql` is retained as a readable schema reference and does not need to be executed separately.

### 7. Review the outputs

```text
outputs/charts/avg_daily_return_by_company.png
outputs/charts/risk_vs_return.png
```

## Verified Results

The following values were recalculated from the supplied cleaned CSV files using close-to-close daily percentage returns.

### Company-level results

| Company           | Sector     | Average daily return | Daily volatility |
| ----------------- | ---------- | -------------------: | ---------------: |
| Nvidia            | Technology |              0.3206% |          3.0720% |
| Goldman Sachs     | Finance    |              0.1365% |          1.7554% |
| JPMorgan Chase    | Finance    |              0.1104% |          1.4927% |
| Apple             | Technology |              0.0946% |          1.6236% |
| Microsoft         | Technology |              0.0751% |          1.4903% |
| Johnson & Johnson | Healthcare |              0.0708% |          1.0852% |
| Bank of America   | Finance    |              0.0638% |          1.6383% |
| ExxonMobil        | Energy     |              0.0529% |          1.4482% |
| Merck             | Healthcare |              0.0387% |          1.5192% |
| Chevron           | Energy     |              0.0289% |          1.4062% |
| ConocoPhillips    | Energy     |              0.0249% |          1.7956% |
| Pfizer            | Healthcare |             -0.0379% |          1.5403% |

Daily volatility is the population standard deviation of daily percentage returns, matching MySQL’s `STDDEV()` behaviour.

### Main observations

* Nvidia had the highest average daily return and highest daily volatility in the selected sample.
* Pfizer was the only company with a negative average daily return over the observation period.
* Johnson & Johnson had the lowest daily volatility, not Johnson & Johnson and Merck jointly.
* The three selected Technology companies had the highest pooled average daily return.
* The selected Healthcare observations had the lowest pooled daily volatility.
* Results varied materially within sectors. For example, Healthcare included both the least volatile company and the only company with a negative average daily return.

### Risk versus return

![Risk versus Return](outputs/charts/risk_vs_return.png)

The scatter plot places average daily return on the horizontal axis and daily-return standard deviation on the vertical axis. It is useful for comparing relative positions within the selected sample, but it does not measure portfolio risk, correlation, drawdown, or risk-adjusted performance.

### Average daily return

![Average Daily Return by Company](outputs/charts/avg_daily_return_by_company.png)

The bar chart ranks the 12 companies by arithmetic mean daily return. Average daily return should not be interpreted as annualised or compounded performance.

## Repository Structure

```text
market_analysis/
├── data_raw/
│   ├── aapl_us.csv
│   ├── ...
│   └── xom_us.csv
├── data_cleaned/
│   ├── aapl_us_clean.csv
│   ├── ...
│   └── xom_us_clean.csv
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
├── LICENSE
├── requirements.txt
└── README.md
```

## Design Decisions

### Configuration-driven company registry

Company identifiers and names are held in a central Python dictionary. This avoids repeating file-to-company mappings across ingestion scripts.

A trade-off is that company metadata is duplicated between `scripts/config.py` and the database seed data in `scripts/load_data.py`. A larger implementation should use one authoritative metadata source.

### Dimensional schema

Sector and company attributes are separated from daily price observations. This makes the schema easier to query and avoids repeating company and sector names on every fact row.

For a dataset of this size, a single flat table would also be workable. The dimensional model is used here to demonstrate relational modelling and to support aggregation by business dimensions.

### SQL-based metric calculation

Returns and summary statistics are calculated in MySQL rather than entirely in Pandas. This keeps the analytical definitions close to the stored data and allows other reporting tools to query the same views.

### Idempotent company reloads

Before inserting a company’s cleaned records, the loading script removes its existing fact rows. This is simpler than implementing row-level upserts for a small batch dataset and prevents duplicate observations across reruns.

## Known Issues

### Duplicate date field in cleaned CSVs

The current cleaning script includes `date` twice when ordering output columns. Pandas writes the second instance as `date.1`.

The database loader ignores the extra field because it explicitly selects the required load columns, so the duplicated field does not affect the loaded fact table. It should nevertheless be corrected by changing the final column selection in `scripts/clean_data.py`.

### Incorrectly labelled total-return calculation

The current SQL definition of `total_return_pct` calculates the difference between the maximum and minimum closing prices:

```sql
(MAX(close_price) - MIN(close_price)) / MIN(close_price) * 100
```

This is a price-range statistic, not chronological total return. It should not be used or reported as total return.

A chronological price return would need to compare the first and last closing prices by date. A true investor total return would additionally require verified dividend and corporate-action adjustments.

### Unnecessary join in the company summary view

`vw_company_summary` joins every daily-return row to every fact-price row for the same company. Although the repetition does not alter the average or population standard deviation, it increases the intermediate row count and makes the view harder to reason about.

The view should calculate return statistics separately and join only to explicitly derived first and last price observations when a chronological price-return metric is required.

## Limitations

* The sample contains only 12 hand-selected companies.
* Each sector is represented by three companies.
* Sector results are not weighted by market capitalisation.
* The analysis does not include a market benchmark.
* Arithmetic mean daily return can be sensitive to outliers.
* Volatility is measured only as the standard deviation of daily returns.
* The analysis does not calculate Sharpe ratio, beta, maximum drawdown, value at risk, or downside deviation.
* Cross-company correlations are not analysed.
* Transaction costs, taxes, spreads, and slippage are excluded.
* The repository does not document the exact data retrieval process or retrieval timestamp.
* The adjustment treatment of the source prices has not been independently verified in the repository.
* The project contains no automated test suite or continuous-integration workflow.
* Database credentials must be configured manually.
* The pipeline expects locally supplied CSV files and does not retrieve current market data automatically.

## Reproducibility Notes

The supplied files allow the included analysis to be recreated, provided that MySQL is available and the environment variables are configured.

For stronger reproducibility, the project should additionally record:

* The exact Stooq download URL or retrieval procedure
* The date and time at which each source file was retrieved
* The supported Python version
* The tested MySQL version
* Row-count and schema validation outputs
* Unit tests for return calculations and data cleaning
* A deterministic environment lock file
* A single orchestration command for the complete workflow

## Potential Extensions

* Correct the chronological return calculation
* Remove the duplicate date field from cleaned outputs
* Add automated data-quality tests
* Add command-line arguments for date ranges and company selection
* Add benchmark-relative measures such as beta and excess return
* Calculate rolling volatility and maximum drawdown
* Analyse company and sector correlations
* Add risk-adjusted metrics
* Replace manually downloaded files with a documented ingestion process
* Add a Streamlit or business-intelligence reporting layer
* Add Docker configuration for reproducible MySQL and Python environments
* Add continuous integration for linting and tests

## Disclaimer

This project is an educational data and analytics exercise. Its outputs are descriptive and should not be treated as financial advice.

## Licence

This project is distributed under the MIT Licence. See [`LICENSE`](LICENSE).