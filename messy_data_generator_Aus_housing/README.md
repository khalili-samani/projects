# Australian Housing Data Quality Simulator

A Python tool that generates synthetic, intentionally messy Australian residential property data (2020–2025) for practising data cleaning, ETL, SQL and BI workflows, without needing access to costly or restricted real-world property datasets.

---

## Overview

Real property datasets from sources like Domain, realestate.com.au and CoreLogic are usually locked behind licensing costs or privacy restrictions, making it hard to find realistic "dirty" data for portfolio or training projects. This simulator solves that by generating synthetic housing transaction data that mirrors real Australian market conditions (RBA cash rate movements, auction clearance rates, COVID-era disruptions, the 2021 boom, 2022 rate hikes, etc.) while deliberately injecting 15 categories of data quality issues: missing values, duplicate records, mixed date formats, inconsistent categorical values, malformed prices, incorrect postcodes and more.

The result is a reproducible, on-demand dataset for practising data cleaning, ETL pipeline development, SQL analysis, dashboarding (Power BI/Tableau) and ML feature engineering.

---

## Key Features

| Feature | Description |
|---|---|
| Geographic coverage | All Australian states and territories |
| Time period | January 2020 – December 2025 |
| Property types | Houses, units, apartments, townhouses, villas, duplexes, land |
| Market realism | Monthly market conditions modelled on RBA cash rate, auction clearance rates, buyer sentiment |
| Data quality issues | 15 intentionally injected issue categories |
| Output format | CSV |
| Dependencies | NumPy, Pandas (no internet or API keys required) |

---

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy
- **Output:** CSV

---

## Skills Demonstrated

- **Synthetic data generation & simulation design** — modelling realistic market behaviour rather than pure randomisation
- **Data quality engineering** — deliberately reproducing missing values, duplicates, outliers, inconsistent formatting and field swaps
- **Domain modelling** — translating real economic and market events (RBA rate changes, COVID lockdowns, auction cycles) into a configurable simulation
- **Python development** — configuration-driven, modular script design with input validation

---

## Data Quality Issues Simulated

The simulator injects 15 categories of realistic data problems, including:

| Issue | Example |
|---|---|
| Missing values | `N/A`, `NA`, `unknown`, `?`, blank |
| Mixed date formats | `15/03/2024`, `2024-03-15`, `15 Mar 2024`, `Mar-24` |
| Inconsistent property types | `House`, `house`, `HOUSE`, `Hse`, `Residential House` |
| Price formatting | `$1,250,000`, `$1.25M`, `POA`, `Offers Over $900,000` |
| State variations | `VIC`, `Vic`, `vic.`, `V.I.C` |
| Duplicate records | Near duplicates and exact duplicates |
| Outliers | `Year Built = 1066`, `Year Built = 9999` |
| Incorrect postcodes | Mismatched suburb/postcode pairs |
| Boolean inconsistencies | `Yes`/`Y`/`True`/`1`, `No`/`N`/`False`/`0` |
| Field swaps | A small percentage of records have incorrectly assigned fields |

---

## Market Realism

Market variables (RBA cash rate, auction clearance rates, buyer sentiment, listing volumes, days on market, rental estimates) are modelled year-by-year against real conditions, e.g.:

- **2020:** COVID-19 lockdowns, emergency RBA rate cuts, collapse in auction volumes
- **2021:** Record-low rates, housing boom, FOMO-driven buyer behaviour
- **2022:** Rapid rate hikes, falling prices, reduced borrowing capacity
- **2023:** Market recovery, rental crisis, supply shortages
- **2024:** Affordability pressure, elevated rates, longer selling times
- **2025:** Rate cuts, renewed price growth, increased market activity

---

## Dataset Structure

| Category | Fields |
|---|---|
| Property | Type, bedrooms, bathrooms, car spaces, land size, building area, year built |
| Location | Address, suburb, postcode, council area, state, region, distance to CBD |
| Transaction | Sale price, sale date, sale method, days on market |
| Market | RBA cash rate, auction clearance rate, market sentiment, rental estimate |

---

## Repository Structure

```text
housing-data-quality-simulator/
├── messy_data_generator.py
└── README.md
```

---

## Installation

```bash
pip install numpy pandas
```

---

## Usage

Run the script:

```bash
python messy_data_generator.py
```

You'll be prompted for:

```text
Which year(s)? 2022
Which month(s)? mar-jun
Which state(s)? VIC, NSW
```

This generates a CSV named according to your selection, e.g. `aus_housing_messy_2022_mar-jun_vic-nsw.csv`. Record volume scales with the selected time period and historical market activity (lockdown periods produce fewer records, boom periods produce more).

---

## Business Value

Recruiters sourcing real Australian property data face licensing costs and privacy restrictions. This simulator provides a free, reproducible alternative for:

- Testing ETL pipelines
- Practising data cleaning workflows
- Building BI dashboards and analytics projects
- Developing ML feature engineering pipelines

---

## Future Improvements

- Rental and commercial property simulation
- Synthetic mortgage datasets
- Interactive Streamlit interface
- PostgreSQL and Parquet export support
- Automated data quality scoring reports

---

## License

MIT License, see [LICENSE](LICENSE) for details.