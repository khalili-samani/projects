# Australian Housing Data Quality Simulator

A Python-based synthetic data generation tool that creates realistic, intentionally messy Australian residential property datasets for data cleaning, ETL, SQL, BI and machine learning projects.

The simulator reproduces common real-world data quality issues found in property market datasets while preserving realistic market behaviour, pricing patterns and housing market conditions across Australia from 2020–2025.

Unlike simple random data generators, this project models actual Australian housing market dynamics, including RBA cash rate changes, auction clearance rates, buyer sentiment, lockdown impacts and regional market trends.

---

# Business Problem

Real-world property datasets rarely arrive in a clean and analysis-ready format.

Analysts, data engineers and BI developers frequently encounter:

* Missing values
* Duplicate records
* Mixed date formats
* Inconsistent categorical values
* Incorrect postcodes
* Free-text price fields
* Data entry errors
* Outliers and anomalies

Obtaining realistic dirty datasets for learning and portfolio projects is often difficult due to privacy restrictions, licensing costs and limited public access.

This project addresses that challenge by generating realistic synthetic housing market data that can be used to practise data cleaning, ETL development, SQL analysis, dashboarding and machine learning workflows.

---

# Why I Built This Project

I built this project to simulate the types of data quality problems commonly encountered in real Australian property datasets sourced from:

* Domain
* realestate.com.au
* CoreLogic
* Government property transaction datasets
* Third-party data providers

The objective was to create a realistic testing environment where analysts and engineers can practise cleaning and transforming messy data before performing analysis.

---

# Key Features

| Feature                   | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| Geographic Coverage       | All Australian states and territories                         |
| Time Period               | January 2020 – December 2025                                  |
| Property Types            | Houses, Units, Apartments, Townhouses, Villas, Duplexes, Land |
| Housing Market Conditions | Realistic monthly market scenarios                            |
| Data Quality Issues       | 15 intentionally injected issues                              |
| Output Format             | CSV                                                           |
| Dependencies              | NumPy, Pandas                                                 |
| Internet Required         | No                                                            |
| API Keys Required         | No                                                            |

---

# Example Use Cases

The generated datasets can be used for:

* Data cleaning projects
* ETL pipeline development
* SQL practice
* Data quality assessment
* Exploratory data analysis
* Power BI dashboards
* Tableau dashboards
* Machine learning feature engineering
* Data warehouse testing

---

# Example Workflow

```text
Generate Synthetic Data
           │
           ▼
Data Cleaning Pipeline
           │
           ▼
Data Validation
           │
           ▼
Database / Data Warehouse
           │
           ▼
Analytics & Visualisation
           │
           ▼
Business Insights
```

---

# Market Realism

The simulator models actual Australian housing market conditions between 2020 and 2025.

Market variables include:

* RBA cash rates
* Auction clearance rates
* Buyer sentiment
* Listing volumes
* Property prices
* Days on market
* Rental estimates

Historical events incorporated into the simulation include:

### 2020

* COVID-19 pandemic
* National lockdowns
* Emergency RBA rate cuts
* Collapse in auction volumes
* Regional migration surge

### 2021

* Housing market boom
* Record-low interest rates
* Strong price growth
* FOMO-driven buyer behaviour
* Melbourne lockdown disruptions

### 2022

* Rapid interest rate increases
* Falling property prices
* Reduced borrowing capacity
* Declining auction clearance rates

### 2023

* Housing market recovery
* Population growth
* Rental crisis
* Supply shortages

### 2024

* Affordability pressures
* Elevated interest rates
* Longer selling times
* Diverging capital city performance

### 2025

* Interest rate cuts
* Improved confidence
* Renewed price growth
* Increased market activity

---

# Skills Demonstrated

## Data Engineering

* Synthetic data generation
* Data quality simulation
* Data modelling
* ETL testing
* Validation design

## Python Development

* Large-scale data generation
* Randomised simulation
* Configuration-driven architecture
* Input validation
* Modular code design

## Data Quality Engineering

* Missing value generation
* Duplicate generation
* Outlier simulation
* Inconsistent formatting
* Data standardisation challenges

## Analytics Preparation

* Feature generation
* Data profiling
* Data cleansing workflows
* Exploratory analysis preparation
* Machine learning dataset creation

## Domain Knowledge

* Australian housing markets
* Property transaction data
* RBA monetary policy
* Auction market dynamics
* Housing affordability trends

---

# Dataset Structure

The generated dataset contains information relating to:

## Property Information

* Property type
* Bedrooms
* Bathrooms
* Car spaces
* Land size
* Building area
* Year built

## Location Information

* Address
* Suburb
* Postcode
* Council area
* State
* Region
* Distance to CBD

## Transaction Information

* Sale price
* Sale date
* Sale method
* Days on market

## Market Information

* RBA cash rate
* Auction clearance rates
* Market sentiment
* Rental estimates
* Local market commentary

---

# Data Quality Issues Simulated

The simulator intentionally introduces realistic data quality problems.

## 1. Missing Values

Examples:

```text
N/A
NA
unknown
?
(blank)
```

## 2. Mixed Date Formats

Examples:

```text
15/03/2024
2024-03-15
15 Mar 2024
Mar-24
```

## 3. Inconsistent Property Types

Examples:

```text
House
house
HOUSE
Hse
Residential House
```

## 4. Sale Price Formatting

Examples:

```text
$1,250,000
$1.25M
POA
Contact Agent
Offers Over $900,000
```

## 5. State Variations

Examples:

```text
VIC
Vic
Victoria
vic.
V.I.C
```

## 6. Duplicate Records

* Near duplicates
* Exact duplicates

## 7. Outliers

Examples:

```text
Year Built = 1066
Year Built = 9999
```

## 8. Incorrect Postcodes

A subset of records intentionally contain postcode mismatches.

## 9. Boolean Inconsistencies

Examples:

```text
Yes
Y
True
1
```

and

```text
No
N
False
0
```

## 10. Field Swaps

A small percentage of records contain incorrectly assigned fields.

In total, the simulator introduces 15 different categories of data quality issues.

---

# Geographic Coverage

The simulator covers all Australian states and territories.

| State              | Example Suburbs                        |
| ------------------ | -------------------------------------- |
| Victoria           | Richmond, Toorak, Brighton, Geelong    |
| New South Wales    | Bondi, Manly, Parramatta               |
| Queensland         | Paddington, Gold Coast, Sunshine Coast |
| Western Australia  | Fremantle, Subiaco, Cottesloe          |
| South Australia    | Norwood, Glenelg                       |
| Tasmania           | Sandy Bay, Launceston                  |
| ACT                | Braddon, Gungahlin                     |
| Northern Territory | Darwin CBD, Palmerston                 |

---

# Repository Structure

```text
housing-data-quality-simulator/
│
├── messy_data_generator.py
├── README.md
```

---

# Installation

Install dependencies:

```bash
pip install numpy pandas
```

---

# Running the Simulator

Run:

```bash
python messy_data_generator.py
```

The simulator will prompt for:

```text
Which year(s)?
Which month(s)?
Which state(s)?
```

Example:

```text
Which year(s)? 2022
Which month(s)? mar-jun
Which state(s)? VIC, NSW
```

---

# Output

A CSV file is generated automatically.

Example:

```text
aus_housing_messy_2022_mar-jun_vic-nsw.csv
```

The number of records generated depends on:

* Selected time period
* Selected states
* Historical market activity

Lockdown periods generate fewer records, while boom periods generate larger datasets.

---

# Example Outputs

Include screenshots showing:

* Terminal execution
* Generated CSV opened in Excel
* Examples of messy data values
* Missing values and duplicates
* Multiple date formats
* Mixed property type categories

---

# Business Value

This project demonstrates how realistic synthetic data can be generated to support analytics and engineering workflows.

The simulator provides a reproducible environment for:

* Testing ETL pipelines
* Practising data cleaning
* Evaluating data quality frameworks
* Building analytics projects
* Developing machine learning workflows

without requiring access to proprietary property market datasets.

---

# Future Improvements

Potential future enhancements include:

* Apartment rental market simulation
* Commercial property simulation
* Synthetic mortgage datasets
* Population and demographic data generation
* Interactive Streamlit interface
* PostgreSQL export support
* Parquet output support
* Data quality scoring reports

---

# Technologies Used

| Category            | Technology                          |
| ------------------- | ----------------------------------- |
| Programming         | Python                              |
| Data Processing     | Pandas                              |
| Numerical Computing | NumPy                               |
| Output Format       | CSV                                 |
| Domain              | Australian Housing Market Analytics |

---

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) for details.