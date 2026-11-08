
# Data Analytics Portfolio

I'm a data professional with an MPhil in Mathematics from the University of Melbourne. My interests span data analytics, analytics engineering, statistical modelling and the practical use of mathematics to investigate complex problems.

This repository brings together nine independent projects using Python, SQL, Power BI, Excel and other analytical tools. They cover Australian electricity markets, healthcare operations, retail, property data and financial markets. Across the projects, I focus on data quality, reproducible workflows and communicating results with appropriate context and limitations.

## Featured projects

### Australian Electricity Market: Price and Demand Analytics

**Python · Pandas · MySQL · SQL · Matplotlib**  
[Explore the project](./australian-nem-price-demand-analytics)

A pipeline that downloads public data from the Australian Energy Market Operator (AEMO), cleans and combines five-minute wholesale electricity price and demand observations, stores them in MySQL and produces SQL summaries and charts.

For March 2026, the documented analysis covers **26,784 observations** across Victoria, New South Wales and Queensland. Victoria had the lowest average wholesale price of those regions (**$44.74/MWh**) but the highest standard deviation of interval prices (**$49.45/MWh**); **25.44%** of Victorian intervals had negative prices. These are descriptive results for the period, not an explanation of what caused negative prices.

### Retail Sales and Profitability Analysis with Power BI

**Power BI · Power Query · DAX · Dimensional modelling**  
[Explore the project](./retail-sales-powerbi-analytics)

A five-page Power BI report built from **6,200 synthetic retail transactions**, covering sales, profitability, discounts, returns, products, staff and customer segments. The work includes data preparation, a star schema, reusable DAX measures and reconciliation checks.

The project focuses on management reporting and interpreting KPIs beyond sales revenue. All trading results are simulated rather than actual retailer performance.

### Australian Housing Data Quality and SQL Warehouse

**MySQL · SQL · Python · Data quality · Data warehousing**  
[Explore the project](./australian-housing-sql-warehouse)

A MySQL analytics warehouse designed to consume deliberately imperfect Australian housing data from the companion [housing data simulator](./australian-housing-data-simulator). The design covers raw ingestion, standardisation, duplicate classification and survivorship, dimensional modelling, reconciliation and analytical reporting marts.

The README documents the intended workflow and SQL techniques. Published end-to-end results should be checked against actual executions before interpreting the synthetic market analysis.

### Queensland Elective Surgery Capacity and Waitlist Recovery Optimiser

**Python · DuckDB · Parquet · Google OR-Tools · Data engineering**  
[Explore the project](./qld-elective-surgery-optimiser)

An engineering and optimisation workflow exploring how hypothetical additional elective surgery capacity could be allocated across Queensland facilities and services. It brings together public-source ingestion, validation, lineage, analytical modelling and constrained allocation scenarios.

The project distinguishes observed public data from derived measures and synthetic planning assumptions. Its README reports **143 passing local tests** for the implemented phases. Processing current Queensland Government resources end to end remains an outstanding live verification milestone; model allocations should not be interpreted as real operational recommendations.

## Other projects

| Project | Overview | Main tools |
| --- | --- | --- |
| [Australian NEM Price Forecasting](./australian-nem-price-forecasting) | A 30-minute-ahead electricity price forecasting prototype using engineered time-series features; the README also documents gaps in baseline evaluation. | Python, XGBoost, MySQL, Streamlit |
| [ASX Equity Intelligence Agent](./asx-equity-intelligence-agent) | A structured workflow for market data, next-session return forecasts, risk measures, rule-based signals and LLM-assisted briefings. Not a live trading system. | Python, LangGraph, DuckDB, scikit-learn, Gemini |
| [Emergency Department Demand and Waiting Time Analysis](./australian-ed-operations-excel) | Excel analysis and dashboard using synthetic Australian emergency department encounters and operational KPIs. | Excel, formulas, data visualisation |
| [Australian Housing Data Quality Simulator](./australian-housing-data-simulator) | Generates configurable synthetic property datasets containing missing values, inconsistent formats and duplicates for testing analytical workflows. | Python, Pandas |
| [Sector-Based Stock Returns Analysis](./sector-stock-returns-analysis) | Descriptive return and volatility comparison for 12 selected US-listed companies, with data processing and SQL reporting. | Python, MySQL, SQL, Matplotlib |

## Skills demonstrated

| Area | Methods and technologies |
| --- | --- |
| Data analysis and programming | Python, Pandas, NumPy, SQL, exploratory analysis, statistical analysis |
| Data engineering and modelling | Data ingestion, ETL, validation, reconciliation, MySQL, DuckDB, dimensional models, analytical views |
| Business intelligence | Power BI, Power Query, DAX, Excel, dashboards, KPI design, Matplotlib |
| Predictive analytics and optimisation | Time-series feature engineering, scikit-learn, XGBoost, Google OR-Tools |
| Workflow development | Reusable scripts, command-line tools, documentation, tests and structured orchestration |

## Data sources, scope and limitations

The portfolio combines **publicly available observed data** and **synthetic datasets**. The NEM analytics and forecasting projects use AEMO data, while the Queensland elective surgery project is designed around public Queensland Government resources. The retail, housing and emergency department projects use simulated records.

Results from synthetic data illustrate methods and workflows, not actual business, housing or health-system performance. These are independent portfolio projects, not systems claimed to have been deployed for clients or used in operational decision-making. Each project README provides its methodology, evidence, assumptions and limitations.

## Contact

I'm based in Melbourne and interested in opportunities across data analytics, data science and analytics engineering.

- [LinkedIn](https://www.linkedin.com/in/ali-khalili-91093b297/)
- [GitHub profile](https://github.com/khalili-samani)