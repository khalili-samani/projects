# Data Preparation Log

## Source

* **Data source:** `data/raw/raw_sales.csv`
* **Source query:** `Raw Sales`
* **Working query:** `Sales Staging`

## Purpose

This document records the data preparation activities performed in Power Query prior to loading the dataset into the Power BI data model.

The objective is to produce a clean, consistent and reliable dataset suitable for reporting while maintaining a clear audit trail of all transformations.

---

## Initial Data Profiling

The raw sales dataset was imported into Power BI and profiled using:

* Column Quality
* Column Distribution
* Column Profile

### Initial Observations

The dataset loaded successfully and is structurally suitable for analysis. Initial profiling identified several data quality issues requiring further investigation, including:

* Duplicate transaction IDs
* Mixed date formats
* Currency values stored as text
* Inconsistent text casing and leading/trailing whitespace
* Inconsistent naming of stores, shopping centres and staff members
* Missing values in selected fields
* Return transactions requiring validation
* A small number of invalid and future-dated transactions

---

# Transformation Log

## Step 1 – Text Standardisation

**Status:** Completed

### Actions Performed

* Trimmed leading and trailing whitespace from all text columns.
* Removed non-printing characters using the **Clean** transformation.
* Preserved all records; no rows were removed during this step.

### Business Rationale

Operational data exported from retail point-of-sale systems commonly contains inconsistent spacing and hidden characters introduced through manual data entry or system integrations. Standardising text values improves data consistency and reduces duplicate values caused solely by formatting differences.

### Outcome

* Improved text consistency across descriptive fields.
* Reduced formatting-related inconsistencies.
* Dataset prepared for subsequent standardisation of names, dates and data types.

## Step 2 – Business Value Standardisation

**Status:** Completed

### Actions Performed

- Standardised store names.
- Standardised shopping centre names.
- Standardised suburb names.
- Standardised category values.
- Standardised colour values.
- Standardised payment method values.
- Standardised customer type values.
- Standardised promotion values.
- Replaced blank promotion values with `No Campaign`.

### Business Rationale

Operational retail data often contains inconsistent naming caused by manual entry, imports and POS system variations. Standardising business values ensures consistent reporting, accurate grouping and reliable filtering within the dashboard.

### Outcome

Business dimension fields now contain consistent descriptive values suitable for analysis.

## Step 3 – Date Preparation

**Status:** Completed

### Actions Performed

- Created `transaction_date_clean` from the original `transaction_date` field.
- Converted `transaction_date_clean` to Date using Australian locale.
- Replaced date parsing errors with null values.
- Created `date_quality_status` to classify records as valid, missing or future-dated.
- Filtered the staging table to retain valid transaction dates for the analytical model.

### Business Rationale

Reliable transaction dates are required for trend analysis, monthly reporting, time intelligence and date-table relationships. Invalid, missing or future-dated transactions can distort sales trends and should not be included in the main analytical model.

### Outcome

The staging table now contains only transactions with valid reporting dates. Invalid, missing and future-dated records will be retained separately in a rejected-records audit table later in the project.

## Step 4 – Numeric Standardisation

**Status:** In Progress

### Completed

* Converted `quantity` to Whole Number.
* Removed `AUD` prefixes from currency fields.
* Removed currency symbols (`$`).
* Removed thousands separators (`,`).
* Converted currency fields to Decimal Number.

### Pending

* Standardise `discount_rate` values.
* Validate negative values and outliers.
* Review conversion errors.

### Business Rationale

Financial metrics must be stored as numeric data types to support accurate calculations, aggregations and DAX measures. Cleaning currency values before conversion ensures consistent financial reporting across the dashboard.