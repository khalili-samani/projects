# Methodology

## Project approach

The project uses a synthetic encounter-level emergency department dataset to demonstrate an end-to-end Excel workflow without exposing personal or confidential information. The design mirrors common operational health analytics tasks: source validation, cleaning, KPI definition, temporal analysis, geographic comparison and executive reporting.

## Data acquisition

The dataset was generated specifically for this portfolio project. It contains 12,079 unique encounters from 1 January 2024 to 31 December 2025 across 12 fictional facilities. The raw extract contains 12,115 rows because 36 exact duplicate rows were intentionally added.

## Data preparation

The cleaning workflow covers:

- standardised snake_case field names;
- Australian date parsing and validation;
- duplicate identification using encounter ID;
- whitespace removal with TRIM logic;
- category mapping for sex values;
- facility validation against the lookup table;
- age-range validation;
- blank and null handling;
- outlier retention with quality flags rather than silent deletion;
- derived age group, month, year, weekday and shift;
- triage target lookup;
- within-target calculation; and
- before-and-after row-count reconciliation.

The workbook includes formula examples for XLOOKUP, INDEX/MATCH, IFS, LET, IFERROR, COUNTIFS, SUMIFS, AVERAGEIFS, TEXT, EOMONTH, FILTER, UNIQUE and SORT.

## Analytical methods

Analysis is descriptive. It includes counts, means, medians, proportions, year-on-year change and grouped comparisons. Monthly, state, region, triage and clinical-presentation summaries are included on `Pivot_Analysis`.

No causal modelling, forecasting, risk adjustment or inferential testing is performed.

## KPI logic

The workbook documents nine KPIs with business definitions, numerator, denominator, formula logic, display format, target/reference value, interpretation and limitations. Triage thresholds are portfolio assumptions of 0, 10, 30, 60 and 120 minutes for categories 1 to 5.

## Dashboard design

The one-page dashboard uses a restrained healthcare-oriented palette, high-contrast KPI cards, line charts for time trends and bar charts for comparisons. It avoids 3D charts, decorative gauges and pie charts. Insight annotations distinguish findings from interpretation.

## Validation process

Validation checks reconcile raw and cleaned row counts, duplicates, missing waits, invalid ages, unmatched facilities, inconsistent categories, invalid date sequences, out-of-range values, formula errors, KPI boundaries and dashboard currency.

## Reproducibility

All source, processed and reference files are included. Another analyst can inspect the data, validate the row counts and reproduce the summaries from `CleanEDTable`.

## Native PivotTable completion

The package generator cannot create native PivotTables, PivotCharts, slicers or timelines through its available Excel interface. The workbook contains complete pivot-style tables and charts. For native interactivity, follow the eight steps in the repository README. This is the only material manual workbook step.
