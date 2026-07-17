# Australian Housing Data Quality Simulator

A Python program that generates synthetic Australian residential property-sale data with deliberately introduced data-quality problems.

The project is designed for practising data cleaning, exploratory analysis, ETL development, SQL transformations, dashboard preparation and feature-engineering workflows without relying on licensed property datasets.

> **Important:** The generated records are entirely synthetic. They do not represent actual properties, listings, transactions or agent details.

## Project overview

Real property data is often expensive, restricted or too clean to demonstrate realistic data-quality work. This generator creates a configurable CSV containing plausible housing attributes alongside inconsistent formats, missing values, invalid values, duplicates and other defects commonly encountered in operational datasets.

Users select:

* one or more years;
* one or more months;
* one or more Australian states or territories.

The script then:

1. selects synthetic suburbs from the requested jurisdictions;
2. generates property and transaction attributes;
3. adjusts row volumes and selected market fields using monthly configuration values;
4. introduces controlled random variation;
5. corrupts selected fields using multiple error patterns;
6. adds exact and near-duplicate records;
7. shuffles the final dataset;
8. exports the result as CSV.

The output is intentionally unsuitable for direct analysis. It is intended to be profiled, cleaned, validated and transformed first.

## What this project demonstrates

This project provides evidence of practical skills in:

* Python and Pandas development;
* synthetic data generation;
* rule-based simulation design;
* data-quality testing and remediation;
* mixed-type and malformed-field handling;
* configurable command-line input;
* domain-oriented data modelling;
* ETL test-data preparation;
* schema inspection and exploratory data analysis.

It is particularly relevant to data analyst, analytics engineer, data engineer, BI and junior data science portfolio work.

## Verified example output

A supplied run covering January 2021 through December 2023 for NSW, Queensland and Victoria produced:

| Measure                           | Observed value |
| --------------------------------- | -------------: |
| Rows                              |         11,190 |
| Columns                           |             37 |
| Fully duplicated rows             |            128 |
| Rows with a repeated `listing_id` |            360 |
| Output format                     |            CSV |

The repeated identifiers include both exact duplicates and modified re-listing-style records.

These figures describe the supplied example only. Row counts and observed issue rates vary between runs because generation is random and monthly listing-volume factors affect the number of records produced.

## Dataset contents

The generated dataset contains 37 fields across five broad areas.

### Property identity and location

| Field                | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| `listing_id`         | Synthetic listing identifier                                               |
| `source`             | Synthetic listing source                                                   |
| `address`            | Generated street address                                                   |
| `suburb`             | Suburb name, potentially corrupted                                         |
| `state`              | State or territory, potentially represented in several formats             |
| `postcode`           | Postcode, including deliberately mismatched or inconsistently typed values |
| `council_area`       | Local government area                                                      |
| `region`             | Broad geographic region                                                    |
| `distance_to_cbd_km` | Simulated distance from the relevant CBD                                   |
| `lat`                | Synthetic latitude                                                         |
| `lon`                | Synthetic longitude                                                        |

### Property characteristics

| Field           | Description                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------- |
| `property_type` | House, unit, apartment, townhouse, villa, duplex or land, with inconsistent labels introduced |
| `bedrooms`      | Bedroom count stored using mixed numeric and textual formats                                  |
| `bathrooms`     | Simulated bathroom count                                                                      |
| `car_spaces`    | Simulated parking capacity                                                                    |
| `toilets`       | Simulated toilet count                                                                        |
| `land_size`     | Land area represented using mixed units and formats                                           |
| `building_area` | Simulated building area                                                                       |
| `year_built`    | Construction year, including deliberately invalid outliers                                    |
| `has_pool`      | Boolean-like field represented using inconsistent values                                      |
| `has_garage`    | Boolean-like field represented using inconsistent values                                      |

### Transaction details

| Field             | Description                                                                            |
| ----------------- | -------------------------------------------------------------------------------------- |
| `sale_price`      | Deliberately inconsistent display version of the sale price                            |
| `price_raw_aud`   | Underlying generated price before display-format corruption, subject to missing values |
| `sale_date`       | Sale date represented using several formats                                            |
| `sale_method`     | Synthetic transaction method                                                           |
| `days_on_market`  | Simulated listing duration                                                             |
| `inspection_note` | Optional generated inspection or market note                                           |

### Agent details

| Field         | Description                                              |
| ------------- | -------------------------------------------------------- |
| `agent_name`  | Synthetic agent name                                     |
| `agency_name` | Synthetic agency name                                    |
| `agent_phone` | Synthetic phone number, potentially malformed or missing |

### Market context

| Field                        | Description                                                               |
| ---------------------------- | ------------------------------------------------------------------------- |
| `rba_cash_rate_pct`          | Monthly cash-rate value taken from the script's configuration table       |
| `market_sentiment`           | Manually assigned monthly sentiment category                              |
| `market_context`             | Manually written contextual note for the selected month                   |
| `suburb_median_price`        | Simulated suburb-level reference price                                    |
| `auction_clearance_rate_pct` | Configured Melbourne or Sydney clearance-rate value with random variation |
| `weekly_rent_aud`            | Synthetic weekly rental estimate                                          |
| `property_count_suburb`      | Synthetic suburb property count                                           |

## Data-quality problems introduced

The generator introduces defects across several dimensions rather than applying a single corruption rule.

### Missing and placeholder values

Fields may contain actual nulls or textual placeholders such as:

```text
N/A
n/a
NA
-
unknown
?
<blank>
```

This requires cleaning logic to distinguish true values from multiple representations of missing data.

### Inconsistent categorical values

Values such as states, suburbs and property types can vary in:

* capitalisation;
* abbreviations;
* punctuation;
* leading or trailing whitespace;
* spelling format;
* descriptive wording.

Examples include:

```text
VIC
Vic
vic.
V.I.C
Victoria
```

### Mixed date formats

The `sale_date` field may contain multiple representations, including numeric, ISO-style and abbreviated month formats.

A cleaning workflow must parse these values into a consistent date type and identify values that cannot be resolved safely.

### Malformed price values

The display version of `sale_price` may contain:

```text
$1,250,000
$1.25M
$780,000.00
POA
Contact Agent
Offers Over $900,000
```

This creates a realistic distinction between extracting a numeric amount and interpreting the meaning of a listing-price phrase.

### Mixed numeric types and units

Numeric fields may be stored as integers, decimal strings, padded strings or values containing unit labels.

For example, `land_size` may be expressed as:

```text
650
650 sqm
650m2
0.065 ha
```

Bedroom counts may similarly appear as numbers, words or strings such as `3 bed`.

### Inconsistent boolean values

Boolean-like fields may use values such as:

```text
Yes
yes
Y
True
true
1
No
no
N
False
false
0
```

### Invalid and implausible values

The generator deliberately inserts values that should fail reasonable validation rules, including construction years such as:

```text
1066
2099
9999
```

Coordinates, postcodes and selected numeric fields may also be missing, malformed or inconsistent with related fields.

### Geographic inconsistencies

A proportion of records contain suburb and postcode combinations that do not agree.

These records are useful for practising:

* cross-field validation;
* reference-data joins;
* exception reporting;
* survivorship rules;
* correction versus rejection decisions.

### Exact and near duplicates

The generator adds:

* complete copies of existing rows;
* modified copies that retain the same listing identity while changing selected attributes.

This supports exercises involving duplicate detection, record linkage and deduplication rules.

## Generation approach

### Configuration-driven geography

The script contains a predefined suburb catalogue with:

* suburb;
* council area;
* postcode;
* state or territory;
* base price;
* region;
* distance from the CBD.

Only suburbs belonging to the selected jurisdictions are included in a run.

### Property generation

Each row is assembled from random and rule-based components, including:

* property type;
* room counts;
* land and building size;
* construction year;
* sale method;
* sale price;
* listing duration;
* rent estimate;
* agent details.

Some variables depend on other generated fields. For example, the sale-price calculation incorporates the suburb reference price, property type, bedroom count and the configured monthly price index.

### Monthly simulation settings

The source code includes a manually configured lookup table for January 2020 through December 2025.

Each month contains values for:

* cash rate;
* price-index multiplier;
* Melbourne auction-clearance rate;
* Sydney auction-clearance rate;
* days-on-market adjustment;
* listing-volume multiplier;
* market sentiment;
* contextual commentary.

These settings shape the synthetic data but should not be treated as an authoritative historical dataset.

### Variable row volume

The script starts from a base number of records per month and multiplies it by the configured monthly volume factor.

Quiet periods therefore produce fewer records, while high-activity periods produce more.

A minimum monthly row count prevents a selected period from producing an empty or impractically small dataset.

### Corruption stage

Data-quality issues are introduced while individual records are generated and again after the initial DataFrame is assembled.

The post-generation stage adds duplicate records and randomises row order before export.

## Repository structure

```text
.
├── LICENSE
├── README.md
└── messy_data_generator.py
```

## Requirements

* Python 3
* NumPy
* Pandas

Install the required packages with:

```bash
python -m pip install numpy pandas
```

For a more reproducible development environment, pin tested package versions in a `requirements.txt` or `pyproject.toml` file.

`[Add tested Python, Pandas and NumPy versions]`

## Running the generator

From the repository directory:

```bash
python messy_data_generator.py
```

The program prompts for a date range and geographic coverage.

Example:

```text
Which year(s)?   2021-2023
Which month(s)?  all
Which state(s)?  NSW, QLD, VIC
```

Supported year input formats include:

```text
2022
2021, 2023
2021-2023
all
```

Supported month input formats include:

```text
mar
3
jan, mar, oct
jan-jun
3-9
all
```

Supported state input includes abbreviations, full names and comma-separated selections:

```text
VIC
Victoria
VIC, NSW, QLD
all
```

## Output

The program writes the CSV to the current working directory.

The filename is derived from the selected period and jurisdictions. For example:

```text
aus_housing_messy_2021-2023_jan-dec_nsw-qld-vic.csv
```

After generation, the script prints:

* the output path;
* row and column counts;
* approximate records generated for each selected month;
* null rates based on actual `NaN` values;
* configured price-index and cash-rate values;
* a catalogue of selected data problems.

Textual missing-value markers are not included in the printed Pandas null-rate calculation. A complete profiling workflow should normalise those markers before measuring missingness.

## Example analysis workflow

The generated data can be used to demonstrate a structured data-quality process:

1. load the CSV without prematurely coercing mixed-type columns;
2. profile distinct values and inferred data types;
3. standardise textual missing-value markers;
4. parse and validate sale dates;
5. normalise state, suburb and property-type labels;
6. extract numeric prices while preserving non-price listing phrases;
7. convert land measurements into a common unit;
8. standardise boolean values;
9. validate construction years, coordinates and postcodes;
10. identify exact and near duplicates;
11. document rejected, corrected and unresolved records;
12. produce a cleaned analytical table and data-quality report.

Possible deliverables include:

* a Pandas or SQL cleaning pipeline;
* automated validation tests;
* a before-and-after data-quality scorecard;
* a dimensional model;
* a Power BI or Tableau dashboard;
* a feature table for a machine-learning experiment.

## Reproducibility

The current implementation is stochastic.

At runtime, it derives a random seed from the current system time and uses that seed for both NumPy and Python's `random` module. As a result:

* identical user selections do not produce identical rows;
* the seed is not currently shown to the user;
* the seed is not written to the output;
* a previous run cannot be recreated exactly from its filename or configuration alone.

A useful future enhancement would be an optional command-line seed:

```bash
python messy_data_generator.py --seed 42
```

The program could then record the seed and generation settings in a metadata file or in its console output.

## Known limitations

### Historical calibration is not independently documented

The monthly market table contains manually assigned values and commentary. The repository does not currently provide source references, calibration calculations or validation against an external dataset.

The market columns should therefore be interpreted as simulation inputs inspired by Australian housing conditions, not as a verified historical time series.

### Date availability and configured periods differ

The input logic considers the current date when checking whether a requested month is in the future. However, the monthly market lookup currently ends in December 2025.

Periods after December 2025 cannot be generated until corresponding configuration entries are added.

### Clearance rates are simplified

The model stores separate Melbourne and Sydney auction-clearance values.

Victoria uses the Melbourne series. Other jurisdictions use the Sydney series rather than state-specific or city-specific values. This is a simplifying assumption and should not be interpreted as a local auction measure for every jurisdiction.

### Geographic coordinates are approximate

Coordinates are synthetic and are not consistently calibrated to each generated suburb. They should not be used for real geospatial analysis or mapping without replacement or validation.

### Phone-number geography is simplified

Generated phone-number prefixes are not consistently matched to every Australian jurisdiction.

### Data corruption is probabilistic

Issue rates are applied through random rules. The final prevalence of each defect therefore varies by run, and the script does not currently produce a complete issue-level ground-truth file.

Without such a file, a cleaning pipeline can measure detected problems but cannot calculate exact detection precision and recall.

### Duplicate reporting requires interpretation

Repeated `listing_id` values include exact duplicates and modified duplicate records. The relationship between the configured duplicate rate and the final percentage of added rows is not presented as a formal evaluation metric.

### No automated tests are included

The repository does not currently contain tests for:

* input parsing;
* supported date ranges;
* schema stability;
* generated-value constraints;
* expected corruption rates;
* filename creation;
* deterministic output under a supplied seed.

## Potential improvements

The most useful next steps would be:

1. add an optional fixed random seed and record it with each output;
2. document and cite the monthly market configuration;
3. make coordinates, phone numbers and clearance rates jurisdiction-specific;
4. export a clean reference dataset alongside the corrupted dataset;
5. produce an issue manifest identifying every injected defect;
6. add automated tests for parsing, generation and corruption rules;
7. add dependency pinning;
8. add non-interactive command-line arguments;
9. support Parquet and database output;
10. generate a machine-readable run manifest containing configuration and schema metadata.

A clean reference dataset and issue manifest would make it possible to evaluate a downstream cleaning pipeline objectively rather than relying only on manual inspection.

## Appropriate use

This project is suitable for:

* portfolio demonstrations;
* classroom exercises;
* ETL prototyping;
* SQL transformation practice;
* data-quality rule development;
* BI pipeline testing;
* interview take-home preparation;
* synthetic test-data generation.

It is not suitable for:

* property valuation;
* investment decisions;
* market forecasting;
* suburb comparisons;
* economic research;
* identifying real properties or agents.

## Licence

This project is available under the MIT Licence. See [`LICENSE`](LICENSE) for details.