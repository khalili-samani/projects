# Luxury Fashion Outlet Sales Analysis in Power BI

![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat\&logo=powerbi\&logoColor=black)
![DAX](https://img.shields.io/badge/DAX-Data%20Modelling-blue)
![Power Query](https://img.shields.io/badge/Power%20Query-M-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

A five-page Power BI report analysing 6,200 synthetic retail transactions across sales, profitability, promotions, products, staff, customer segments, returns, and contact-data capture.

The project demonstrates how a transaction-level CSV can be cleaned, modelled, validated, and presented as a management reporting tool. It includes documented Power Query transformations, a star schema, reusable DAX measures, reconciliation checks, and dashboard pages designed for both summary review and detailed investigation.

> **Data disclaimer:** The dataset is synthetic and was created for portfolio use. It is not sourced from, endorsed by, or connected with any retailer. Staff names, transactions, customer records, and operating scenarios are fictional.

---

## Project Summary

| Area                 | Details                             |
| -------------------- | ----------------------------------- |
| Business domain      | Luxury fashion outlet retail        |
| Source data          | One synthetic transaction-level CSV |
| Raw records          | 6,287 rows                          |
| Cleaned fact records | 6,200 unique transactions           |
| Analysis period      | 1 July 2024 to 10 July 2026         |
| Reporting tool       | Power BI Desktop                    |
| Transformation layer | Power Query                         |
| Analytical language  | DAX                                 |
| Data model           | Star schema                         |
| Report pages         | 5                                   |

### Headline Results

| Metric              |        Result |
| ------------------- | ------------: |
| Net Sales           | $2,368,788.63 |
| Gross Profit        |   $917,028.73 |
| Gross Margin        |        38.71% |
| Average Order Value |       $382.06 |
| Return Rate         |         3.24% |
| Email Capture Rate  |        44.10% |

The final month is partial. Transactions with unresolved dates are retained in the fact table but do not contribute to date-based reporting.

---

## Business Problem

Top-line revenue alone does not tell a retail manager whether trading performance is sustainable. Managers also need to understand margin, discounting, returns, product mix, customer behaviour, and data-capture quality.

This report was designed around five management questions:

1. How are net sales and gross margin changing over time?
2. Which product categories combine revenue scale with healthy margins?
3. How does promotional trading compare with non-promotional trading?
4. How much variation exists between staff members in sales and average order value?
5. Which customer segments generate revenue, and how consistently is customer email information captured?

The dashboard is intended to support investigation and prioritisation. Because the data is synthetic and observational, it does not establish that a promotion, staff member, or customer segment caused a particular result.

---

## What This Project Demonstrates

### Data Preparation

* Cleaning mixed date, currency, percentage, and Boolean formats in Power Query
* Normalising transaction identifiers before deduplication
* Distinguishing missing data from values that can be deterministically recovered
* Correcting return quantity signs using an explicit return flag
* Standardising staff identities through an ID-based lookup
* Documenting transformations and data-quality assumptions

### Data Modelling

* Defining the fact-table grain as one row per normalised transaction ID
* Separating sales transactions from product, staff, promotion, customer-type, and date dimensions
* Generating a continuous calendar table for time-intelligence calculations
* Centralising measures in a dedicated measures table
* Organising DAX measures into display folders

### Analysis and Reporting

* Building reusable measures for sales, profitability, returns, customer-data capture, and year-on-year analysis
* Comparing categories and campaigns using both revenue and margin
* Reconciling Power BI outputs against the cleaned source data
* Presenting findings for a non-technical management audience without overstating what the data proves

---

## Dashboard Pages

### 1. Executive Overview

Summarises net sales, gross profit, gross margin, average order value, and return rate. It also provides monthly sales and margin trends, category contribution, and promotion-level margin context.

![Executive Overview](screenshots/executive_overview.jpeg)

### 2. Product and Category

Compares product categories by net sales, gross profit, margin, and units sold. It also supports investigation of products that record negative-margin transactions under discounting.

![Product and Category](screenshots/product_and_category.jpeg)

### 3. Promotions

Compares campaign performance using net sales, gross margin, and effective discount rate. “No Campaign” is retained as a reporting category so promotional and non-promotional trading can be compared.

![Promotions](screenshots/promotions.jpeg)

### 4. Staff

Compares staff members by net sales, transaction count, and average order value. These results provide a descriptive baseline, not a complete performance assessment, because hours worked, roster allocation, customer traffic, and shift mix are not available.

![Staff](screenshots/staff.jpeg)

### 5. Customer

Shows revenue and average order value by customer type, together with receipt-email capture rates.

![Customer](screenshots/customer.jpeg)

---

## Key Findings

### Category Performance

| Category    |   Net Sales | Gross Profit | Gross Margin |
| ----------- | ----------: | -----------: | -----------: |
| Womenswear  | $693,468.19 |  $277,839.49 |       40.07% |
| Menswear    | $561,540.41 |  $225,677.89 |       40.19% |
| Outerwear   | $493,875.03 |  $161,456.38 |       32.69% |
| Shoes       | $394,965.75 |  $155,598.10 |       39.40% |
| Accessories | $224,939.25 |   $96,456.87 |       42.88% |

Womenswear generated the highest net sales. Accessories generated the least revenue but recorded the highest category margin.

Outerwear generated $493,875.03 in net sales but had the lowest category margin at 32.69%. This makes it a reasonable candidate for closer review of product costs, pricing, and discount depth.

### Customer Mix and Contact-Data Capture

| Customer Type  |   Net Sales | Email Capture Rate |
| -------------- | ----------: | -----------------: |
| Walk-in        | $938,997.62 |             43.58% |
| Loyalty Member | $804,961.04 |             44.13% |
| Tourist        | $406,917.63 |             43.47% |
| VIP            | $142,166.79 |             47.81% |
| Corporate      |  $75,745.55 |             46.15% |

Walk-in customers generated the highest net sales, followed by loyalty members.

The overall email capture rate was 44.10%. Capture rates were relatively similar across segments, ranging from 43.47% for tourists to 47.81% for VIP customers. The data shows a contact-information gap, but it does not establish whether missing emails reflect customer preference, consent requirements, staff process, or system limitations.

### Promotion Performance

| Promotion          |     Net Sales | Gross Margin | Effective Discount |
| ------------------ | ------------: | -----------: | -----------------: |
| No Campaign        | $1,578,030.73 |       44.48% |              7.63% |
| VIP Weekend        |   $139,560.06 |       39.90% |             15.03% |
| Fashion Week       |    $62,179.58 |       38.84% |             16.65% |
| Mid-Season Sale    |   $166,121.22 |       31.89% |             25.21% |
| Easter Outlet Sale |    $56,750.93 |       29.15% |             27.50% |
| Winter Sale        |   $187,141.12 |       23.36% |             32.90% |
| Black Friday       |    $43,265.55 |       18.60% |             37.57% |
| Boxing Day         |    $35,756.63 |       16.34% |             38.79% |
| EOFY Clearance     |    $99,982.81 |        8.21% |             44.07% |

Non-promotional trading recorded the highest gross margin. Among the named campaigns, VIP Weekend and Fashion Week retained more margin than the deeper-discount events.

These figures describe observed transactions only. They do not measure incremental sales, customer acquisition, inventory-clearance value, or what sales would have occurred without each campaign.

### Staff Performance

Net sales by staff member ranged from $208,997.07 to $255,598.19. Average order value ranged from $362.84 to $408.97.

The relatively narrow ranges indicate that no staff member was an extreme outlier in this dataset. A fair performance assessment would require additional variables such as hours worked, roster coverage, store traffic, role seniority, and customer allocation.

---

## Recommendations

The analysis supports the following areas for further investigation:

1. **Review outerwear economics.** Examine unit costs, price points, markdown frequency, and promotion participation to understand why its margin trails the other categories.

2. **Test the role of accessories in the product mix.** Accessories have the highest category margin but the lowest revenue. Further analysis could determine whether placement, availability, or attachment selling affects sales.

3. **Evaluate promotions using incremental profit.** VIP Weekend and Fashion Week retained more margin than the deeper-discount campaigns, but campaign decisions should also consider incremental volume, stock clearance, customer acquisition, and repeat purchasing.

4. **Investigate email-capture workflow.** Determine whether the 44.10% capture rate reflects customer consent, POS design, staff behaviour, data-entry rules, or another operational constraint before setting improvement targets.

5. **Use staff metrics as diagnostic indicators.** Sales and average order value can highlight differences for investigation, but they should not be treated as standalone measures of individual performance.

---

## Data Preparation

### Record Reconciliation

The source contained 6,287 rows.

* 1 row had no transaction ID and was removed.
* 86 additional rows were removed when transaction IDs were normalised and deduplicated.
* The 86 duplicate IDs included 42 exact duplicate rows.
* The final fact table contains 6,200 unique transaction IDs.

The `Fact_Sales` table therefore has a grain of one row per normalised transaction ID.

### Main Cleaning Rules

| Data issue                  | Treatment                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| Missing transaction ID      | Removed because the row could not be reliably identified                                   |
| Manual transaction suffixes | Removed `-MANUAL`, converted IDs to uppercase, then deduplicated                           |
| Blank text fields           | Converted empty strings to null values                                                     |
| Store and location variants | Standardised to one outlet and location                                                    |
| Mixed currency formats      | Removed currency symbols, `AUD` labels, and thousands separators before numeric conversion |
| Mixed discount formats      | Converted percentages and decimal fractions to a consistent decimal representation         |
| Mixed date formats          | Parsed expected `D/M/YYYY` and `DD-Mon-YY` formats                                         |
| Invalid or missing dates    | Retained as null rather than inferred                                                      |
| Missing net sales           | Recalculated as gross sales minus discount amount                                          |
| Missing product codes       | Recovered through a product-style lookup where a matching style was available              |
| Missing categories          | Recovered from the category component of the product code                                  |
| Mixed Boolean formats       | Standardised values such as `Y`, `N`, `1`, `0`, `TRUE`, and `FALSE`                        |
| Return quantity errors      | Changed positive quantities to negative where `Is Return` was true                         |
| Staff name variants         | Replaced with canonical names using `Staff Id`                                             |
| Missing customer IDs        | Retained as missing rather than imputed                                                    |

After deduplication, 32 transactions have unresolved dates. These records remain in the fact table but are excluded from analyses that depend on the date relationship.

Full transformation details are documented in [`power_bi/power_query_steps.md`](power_bi/power_query_steps.md).

---

## Data Model

The report uses a star schema centred on `Fact_Sales`.

```text
                         ┌──────────────┐
                         │     Date     │
                         └──────┬───────┘
                                │
┌──────────────┐         ┌──────▼───────┐         ┌──────────────┐
│   Product    │────────▶│  Fact_Sales  │◀────────│    Staff     │
└──────────────┘         └──────┬───────┘         └──────────────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
          ┌──────▼───────┐ ┌────▼─────────┐ ┌──▼─────────────┐
          │  Promotion   │ │ Customer Type│ │ Measures Table │
          └──────────────┘ └──────────────┘ └────────────────┘
```

### Model Tables

| Table           | Purpose                                                    |
| --------------- | ---------------------------------------------------------- |
| `Fact_Sales`    | One row per cleaned transaction                            |
| `Date`          | Continuous calendar generated from valid transaction dates |
| `Product`       | One row per product code                                   |
| `Staff`         | One row per staff ID                                       |
| `Promotion`     | One row per promotion name                                 |
| `Customer Type` | One row per customer segment                               |
| Measures table  | Central location for DAX measures                          |

The dimension tables are generated from the cleaned fact data because no independent product, staff, promotion, or customer master-data files were supplied.

Column definitions and derivation rules are documented in [`data/data_dictionary.md`](data/data_dictionary.md).

---

## DAX Measures

The measures table is organised into Sales, Profitability, Returns, Data Quality, and Time Intelligence display folders.

### Core Measures

```dax
Transactions =
DISTINCTCOUNT ( Fact_Sales[Transaction Id] )
```

```dax
Net Sales =
SUM ( Fact_Sales[Net Sales] )
```

```dax
Gross Profit =
SUM ( Fact_Sales[Gross Profit] )
```

```dax
Margin =
DIVIDE ( [Gross Profit], [Net Sales] )
```

```dax
Avg Order Value =
DIVIDE ( [Net Sales], [Transactions] )
```

```dax
Returned Transactions =
CALCULATE (
    [Transactions],
    Fact_Sales[Is Return] = TRUE ()
)
```

```dax
Return Rate =
DIVIDE ( [Returned Transactions], [Transactions] )
```

```dax
Email Capture Rate =
DIVIDE (
    CALCULATE (
        [Transactions],
        Fact_Sales[Receipt Email Captured] = TRUE ()
    ),
    [Transactions]
)
```

The full measure set, including discount, units, manual adjustments, negative-margin transactions, and year-on-year calculations, is documented in [`power_bi/dax_measures.md`](power_bi/dax_measures.md).

---

## Validation

Headline measures were independently recalculated from `data/cleaned/clean_sales.csv` and compared with the Power BI report.

| Metric              | Recalculated value | Dashboard display |
| ------------------- | -----------------: | ----------------: |
| Transactions        |              6,200 |             6,200 |
| Net Sales           |      $2,368,788.63 |            $2.37M |
| Gross Profit        |        $917,028.73 |          $917.03K |
| Gross Margin        |          38.71298% |            38.71% |
| Average Order Value |         $382.06268 |           $382.06 |
| Return Rate         |           3.24194% |             3.24% |
| Email Capture Rate  |          44.09677% |            44.10% |

The differences shown above are display rounding only.

---

## Limitations

* The dataset is synthetic, so the results should not be interpreted as evidence about a real retailer.
* Promotion comparisons are descriptive and do not estimate incremental or causal impact.
* The source does not include store traffic, labour hours, inventory levels, stock availability, marketing costs, consent records, or customer lifetime value.
* Thirty-two transactions have no valid date and therefore cannot be included in time-based analysis.
* Product-code recovery uses the first available style-to-code match. The source was not tested against an independent product master.
* Product attributes in the generated dimension are taken from the first matching fact record for each product code.
* Staff comparisons are not adjusted for hours worked, shifts, traffic, role, or customer mix.
* The date table is generated from the minimum and maximum valid transaction dates, so an undetected outlier date could extend the reporting calendar.
* Email capture indicates only whether an email was recorded. It does not show whether the customer consented to marketing or whether the address was valid.

---

## Repository Structure

```text
dfo_sales/
├── data/
│   ├── raw/
│   │   └── raw_sales.csv
│   ├── cleaned/
│   │   └── clean_sales.csv
│   └── data_dictionary.md
├── power_bi/
│   ├── dfo_sales.pbix
│   ├── dax_measures.md
│   └── power_query_steps.md
├── screenshots/
│   ├── executive_overview.jpeg
│   ├── product_and_category.jpeg
│   ├── promotions.jpeg
│   ├── staff.jpeg
│   └── customer.jpeg
├── LICENSE
└── README.md
```

---

## How to Run the Project

### Requirements

* Power BI Desktop
* Git, or another method for downloading the repository

### Steps

1. Clone the repository.

   ```bash
   git clone [Add repository URL]
   cd dfo_sales
   ```

2. Open [`power_bi/dfo_sales.pbix`](power_bi/dfo_sales.pbix) in Power BI Desktop.

3. Review the supporting documentation:

   * [`power_bi/power_query_steps.md`](power_bi/power_query_steps.md) for transformation logic
   * [`power_bi/dax_measures.md`](power_bi/dax_measures.md) for measure definitions
   * [`data/data_dictionary.md`](data/data_dictionary.md) for table and column definitions

4. To inspect the source and transformed records directly, compare:

   * `data/raw/raw_sales.csv`
   * `data/cleaned/clean_sales.csv`

The screenshots in this README provide a static preview for reviewers who do not have Power BI Desktop installed.

---

## Suggested Extensions

The current dataset supports descriptive reporting. Useful extensions would require additional data rather than further interpretation of the existing fields.

Potential next steps include:

* introducing an independent product master to validate recovered product attributes;
* adding inventory and stock-on-hand data to assess sell-through and availability;
* incorporating labour hours and store traffic for normalised staff comparisons;
* adding customer-level purchase history for retention and cohort analysis;
* including campaign costs and control periods for incremental-profit evaluation;
* adding automated data-quality tests before report refresh.

---

## Licence

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.