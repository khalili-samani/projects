# Luxury Fashion Outlet Sales Analysis | Power BI

A five-page Power BI dashboard built from a synthetic retail transaction dataset to support store-management decisions across profitability, promotions, product mix, staff performance and customer contact capture.

The project demonstrates practical Power Query cleaning, dimensional modelling, DAX measure design and stakeholder-focused dashboard communication.

**Tools:** Power BI Desktop, Power Query (M), DAX

**Dataset:** Synthetic transaction-level data created for portfolio use. It is not sourced from, endorsed by or connected with Armani, DFO South Wharf or any real retailer. Staff names and commercial scenarios are fictional.

**Analysis period:** 1 July 2024 to 10 July 2026. This is a 25-month inclusive calendar span, with July 2026 representing a partial month.

**Headline results:** $2.369 million net sales, $917,029 gross profit, 38.71% gross margin, $382.06 average order value, 3.24% transaction return rate and 44.10% email capture rate.

---

## The business questions

The dashboard is designed for a store manager or regional retail manager who needs a concise view of revenue quality, margin performance and operational opportunities. It supports five management questions:

1. Are sales and gross profit moving in the same direction over time?
2. Which product categories generate revenue efficiently, and which require closer margin review?
3. How do promotional periods compare with normal trading on sales, discounts and gross margin?
4. Is staff performance materially different across sales volume and average order value?
5. Which customer segments generate revenue, and how consistently are contact details captured?

The dashboard is intended to support investigation and prioritisation. Because the data is synthetic and observational, it does not establish causal business impact.

## Data scope and model grain

- **Raw rows:** 6,287
- **Rows with missing transaction ID removed:** 1
- **Unique normalised transaction IDs after deduplication:** 6,200
- **Date range:** 1 July 2024 to 10 July 2026
- **Unparseable transaction dates after cleaning:** 19
- **Fact table grain:** one retained row per normalised transaction ID in the current model

## Data preparation and quality controls

The raw CSV was cleaned in Power Query before modelling. Key transformations included:

| Data-quality issue | Treatment | Validation or risk |
|---|---|---|
| Missing transaction IDs | Rows without an ID were removed because they could not be reliably linked to a sale | 1 raw row is affected |
| Duplicate and manually adjusted IDs | Transaction IDs were uppercased and the `-MANUAL` suffix was removed before deduplication | 6,200 unique normalised IDs remain from 6,287 raw rows |
| Mixed currency formats | `$`, `AUD` and thousands separators were removed before numeric conversion | `[AUTHOR TO ADD: conversion-error count after type assignment]` |
| Mixed date formats | Dates were parsed using two expected formats; unresolved values were left null | 19 unresolved dates in the full cleaned population |
| Missing net sales | Recalculated as gross sales less discount amount | `[AUTHOR TO ADD: number of values recovered and reconciliation tolerance]` |
| Missing product codes | Recovered using a style-name lookup built from matching product records | The lookup currently keeps the first matching code and does not test for collisions |
| Missing categories | Recovered from the product-code prefix | `[AUTHOR TO ADD: unrecovered product and category counts after transformation]` |
| Return quantity sign errors | Positive quantities on rows flagged as returns were reversed | `Is Return` is treated as the source of truth |
| Staff name variants | Resolved through a canonical staff-ID lookup | 10 staff IDs are represented |
| Missing customer IDs | Retained as a genuine operational data gap rather than imputed | Customer ID and email-capture status should not be treated as equivalent fields |

The cleaning process removed 86 records after transaction IDs were normalised, leaving 6,200 unique IDs. Of the source rows, 42 were fully identical across all columns.

## Data model

The Power BI model uses a star-schema pattern:

- `Fact_Sales`
- `Date`
- `Product`
- `Staff`
- `Promotion`
- `Customer Type`
- a dedicated measures table

Dimension tables are generated in Power Query from the cleaned fact data because no independent master-data sources were supplied.

## Measures and metric definitions

Core measures are implemented in DAX and organised into display folders.

| Metric | Definition |
|---|---|
| Transactions | Distinct count of normalised transaction IDs |
| Net Sales | Sum of net sales after discount |
| Gross Profit | Sum of net sales less estimated cost |
| Gross Margin | Gross profit divided by net sales |
| Average Order Value | Net sales divided by distinct transactions |
| Return Rate | Distinct returned transactions divided by all distinct transactions |
| Email Capture Rate | Distinct transactions flagged as email captured divided by all distinct transactions |
| Discount | Total discount amount divided by gross sales |

The supplied raw data reproduces the dashboard headline figures after the documented ID normalisation and deduplication:

| KPI | Recalculated result | Dashboard result |
|---|---:|---:|
| Net sales | $2,368,788.63 | $2.37M |
| Gross profit | $917,028.73 | $917.03K |
| Gross margin | 38.713% | 38.71% |
| Average order value | $382.06 | $382.06 |
| Return rate | 3.242% | 3.24% |
| Email capture rate | 44.097% | 44.10% |

### Measured results

- Walk-in customers generated the highest net sales at $938,998, followed by loyalty members at $804,961.
- Overall email capture was 44.10%. Capture rates by customer segment were relatively close, ranging from 43.47% for tourists to 47.81% for VIP customers.
- Womenswear generated the highest category sales at $693,468. Accessories generated the lowest sales at $224,939 but the highest gross margin at 42.88%.
- Outerwear generated $493,875 in net sales and had the lowest category margin at 32.69%.
- No-campaign trading had the highest observed gross margin among the displayed promotion groups. VIP Weekend and Fashion Week retained materially higher margins than the deepest-discount events.
- Staff sales and average-order-value results were visually concentrated, with no obvious extreme outlier in the displayed rankings.

### Management hypotheses

- Review outerwear pricing, cost assumptions and discount depth because the category combines meaningful revenue with a lower gross margin.
- Test whether accessories deserve greater visibility or attachment-selling support, while accounting for stock availability and customer demand.
- Compare VIP-style targeted promotions with broad discount events using incremental profit, not only observed margin.
- Improve point-of-sale email-capture processes, but first confirm consent, privacy notice and system usability requirements.
- Use staff results as a baseline for process monitoring rather than as a standalone performance-management tool.

## Repo structure

```
/data
  /raw/raw_sales.csv
  /cleaned/clean_sales.csv
  data_dictionary.md
/power_bi
  dfo_sales.pbix
  dax_measures.md
  power_query_steps.md
/screenshots
  executive_overview.jpeg
  product_and_category.jpeg
  promotions.jpeg
  staff.jpeg
  customer.jpeg
README.md
```

## Dashboard preview

### Executive overview

Provides headline sales, profit, margin, average-order-value and return-rate KPIs, plus monthly category and promotion context.

![Executive Overview](screenshots/executive_overview.jpeg)

### Product and category

Compares category revenue, gross margin and units sold.

![Product and Category](screenshots/product_and_category.jpeg)

### Promotions

Compares gross margin, net sales and aggregate discount by promotion.

![Promotions](screenshots/promotions.jpeg)

### Staff

Compares staff net sales, average order value and transaction counts.

![Staff](screenshots/staff.jpeg)

### Customer

Compares customer-segment sales, average order value and email capture.

![Customer](screenshots/customer.jpeg)

---

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) for details.
