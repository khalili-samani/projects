# Armani Outlet Sales Analysis, Power BI Dashboard

A Power BI dashboard analysing two years of transaction-level sales data from Armani Outlet, DFO South Wharf, built to answer real retail management questions around profitability, promotions, and customer capture.

**Tools:** Power BI · Power Query (M) · DAX

**A note on the data:** This dataset was deliberately generated with realistic messiness, duplicate transactions, inconsistent formatting, and missing values, specifically to demonstrate data-cleaning and Power Query skills, rather than sourced from an actual retailer. "Armani Outlet" and all staff names are illustrative and used for educational/portfolio purposes only.

**Status:** All five dashboard pages complete, Executive Overview, Product and Category, Promotions, Staff, and Customer.

**Headline numbers (Jul 2024 – Jul 2026):** $2.37M net sales · $917K gross profit · 38.71% margin · $382.06 average order value · 3.24% return rate.

---

## The business questions

Rather than build charts for the sake of it, this dashboard was built around specific questions a store or regional manager would actually ask:

1. Is the store growing, and is that growth profitable?
2. Which products and categories earn their shelf space, and which quietly cost the store money?
3. Are promotional campaigns driving genuine profit, or just pulling forward sales at a loss?
4. How does staff performance compare, and where does it matter?
5. Who's actually buying, and how well is the store capturing their details for future contact?

## What the data showed

- **Promotions are the clearest story in the dataset.** Deep-discount events (EOFY Clearance, Black Friday, Boxing Day) drive volume but collapse margin well below baseline, EOFY Clearance runs the deepest average discount and the lowest margin of any campaign. VIP Weekend is the exception: margin holds near the no-campaign baseline while still moving real volume, making it the template worth repeating over the heavier discount events.
- **Outerwear underperforms on margin** despite being the second-highest revenue category ($493,875), it returns the lowest margin (32.69%) of any category, well behind Accessories (42.88%), which is the smallest category by revenue but the most profitable per dollar sold.
- **No single product is unprofitable overall**, but trench coats, chinos, and boots are disproportionately likely to go margin-negative on individual heavily-discounted sales.
- **Customer contact capture is a genuine gap, not a data error.** Only 44.10% of transactions capture an email — meaning over half the store's customers can't be re-contacted. Walk-in and Tourist segments drive the bulk of revenue ($938,998 and $406,918 respectively) yet carry the lowest capture rates in the store, so the gap sits exactly where the revenue is.
- **Staff performance is tight, not spread out.** Net sales and average order value across the team sit in a narrow band with no clear outlier — more useful as a baseline to monitor after a process change than as a tool for singling anyone out.

## Data quality — the interesting part

The raw export (`data/raw/raw_sales.csv`, ~6,300 rows) needed significant repair before it was analysis-ready. Rather than hide this, I've documented it because the judgement calls involved are arguably more representative of real analyst work than the final charts:

| Issue | What I found | How it was handled |
|---|---|---|
| Duplicate transactions | 70 exact duplicate rows, plus 16 more hidden behind a `-MANUAL` suffix on the transaction ID | Suffix stripped before dedup so both types collapse correctly |
| Inconsistent currency formatting | Money columns mixed `$566.97`, `AUD 609.43`, and plain numbers | Stripped and converted to proper numeric types |
| Two date formats | `D/M/YYYY` and `DD-Mon-YY` mixed throughout | Parsed with a fallback chain, unparseable dates left null for review rather than guessed |
| Inconsistent categorical text | Store names, categories, and payment methods each had 5–30+ spelling/casing variants | Standardised via text cleaning and, where a single outlet, hardcoded to the correct value |
| Missing product codes / categories (35 / 28 rows) | Product names follow a predictable colour + style pattern, and category is embedded in the product code prefix | Recovered ~97% of both fields using pattern-matching logic built directly into the query, rather than dropping the rows |
| Sign errors on returns | 13 return transactions recorded quantity as positive when 187 others correctly used negative | Corrected using `is_return` as the source of truth, verified against `net_sales` first |
| Inconsistent staff naming | Same 10 staff members recorded under 2–3 name variants each (e.g. "O. Martin" / "Oliver Martin") | Standardised via a canonical lookup table keyed on staff ID |

Full reasoning for each decision, including the cases I chose *not* to "fix" (unexplained future-dated transactions, missing customer IDs), is in the Power Query step names themselves, written in plain English so this logic doesn't require reading M code to follow. A step-by-step summary is also in [`power_bi/power_query_steps.md`](power_bi/power_query_steps.md).

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

Note: the five dimension tables (`Date`, `Product`, `Staff`, `Promotion`, `Customer Type`) are generated directly inside Power Query from the cleaned `Fact_Sales` table, they don't exist as separate source files, since there's nothing to import that Fact_Sales doesn't already contain. `data/cleaned/clean_sales.csv` is an export of `Fact_Sales` only; the dimension tables are documented in the data dictionary but live purely inside the `.pbix`.

## Tech and approach

- **Power Query (M)** for all data cleaning, every transformation step is named as a plain-English action (e.g. *"Filled missing category from product code"*) so the cleaning logic is followable without reading M syntax
- **Star schema data model** — one fact table (`Fact_Sales`) with five dimension tables (`Date`, `Product`, `Staff`, `Promotion`, `Customer Type`), each generated directly from `Fact_Sales` in Power Query rather than imported separately, since the raw export only ever recorded these attributes at transaction level
- **DAX measures** organised into a dedicated measures table with display folders (Sales, Profitability, Returns, Data Quality, Time Intelligence)
- Every report page includes a plain-language purpose statement, KPI explanations, and a recommendation callout — written for a non-technical reader, not just for whoever built it

## Dashboard preview

**Executive Overview**, *Is the store growing, and is that growth profitable?*
![Executive Overview](screenshots/executive_overview.jpeg)

**Product & Category**, *Which products and categories earn their shelf space?*
![Product and Category](screenshots/product_and_category.jpeg)

**Promotions**, *Are sale events making money, or just pulling forward sales at a loss?*
![Promotions](screenshots/promotions.jpeg)

**Staff**, *How does the team compare, and where does coaching matter more than concern?*
![Staff](screenshots/staff.jpeg)

**Customer**, *Who's buying, and how well is the store capturing their details?*
![Customer](screenshots/customer.jpeg)

---