# Armani Outlet Sales Analysis — Power BI Dashboard

A Power BI dashboard analysing two years of transaction-level sales data from Armani Outlet, DFO South Wharf, built to answer real retail management questions around profitability, promotions, and customer capture.

**Status:** Executive Overview page complete. Product & Category, Promotions, Staff, and Customer pages in progress.

---

## The business questions

Rather than build charts for the sake of it, this dashboard was built around specific questions a store or regional manager would actually ask:

1. Is the store growing, and is that growth profitable?
2. Which products and categories earn their shelf space, and which quietly cost the store money?
3. Are promotional campaigns driving genuine profit, or just pulling forward sales at a loss?
4. How does staff performance compare, and where does it matter?
5. Who's actually buying, and how well is the store capturing their details for future contact?

## What the data showed

A few findings worth calling out, since they shaped both the cleaning approach and the dashboard's recommendations:

- **Promotions are the clearest story in the dataset.** Deep-discount events (EOFY Clearance, Black Friday, Boxing Day) drive volume but collapse margin well below baseline. One campaign — VIP Weekend — holds margin near baseline while still driving real volume, making it the template worth repeating over the heavier discount events.
- **Outerwear underperforms on margin** despite being a strong revenue category — it sells well but returns noticeably less profit per dollar than every other category.
- **No single product is unprofitable overall**, but certain items (trench coats, chinos, boots) are disproportionately likely to go margin-negative on individual heavily-discounted sales.
- **41% of transactions have no customer ID captured**, concentrated in Walk-in and Tourist segments — a real gap in the store's ability to re-contact a large share of its customer base, not a data error.

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

Full reasoning for each decision — including the cases I chose *not* to "fix" (unexplained future-dated transactions, missing customer IDs) — is in the Power Query step names themselves, which are written in plain English specifically so this logic doesn't require reading M code to follow.

## Repo structure

```
/data
  /raw/raw_sales.csv         — original, untouched export
/power_bi
  dfo_sales.pbix
/screenshots
  executive_overview.jpeg
README.md
```

*(Note: this project currently uses the standard `.pbix` format rather than the text-based `.pbip` project format, so Power Query and DAX logic live inside the file itself rather than as separate readable text files in this repo.)*

## Tech and approach

- **Power Query (M)** for all data cleaning — every transformation step is named as a plain-English action (e.g. *"Filled missing category from product code"*) so the cleaning logic is followable without reading M syntax
- **Star schema data model** — one fact table (`Fact_Sales`) with five dimension tables (`Date`, `Product`, `Staff`, `Promotion`, `Customer Type`)
- **DAX measures** organised into a dedicated measures table with display folders (Sales, Profitability, Returns, Data Quality, Time Intelligence)
- Every report page includes a plain-language purpose statement, KPI explanations, and a recommendation callout — written for a non-technical reader, not just for whoever built it

## Dashboard preview

**Executive Overview**

![Executive Overview](screenshots/executive_overview.jpeg)

*(Further pages added as they're built.)*

## What's next

- [ ] Product & Category page
- [ ] Promotions page
- [ ] Staff page
- [ ] Customer page
- [ ] Full screenshot set