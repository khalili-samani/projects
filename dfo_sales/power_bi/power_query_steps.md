# Power Query Steps — Plain-English Summary

This file mirrors the Applied Steps pane in Power Query for each table in the model, in order, written in plain English so the cleaning logic can be followed without reading M code. See the [data dictionary](../data/data_dictionary.md) for column-level detail and the README for the business context behind each decision.

---

## Fact_Sales

1. **Loaded raw sales export** — Imported `raw_sales.csv` (~6,300 rows).
2. **Promoted first row to headers** — Used the first row of the CSV as column names.
3. **Trimmed spaces and hidden characters from text** — Removed leading/trailing whitespace and non-printable characters from every text column, so later matching (categories, promotions, staff names) isn't broken by invisible formatting differences.
4. **Standardised transaction ID format** — Stripped the `-MANUAL` suffix from manually-adjusted transaction IDs so they collapse into the same ID as their original when deduplicated later.
5. **Removed rows with no transaction ID** — Dropped any row missing a transaction ID entirely, since it can't be reliably tied to a sale.
6. **Replaced blank cells with nulls** — Converted empty-string cells to proper nulls across every column, so missing-value logic downstream (in category, product code, customer ID, etc.) works consistently.
7. **Standardised store and centre names** — Since every row is the same single outlet, hardcoded `store_name` to "Armani Outlet" and `centre` to "DFO South Wharf" rather than trying to reconcile 10+ spelling/casing variants.
8. **Standardised suburb** — Hardcoded to "South Wharf" for the same reason.
9. **Standardised product name casing** — Converted product names to proper case (e.g. "white suit jacket" → "White Suit Jacket").
10. **Standardised colour names** — Converted colour values to proper case.
11. **Added style key for product code lookup** — Built a helper column that strips the leading colour word off each product name (e.g. "White Suit Jacket" → "suit jacket"), so products can be matched to a code by style alone, regardless of colour.
12. **Built product code lookup table** — Created a reference table of style → product code, using whichever product code was already present on a matching style elsewhere in the data.
13. **Merged product code lookup** — Joined that reference table back onto every row by style key.
14. **Filled missing product code from product name** — Where `product_code` was blank, filled it from the matched lookup; otherwise kept the original value.
15. **Replaced product code with recovered version** — Swapped in the recovered/original product code and removed the now-unneeded helper columns.
16. **Standardised category names** — Converted category values to proper case.
17. **Filled missing category from product code** — Where `category` was blank, recovered it from the 3-letter prefix embedded in the product code (e.g. `MEN` → Menswear, `OUT` → Outerwear).
18. **Replaced category with recovered version** — Swapped in the recovered/original category and dropped the intermediate column.
19. **Standardised payment methods** — Converted payment method values to proper case.
20. **Standardised promotion campaign** — Filled any blank promotion value with "No Campaign" rather than leaving it null, since "no promotion running" is itself a meaningful category for comparison.
21. **Shortened long promotion names for chart display** — Renamed "Melbourne Fashion Weekend" to "Fashion Week" so it displays cleanly on chart axes.
22. **Cleaned currency symbols from money columns** — Stripped `$`, `AUD`, and thousands-separator commas from `unit_price`, `gross_sales`, `discount_amount`, `net_sales`, `estimated_cost`, and `gross_profit`.
23. **Converted money columns to numbers** — Converted all six columns above from text to proper numeric type.
24. **Filled missing net sales from gross sales less discount** — Where `net_sales` was blank, recalculated it as `gross_sales − discount_amount` rather than leaving it null or dropping the row.
25. **Normalised discount rate to a fraction** — Standardised `discount_rate` so values recorded as a percentage (e.g. `22%` or `22`) and values already recorded as a fraction (e.g. `0.22`) all end up as the same decimal fraction.
26. **Converted transaction dates (two formats) to real dates** — Parsed `transaction_date` first as `D/M/YYYY`, then, where that failed, as `DD-Mon-YY`; anything matching neither format was left null rather than guessed.
27. **Converted Yes/No flags to true or false** — Standardised `is_return` and `receipt_email_captured`, each recorded inconsistently as `1`/`0`, `Y`/`N`, `Yes`/`No`, or `TRUE`/`FALSE`, into a single consistent true/false value.
28. **Converted quantity to a whole number** — Converted `quantity` from text to a proper whole-number type.
29. **Replaced quantity with corrected sign** — Where a row was flagged as a return (`is_return = true`) but `quantity` was still recorded as positive, flipped it negative; left all other rows unchanged.
30. **Built staff name lookup table** — Created a small reference table mapping each of the 10 staff IDs to one canonical staff name.
31. **Merged staff name lookup** — Joined that reference table onto every row by `staff_id`.
32. **Replaced staff member with canonical name** — Replaced the as-recorded `staff_member` value (which had 2–3 spelling variants per person) with the canonical name from the lookup, matched on `staff_id` rather than on the name itself.
33. **Removed duplicate transactions** — Deduplicated on `transaction_id`, now that the `-MANUAL` suffix has been stripped, so both types of duplicate collapse to a single row.
34. **Added profit margin %** — Added a calculated column: `gross_profit ÷ net_sales`, left null where `net_sales` is null or zero to avoid a divide-by-zero error.
35. **Flagged negative-margin sales** — Added a true/false column marking any transaction where `gross_profit` is negative, used to support the "which products go margin-negative when discounted" analysis.
36. **Set final data types** — Locked in the correct type for `transaction_id`, `customer_id`, `staff_id`, and `transaction_time`.
37. **Renamed columns to title case** — Renamed every column from `snake_case` to Title Case (e.g. `transaction_date` → "Transaction Date") for clean display in report visuals.

**A note on step 12 (product code lookup):** where more than one row shares the same style key, the lookup takes the first matching code found (`List.First`) rather than checking they all agree. Worth a quick sense-check that no style genuinely spans two different product codes — if a hiring manager or reviewer asks how collisions are handled, this is the honest answer.

---

## Date

A standalone calendar table generated from the range of transaction dates, rather than imported from a source file — this is what powers all the Time Intelligence measures and the month/year axis on the Executive Overview page.

1. **Referenced Fact_Sales dates** — Pulled the `Transaction Date` column from Fact_Sales, with nulls removed (so any unparseable dates from the raw export don't distort the calendar range).
2. **Found earliest transaction date** — Identified the minimum date in that list.
3. **Found latest transaction date** — Identified the maximum date in that list.
4. **Counted days in range** — Calculated the number of days between earliest and latest date, inclusive.
5. **Generated date list** — Built a complete, continuous list of every calendar day between the earliest and latest transaction date — including days with no sales, so charts don't show gaps or misleading averages.
6. **Converted list to table** — Turned that list into a single-column table.
7. **Set date column type** — Set the `Date` column to a proper date type.
8. **Added year** — Added a `Year` column (e.g. 2024).
9. **Added month number** — Added a `Month Number` column (1–12), used for sorting rather than display.
10. **Added month name** — Added a `Month` column as a 3-letter abbreviation (e.g. "Jan").
11. **Added quarter** — Added a `Quarter` column (e.g. "Q1").
12. **Added financial year** — Added a `Financial Year` column on an Australian FY basis (July–June), so July onward rolls into the next financial year (e.g. July 2024 → "FY2025").
13. **Added year month** — Added a `Year Month` column formatted as `yyyy-MM` (e.g. "2024-07"), used as the readable label on the Executive Overview trend chart.
14. **Added year month sort key** — Added a numeric `Year Month Sort` column (`Year × 100 + Month Number`) so the Year Month labels sort chronologically rather than alphabetically in visuals.
15. **Locked final date type** — Re-confirmed the `Date` column type after all the additions above.
16. **Renamed columns to title case** — Renamed every column to Title Case for clean display in report visuals.

**A note on this table:** because it's generated from the min/max of transaction dates rather than a fixed calendar range, if the raw export ever grows a transaction dated well outside the real business period (e.g. an unexplained future-dated row — see the README's data quality section), the calendar will silently stretch to include it. That's a deliberate trade-off: it keeps the Date table always in sync with the data without manual updates, but it's worth knowing the calendar range isn't independently fixed.

---

## Product

A dimension table built by collapsing Fact_Sales down to one row per unique product, rather than imported separately — since the raw export only ever recorded product details at transaction level.

1. **Referenced Fact_Sales query** — Started from the already-cleaned Fact_Sales table (so this table inherits the recovered product codes and categories from that cleaning, rather than the raw values).
2. **Added colour-free style name** — Rebuilt the same "strip the leading colour word" logic used in Fact_Sales, to get a clean style name (e.g. "Suit Jacket") independent of colour, for display purposes.
3. **Selected product columns** — Reduced the table down to just `Product Code`, `Category`, and the new `style_name` column.
4. **Removed rows with no product code** — Dropped any row where `Product Code` is still null even after the recovery logic in Fact_Sales (i.e. products that couldn't be identified by either code or name).
5. **Grouped to one row per product code** — Collapsed the table to one row per unique `Product Code`, taking the first `Category` and first `style_name` seen for that code as the representative value.
6. **Renamed columns to title case** — Renamed every column to Title Case for clean display in report visuals.

**A note on this table:** because grouping takes the *first* category and style name seen for each product code (`List.First`), it assumes every row sharing a product code genuinely describes the same product with the same category — the same assumption flagged for the Fact_Sales product code lookup. If a product code were ever recorded against two different categories (a data entry error rather than a formatting one), this table would silently keep whichever came first and mask the inconsistency. Worth a quick validation check (e.g. a Power Query step counting distinct categories per product code) if this becomes a live/updating dataset rather than a fixed snapshot.

---

## Staff

A simple dimension table listing each unique staff member, built from the already-cleaned Fact_Sales table.

1. **Referenced Fact_Sales query** — Started from the cleaned Fact_Sales table, so this table inherits the canonical staff names already resolved via the staff ID lookup in Fact_Sales, rather than the raw, inconsistent name spellings.
2. **Selected staff columns** — Reduced the table down to just `Staff Id` and `Staff Member`.
3. **Removed rows with no staff ID** — Dropped any row missing a staff ID.
4. **Removed duplicate staff** — Deduplicated to one row per unique `Staff Id`.
5. **Renamed columns to title case** — Renamed every column to Title Case for clean display in report visuals.

**A note on this table:** because it's built after Fact_Sales has already standardised staff names via the ID-keyed lookup table, this table doesn't need to repeat any name-matching logic itself — it's simply picking up the one canonical name per ID that's already correct by the time it gets here.

---

## Promotion

A simple dimension table listing each unique promotional campaign, built from the already-cleaned Fact_Sales table.

1. **Referenced Fact_Sales query** — Started from the cleaned Fact_Sales table, so this table inherits the already-standardised promotion names (including blanks filled to "No Campaign" and "Melbourne Fashion Weekend" shortened to "Fashion Week") rather than the raw, inconsistent values.
2. **Selected promotion column** — Reduced the table down to just `Promotion`.
3. **Removed rows with no promotion** — Dropped any remaining null (in practice, none should remain given "No Campaign" was already filled in during Fact_Sales cleaning).
4. **Removed duplicate promotions** — Deduplicated to one row per unique campaign name.
5. **Renamed columns to title case** — Renamed the column to Title Case for clean display in report visuals.

**A note on this table:** this is the simplest dimension table in the model — one column, deduplicated — since Fact_Sales had already done all the standardisation work. It exists mainly to give the Promotion field a proper one-to-many relationship to Fact_Sales, so slicers and filters on this table behave correctly.

---

## Customer Type

A simple dimension table listing each unique customer segment, built from the already-cleaned Fact_Sales table.

1. **Referenced Fact_Sales query** — Started from the cleaned Fact_Sales table, so this table inherits the already-standardised customer type values (Corporate, VIP, Tourist, Loyalty Member, Walk-in) rather than the raw, inconsistent spelling/casing variants.
2. **Selected customer type column** — Reduced the table down to just `Customer Type`.
3. **Removed rows with no customer type** — Dropped any row missing a customer type.
4. **Removed duplicate customer types** — Deduplicated to one row per unique segment.
5. **Renamed columns to title case** — Renamed the column to Title Case for clean display in report visuals.

**A note on this table:** same pattern as Promotion — a minimal one-column dimension table, existing to give Customer Type a proper relationship to Fact_Sales for slicers and filters, since all the actual standardisation happened upstream.

---

This completes the model's five dimension tables (Date, Product, Staff, Promotion, Customer Type) plus the Fact_Sales fact table. Next: [`power_bi/dax_measures.md`](dax_measures.md) for how these tables are used to calculate the report's KPIs.