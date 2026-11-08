# Data Dictionary — Armani Outlet Sales Model

This covers the full Power BI star schema: one fact table (`Fact_Sales`) and five dimension tables (`Date`, `Product`, `Staff`, `Promotion`, `Customer Type`). Column names below are given exactly as they appear in the model — every table's last Power Query step renames columns from `snake_case` to Title Case for display.

`data/cleaned/clean_sales.csv` is an export of the final `Fact_Sales` table only. The dimension tables are generated inside the Power BI model directly from `Fact_Sales` — they don't exist as separate source files, so they're documented here rather than as their own CSVs.

See [`power_bi/power_query_steps.md`](../power_bi/power_query_steps.md) for the full step-by-step cleaning logic behind every column below.

---

## Fact_Sales

| Column | Type | Source / Derived | Description | Cleaning applied |
|---|---|---|---|---|
| `Transaction Id` | Text | Source | Unique identifier for each sale, e.g. `TXN-202787` | `-MANUAL` suffix stripped and converted to uppercase, so manually-adjusted rows dedup correctly against their original; rows with no ID were dropped entirely |
| `Transaction Date` | Date | Source | Date of sale | Parsed first as `D/M/YYYY`, then, where that failed, as `DD-Mon-YY`; unparseable dates left null rather than guessed |
| `Transaction Time` | Text | Source | Time of sale (24hr) | Whitespace/hidden characters trimmed only |
| `Store Name` | Text | Source (hardcoded) | Outlet name | Every row is the same single outlet, so hardcoded to "Armani Outlet" rather than reconciling 10+ spelling/casing variants |
| `Centre` | Text | Source (hardcoded) | Shopping centre name | Hardcoded to "DFO South Wharf" for the same reason |
| `Suburb` | Text | Source (hardcoded) | Suburb | Hardcoded to "South Wharf" |
| `Product Code` | Text | Source (recovered where missing) | Internal SKU, category embedded in the middle 3-letter prefix (e.g. `AX-MEN-SUI-006`) | Where blank, recovered via a style-key lookup built from matching product names elsewhere in the data |
| `Category` | Text | Source (recovered where missing) | Product category: Menswear, Womenswear, Outerwear, Shoes, Accessories | Converted to proper case; where blank, recovered from the `Product Code` prefix (`MEN`, `OUT`, `SHO`, `WOM`, `ACC`) |
| `Product Name` | Text | Source | Descriptive product name, colour + style pattern (e.g. "White Suit Jacket") | Converted to proper case; used as the pattern-matching source for recovering missing product codes |
| `Size` | Text | Source | Size recorded at sale | Whitespace/hidden characters trimmed only |
| `Colour` | Text | Source | Product colour | Converted to proper case |
| `Quantity` | Whole number | Source (corrected) | Units sold on the line; negative for returns | Converted to a whole number; sign corrected to negative wherever `Is Return = true` but quantity was recorded as positive |
| `Unit Price` | Decimal | Source | Price per unit before discount | Currency symbols (`$`, `AUD`) and thousands commas stripped, converted to numeric |
| `Gross Sales` | Decimal | Source | `Unit Price × Quantity`, before discount | Same currency cleaning as `Unit Price` |
| `Discount Amount` | Decimal | Source | Dollar value of discount applied | Same currency cleaning as `Unit Price` |
| `Discount Rate` | Decimal | Source | Discount as a fraction of gross sales | Standardised so values recorded as a percentage (e.g. `22%` or `22`) and values already recorded as a fraction (e.g. `0.22`) all resolve to the same decimal |
| `Net Sales` | Decimal | Source (recovered where missing) | Revenue after discount | Currency-cleaned; where blank, recalculated as `Gross Sales − Discount Amount` rather than left null or dropped |
| `Estimated Cost` | Decimal | Source | Estimated cost of goods sold | Same currency cleaning as `Unit Price` |
| `Gross Profit` | Decimal | Source | `Net Sales − Estimated Cost` | Same currency cleaning as `Unit Price` |
| `Payment Method` | Text | Source | Payment method at POS | Converted to proper case |
| `Customer Type` | Text | Source | Customer segment: Corporate, VIP, Tourist, Loyalty Member, Walk-in | Left as recorded (standardisation happens implicitly via proper-case elsewhere in the model; see the `Customer Type` dimension table) |
| `Customer Id` | Text | Source | Customer identifier, where captured | Left null where missing — a genuine data gap (no ID captured at sale), not a formatting issue, so not imputed or dropped |
| `Staff Id` | Text | Source | Internal staff identifier | Used as the lookup key to resolve `Staff Member` name variants |
| `Staff Member` | Text | Source (recovered) | Staff name as recorded at POS | Replaced with a canonical name via a 10-row lookup table keyed on `Staff Id`, resolving the 2–3 spelling variants recorded per person |
| `Promotion` | Text | Source (filled where missing) | Campaign active at time of sale | Blank values filled to "No Campaign"; "Melbourne Fashion Weekend" shortened to "Fashion Week" for chart display |
| `Is Return` | True/False | Source | Flags whether the transaction is a return | Mixed formats (`1`/`0`, `Y`/`N`, `Yes`/`No`, `TRUE`/`FALSE`) standardised to a single boolean; used as the source of truth for the `Quantity` sign correction |
| `Receipt Email Captured` | True/False | Source | Whether an email was captured at time of sale | Same boolean standardisation as `Is Return`; underpins the Email Capture Rate measure |
| `Pos Terminal` | Text | Source | Terminal ID the sale was processed on | Whitespace/hidden characters trimmed only |
| `Source System` | Text | Source | System that logged the transaction (POS, Manual Adjustment) | Retained as-is; distinguishes manually-adjusted transactions during dedup |
| `Notes` | Text | Source | Free-text note on the transaction, where present | Whitespace/hidden characters trimmed only; not used in any measure |
| `Margin Pct` | Decimal | **Derived** | `Gross Profit ÷ Net Sales` | Calculated column added after cleaning; left null where `Net Sales` is null or zero to avoid a divide-by-zero error |
| `Is Negative Margin` | True/False | **Derived** | Flags any transaction where `Gross Profit` is negative | Calculated column added after cleaning; supports the "which products go margin-negative when discounted" analysis |

**Rows removed entirely:** rows with no `Transaction Id`; exact duplicates and `-MANUAL` duplicates (70 + 16 rows) collapsed during dedup.

**A helper column that doesn't survive to the final table:** a `style_key` column (product name with the leading colour word stripped) is created mid-query purely to power the product code recovery lookup, then removed once its job is done — it won't appear if you inspect the final table or export.

---

## Date

A calendar table generated in Power Query from the min/max of `Fact_Sales[Transaction Date]`, not imported from a source file.

| Column | Type | Source / Derived | Description |
|---|---|---|---|
| `Date` | Date | Derived | One row per calendar day, spanning every day between the earliest and latest transaction date (inclusive) — including days with no sales |
| `Year` | Whole number | Derived | Calendar year, e.g. 2024 |
| `Month Number` | Whole number | Derived | Month as a number (1–12), used for sorting |
| `Month` | Text | Derived | 3-letter month abbreviation, e.g. "Jan" |
| `Quarter` | Text | Derived | Calendar quarter, e.g. "Q1" |
| `Financial Year` | Text | Derived | Australian financial year (July–June), e.g. July 2024 → "FY2025" |
| `Year Month` | Text | Derived | Readable month label, `yyyy-MM` format (e.g. "2024-07") — used on the Executive Overview trend chart |
| `Year Month Sort` | Whole number | Derived | Numeric sort key (`Year × 100 + Month Number`) so `Year Month` labels sort chronologically, not alphabetically |

**Note:** because the calendar range is derived from the data itself rather than fixed, an unexplained future-dated transaction in the raw export (see Fact_Sales data quality notes) would silently extend this table's range too.

---

## Product

A dimension table collapsing `Fact_Sales` to one row per unique product code — there's no separate product source file.

| Column | Type | Source / Derived | Description |
|---|---|---|---|
| `Product Code` | Text | Derived (from Fact_Sales) | Unique SKU, one row per code |
| `Category` | Text | Derived (from Fact_Sales) | Category for that product code, taken from the first matching row in Fact_Sales |
| `Product Name` | Text | Derived (from Fact_Sales) | Colour-free style name (e.g. "Suit Jacket"), taken from the first matching row in Fact_Sales |

**Note:** `Category` and `Product Name` are each taken from the *first* row seen for a given product code, on the assumption every row sharing that code describes the same product consistently. This is the same assumption used in the Fact_Sales product code recovery lookup — see `power_query_steps.md` for the caveat on what happens if that assumption is ever wrong.

---

## Staff

A dimension table listing each unique staff member, built from the already-cleaned `Fact_Sales` table.

| Column | Type | Source / Derived | Description |
|---|---|---|---|
| `Staff Id` | Text | Source | Internal staff identifier, one row per person (10 total) |
| `Staff Member` | Text | Derived (from Fact_Sales) | Canonical staff name — already resolved via the ID-keyed lookup in Fact_Sales, so no further name-matching happens here |

---

## Promotion

A dimension table listing each unique campaign, built from the already-cleaned `Fact_Sales` table.

| Column | Type | Source / Derived | Description |
|---|---|---|---|
| `Promotion` | Text | Derived (from Fact_Sales) | One row per unique campaign name, including "No Campaign"; already standardised upstream in Fact_Sales |

---

## Customer Type

A dimension table listing each unique customer segment, built from the already-cleaned `Fact_Sales` table.

| Column | Type | Source / Derived | Description |
|---|---|---|---|
| `Customer Type` | Text | Derived (from Fact_Sales) | One row per unique segment: Corporate, VIP, Tourist, Loyalty Member, Walk-in |

---

## Notes on data not corrected

- **Missing `Customer Id`** was treated as a genuine finding (see README), not a defect to clean away.
- **Inconsistent `Suburb` values** in a small number of raw rows (e.g. "Melbourne" instead of "South Wharf") are overwritten by the hardcoded suburb value in Fact_Sales, so this doesn't surface as an issue downstream — but it's worth knowing the raw export contains it.
- **Unparseable transaction dates** are left null rather than guessed, so they can be surfaced by a Data Quality measure rather than silently distorting time-based analysis.