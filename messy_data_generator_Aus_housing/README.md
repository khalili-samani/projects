# Messy Data Generator for Australian Housing Market

> **Synthetic, uncleaned Australian residential property data — grounded in real market conditions.**

---

## What this is

A single Python script that generates a realistic, intentionally **uncleaned** Australian residential property dataset as a CSV file.

Every run produces a **fresh, unique dataset** — different records, different mess patterns, different missing value distributions. The market-level signals (prices, RBA cash rate, auction clearance rates, listing volume, sentiment) are grounded in real conditions for the period and states you choose.

This is the **input** to a data cleaning pipeline, EDA notebook, or ETL process — not the output.

---

## Why it exists

Anyone working with real Australian property data — scrapes from Domain.com.au, DataVic VPSR reports, CoreLogic exports — immediately encounters dirty, inconsistent, multi-format data. This generator replicates those exact issues synthetically, so you have a reproducible dirty dataset to practise cleaning without needing to scrape anything yourself.

---

## Quick start

```bash
pip install numpy pandas
python generate_messy_housing_data.py
```

You'll be asked three questions:

```
Which year(s)?   2022
Which month(s)?  mar-jun
Which state(s)?  VIC, NSW
```

The CSV is saved automatically to your current working directory.

---

## Input formats

### Years

| Format | Example | Meaning |
|---|---|---|
| Specific (single) | `2021` | That year only |
| Specific (list) | `2021, 2023` | Those two years |
| Interval | `2021-2023` | Every year from 2021 to 2023 inclusive |
| All | `all` | Every available year |

### Months

| Format | Example | Meaning |
|---|---|---|
| Specific (name) | `jan` | January only |
| Specific (number) | `3` | March only |
| Specific (list) | `jan, mar, oct` | Those three months |
| Specific (mixed) | `1, mar, 6` | January, March, June |
| Interval (named) | `jan-jun` | January through June |
| Interval (numeric) | `3-9` | March through September |
| All | `all` | Every month |

### States and territories

| Format | Example | Meaning |
|---|---|---|
| Abbreviation | `VIC` | Victoria only |
| List | `VIC, NSW, QLD` | Those states |
| Full name | `Victoria` | Same as `VIC` |
| All | `all` | All 8 states and territories |

Supported states: `VIC` `NSW` `QLD` `WA` `SA` `TAS` `ACT` `NT`

Full names also accepted: `Victoria`, `New South Wales`, `Queensland`, `Western Australia`, `South Australia`, `Tasmania`, `Australian Capital Territory`, `Northern Territory`, `Tassie`, `Canberra`

**Data range:** January 2020 through to the most recently completed month. The script determines this automatically at runtime — you will never be able to request future data.

---

## Input validation

The script validates every input and re-prompts on error. It will never silently accept bad data.

| What you enter | What happens |
|---|---|
| A year before 2020 | Rejected: *"before the data start date (January 2020)"* |
| A future year | Rejected: *"in the future — no data exists yet"* |
| A non-numeric year like `abc` | Rejected: *"not a valid year — only numbers are accepted"* |
| A reversed year interval like `2023-2021` | Rejected: *"Year interval 2023-2021 is reversed"* |
| A month number outside 1–12 | Rejected: *"must be between 1 (January) and 12 (December)"* |
| An unrecognised month name like `foo` | Rejected: *"not a recognised month name or number"* |
| A reversed month interval like `jun-jan` | Rejected: *"Month interval 'jun-jan' is reversed"* |
| A future month in the current year | Warning shown, future months silently dropped, generation continues with available periods |
| An unrecognised state like `XYZ` | Rejected: *"not a recognised Australian state or territory"* |

---

## Output

The CSV is saved to the directory you run the script from. The filename encodes your selection:

```
aus_housing_messy_<years>_<months>_<states>.csv
```

Examples:
```
aus_housing_messy_2022_mar-jun_vic-nsw.csv
aus_housing_messy_2021-2023_all_wa.csv
aus_housing_messy_2020_jan_all_states.csv
```

Row count is derived automatically — roughly **300 rows per month per selection**, scaled by a market volume factor that reflects real activity levels. Lockdown months generate fewer rows; boom months generate more.

---

## Columns

| Column | Description | Key mess |
|---|---|---|
| `listing_id` | Unique record identifier | — |
| `source` | Platform (Domain, REA, etc.) | Leading/trailing whitespace |
| `address` | Street address | ~8% missing |
| `suburb` | Suburb name | Mixed case, underscores, spaces |
| `state` | State or territory | 7+ variants per state — e.g. `VIC`, `Vic`, `vic`, `Victoria`, `vic.`, `V.I.C` |
| `postcode` | Postcode | Int vs string; ~6% wrong for the suburb |
| `council_area` | Local government area | Mixed case; ~1% swapped with suburb |
| `region` | Metropolitan region | ~12% missing |
| `distance_to_cbd_km` | Distance to CBD | Numeric or `"12.5 km"` string |
| `lat` / `lon` | Coordinates | ~25% missing |
| `property_type` | Property type | 30+ variants of 7 clean types |
| `bedrooms` | Bedroom count | `"3"`, `"three"`, `"3.0"`, `"3 bed"`, `"03"`, `"3 BR"` |
| `bathrooms` | Bathroom count | `"1"`, `"1.0"`, `"1 bath"`, `"1 Bathrooms"` |
| `car_spaces` | Parking spots | Int or `"2 car"` |
| `toilets` | Toilet count | ~20% missing |
| `land_size` | Land area | sqm int, `"650 sqm"`, `"0.065 ha"`, `"650m2"` |
| `building_area` | Floor area | ~35% missing; sometimes `"203sqm"` |
| `year_built` | Year built | ~28% missing; outliers: 1066, 9999, 2099 |
| `has_pool` | Pool present | `Yes/yes/Y/True/true/1` vs `No/no/N/False/false/0` |
| `has_garage` | Garage present | Same boolean mess |
| `sale_price` | Sale price (string) | `"$1.25M"`, `"Contact Agent"`, `"POA"`, `"Offers Over $800,000"` |
| `price_raw_aud` | Numeric price if extractable | ~6% missing or null string |
| `sale_date` | Date of sale | 9 different date formats mixed throughout |
| `sale_method` | Auction / Private Sale / EOI | Multiple abbreviation variants |
| `days_on_market` | Days listed before sale | Int or `"N/A"`, `"-"`, `"unknown"` |
| `inspection_note` | Inspection details | Lockdown periods inject virtual inspection notes |
| `agent_name` | Agent name | Mixed case; ~8% missing |
| `agency_name` | Agency name | Same name as `agent_name` but different casing |
| `agent_phone` | Contact number | Formatted vs unformatted |
| `rba_cash_rate_pct` | RBA cash rate in effect | Real rate for the record's month |
| `market_sentiment` | Buyer sentiment label | Panic / FOMO / Falling / Resilient / Strong etc. |
| `market_context` | One-line real-world context | ~15% missing |
| `suburb_median_price` | Suburb median (AUD) | ~18% missing |
| `auction_clearance_rate_pct` | Clearance rate | Real period average ± noise; ~30% missing |
| `weekly_rent_aud` | Estimated weekly rent | ~45% missing |
| `property_count_suburb` | Properties in suburb | ~15% missing |

---

## Mess catalogue

15 data quality issues deliberately injected:

1. **`sale_price` as string** — `"$1,250,000"`, `"$1.25M"`, `"Contact Agent"`, `"POA"`, `"Offers Over $800,000"`
2. **Suburb casing** — `"Richmond"`, `"RICHMOND"`, `"richmond"`, `" richmond"`, `"richmond_"`
3. **State variants** — 7+ representations per state (e.g. `VIC`, `Vic`, `vic`, `Victoria`, `victoria`, `vic.`, `V.I.C`)
4. **Mixed date formats** — 9 formats: `DD/MM/YYYY`, `YYYY-MM-DD`, `DD Mon YYYY`, `Mon-YY`, and more
5. **Property type fragmentation** — 30+ variants: `"h"`, `"House"`, `"HOUSE"`, `"house,cottage,villa"`, `"Residential House"`
6. **Bedrooms as mixed types** — `"3"`, `"three"`, `"3.0"`, `"3 bed"`, `"03"`, `"3 BR"`, `-1` (outlier)
7. **Land size units** — sqm integer, `"650 sqm"`, `"0.065 ha"`, `"650m2"`, `"0.065ha"`
8. **Boolean inconsistency** — `"Yes"`, `"yes"`, `"Y"`, `"True"`, `"true"`, `"1"` all mean the same thing
9. **Null proliferation** — `NaN`, `"N/A"`, `"n/a"`, `"NA"`, `"-"`, `"--"`, `"unknown"`, `" "`, `"?"`
10. **Postcode mismatches** — ~6% of rows have a postcode that doesn't match the suburb
11. **Year built outliers** — `1066`, `9999`, `2099`, `0`
12. **Near-duplicate rows** — ~2.5% are re-listings of the same property with minor differences
13. **Exact duplicate rows** — ~1% exact copies (common in multi-source scrapes)
14. **Source whitespace** — `"Domain "`, `" REA"` (leading/trailing spaces)
15. **Field confusion** — ~1% of rows have `suburb` and `council_area` swapped

---

## How market conditions affect the data

The generated data is not uniformly random. Each month's records reflect real conditions for that period.

| Period | RBA rate | What it affects |
|---|---|---|
| Mar–Apr 2020 | 0.25% | Auction volumes ~40% of normal; clearance rates ~25–30%; lockdown virtual inspection notes injected for VIC |
| Nov 2020 – Apr 2021 | 0.10% | HomeBuilder rush; regional tree-change price premium; volumes recovering |
| 2021 | 0.10% | Peak FOMO; prices +13–19% above Jan 2020 base; boom volumes; very short days-on-market |
| Jul–Oct 2021 | 0.10% | Melbourne lockdowns #5 and #6; Sydney Delta lockdown; auction volumes suppressed |
| May–Dec 2022 | 0.35% → 3.10% | Prices falling; clearance rates 43–56%; longer days-on-market; `Falling` / `Distressed` sentiment |
| 2023 | 3.10% → 4.35% | Surprise recovery despite high rates; `Resilient` sentiment; supply-constrained |
| 2024 | 4.35% → 4.10% | Melbourne flat; elevated days-on-market; affordability stress; Dec cut lifts confidence |
| 2025 | 4.35% → 3.60% | Three RBA cuts; renewed confidence; prices +8.6%; `Strong` sentiment returns |

---

## Suburb coverage by state

Real suburbs with accurate postcodes, council areas, and price anchors:

| State | Example suburbs |
|---|---|
| VIC | Richmond, Toorak, Brighton, Footscray, Geelong, Ballarat, Pakenham |
| NSW | Surry Hills, Newtown, Bondi, Manly, Parramatta, Penrith |
| QLD | Paddington, Sunnybank, Gold Coast, Sunshine Coast, Toowoomba |
| WA | Subiaco, Cottesloe, Fremantle, Joondalup, Mandurah, Bunbury |
| SA | Norwood, Unley, Glenelg, Prospect, Tea Tree Gully, Port Adelaide |
| TAS | Sandy Bay, Glenorchy, Launceston, Devonport |
| ACT | Braddon, Gungahlin, Tuggeranong, Woden Valley |
| NT | Darwin CBD, Palmerston, Alice Springs |

---

## What to do with the output

This CSV is designed as the starting point for:

- **Data cleaning notebook** — parse prices, standardise dates, resolve nulls, deduplicate, fix units
- **MySQL / PostgreSQL ETL** — load after cleaning for dashboard or BI tool connection
- **Power BI / Tableau** — connect post-cleaning for visualisation
- **EDA** — explore distributions, outliers, missing patterns before cleaning
- **ML feature engineering** — encode property type, impute missing values, engineer suburb features

---

## Data realism

- **Suburbs** are real properties from all 8 Australian states and territories with correct council areas and postcodes
- **Price medians** reflect real 2020–2025 market levels per suburb, anchored to CoreLogic / Domain data
- **RBA cash rates** are the actual rates in effect each month (verified against RBA records)
- **Auction clearance rates** are calibrated to real Melbourne and Sydney historical averages
- **Missing rates** reflect realistic scrape incompleteness (`building_area` ~35%, `lat/lon` ~25%)
- **Listing volumes** scale with real market activity (lockdown months fewer rows, boom months more)
- **Every run is unique** — seeded from the system clock so no two outputs are identical

---

## Dependencies

```
numpy>=1.26
pandas>=2.1
```

No API keys. No data downloads. No internet connection required.

---