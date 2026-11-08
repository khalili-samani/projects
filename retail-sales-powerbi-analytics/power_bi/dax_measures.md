# DAX Measures — Armani Outlet Sales Model

All measures live in a dedicated measures table, organised into five display folders: Sales, Profitability, Returns, Data Quality, and Time Intelligence. Grouped below in the same order. Several measures reference other measures rather than the fact table directly — where that happens, it's noted, since it means changing a base measure (e.g. `Net Sales`) automatically ripples through everything built on top of it.

---

## Sales

### Transactions
**Purpose:** Counts the number of unique sales, not rows — since a return or exchange can generate more than one line against the same transaction ID, a plain row count would overstate the true number of sales.
```dax
Transactions = DISTINCTCOUNT ( Fact_Sales[Transaction Id] )
```

### Net Sales
**Purpose:** Total revenue after discounts — the headline revenue figure used across every page.
```dax
Net Sales = SUM ( Fact_Sales[Net Sales] )
```

### Gross Sales
**Purpose:** Total revenue before discounts, used alongside Net Sales to calculate the effective discount rate.
```dax
Gross Sales = SUM ( Fact_Sales[Gross Sales] )
```

### Discount Amount
**Purpose:** Total dollar value given away in discounts.
```dax
Discount Amount = SUM ( Fact_Sales[Discount Amount] )
```

### Discount
**Purpose:** Discount given as a percentage of gross sales — this is what powers the "Discount by Promotion" chart on the Promotions page.
```dax
Discount = DIVIDE ( [Discount Amount], [Gross Sales] )
```

### Units Sold
**Purpose:** Total quantity of items sold (returns already reduce this, since `Quantity` is negative on return rows).
```dax
Units Sold = SUM ( Fact_Sales[Quantity] )
```

### Avg Order Value
**Purpose:** Average revenue per transaction — used on the Staff and Customer pages to compare spend patterns.
```dax
Avg Order Value = DIVIDE ( [Net Sales], [Transactions] )
```

### Avg Units per Sale
**Purpose:** Average number of items per transaction.
```dax
Avg Units per Sale = DIVIDE ( [Units Sold], [Transactions] )
```

---

## Profitability

### Gross Profit
**Purpose:** Total profit in dollars — Net Sales less cost of goods sold.
```dax
Gross Profit = SUM ( Fact_Sales[Gross Profit] )
```

### Total Cost
**Purpose:** Total estimated cost of goods sold, shown alongside Gross Profit for context.
```dax
Total Cost = SUM ( Fact_Sales[Estimated Cost] )
```

### Margin
**Purpose:** Profit as a percentage of revenue — the core profitability KPI used to compare categories, promotions, and time periods on a level footing, since dollar profit alone favours high-revenue categories regardless of how efficiently they convert to profit.
```dax
Margin = DIVIDE ( [Gross Profit], [Net Sales] )
```

---

## Returns

### Returned Transactions
**Purpose:** Counts transactions flagged as returns.
```dax
Returned Transactions = 
CALCULATE ( [Transactions], Fact_Sales[Is Return] = TRUE () )
```

### Return Rate
**Purpose:** Returned transactions as a percentage of all transactions — the KPI shown on the Executive Overview.
```dax
Return Rate = 
DIVIDE ( [Returned Transactions], [Transactions] )
```

---

## Data Quality

### Email Capture Rate
**Purpose:** Percentage of transactions where a customer email was captured at point of sale — the KPI underpinning the Customer page's core finding about re-contactability.
```dax
Email Capture Rate = 
DIVIDE (
    CALCULATE ( [Transactions], Fact_Sales[Receipt Email Captured] = TRUE () ),
    [Transactions]
)
```

### Manual Adjustment
**Purpose:** Percentage of transactions logged via manual adjustment rather than the POS system — a data quality indicator, since manually-entered transactions are more exposed to the kind of formatting inconsistencies documented in the cleaning process.
```dax
Manual Adjustment = 
DIVIDE (
    CALCULATE ( [Transactions], Fact_Sales[Source System] = "Manual Adjustment" ),
    [Transactions]
)
```

### Negative Margin Lines
**Purpose:** Counts individual transactions where gross profit is negative — the measure behind the "trench coats, chinos, and boots go margin-negative when heavily discounted" finding on the Product & Category page.
```dax
Negative Margin Lines = 
CALCULATE ( [Transactions], Fact_Sales[Is Negative Margin] = TRUE () )
```

---

## Time Intelligence

### Net Sales PY
**Purpose:** Net Sales for the same period one year earlier, used as the comparison base for year-on-year measures.
```dax
Net Sales PY = 
CALCULATE ( [Net Sales], SAMEPERIODLASTYEAR ( 'Date'[Date] ) )
```

### Net Sales YoY
**Purpose:** Dollar change in Net Sales versus the same period last year.
```dax
Net Sales YoY = 
[Net Sales] - [Net Sales PY]
```

### Net Sales YoY %
**Purpose:** Percentage change in Net Sales versus the same period last year — this is what tells the "is the store growing" story on the Executive Overview.
```dax
Net Sales YoY % = 
DIVIDE ( [Net Sales YoY], [Net Sales PY] )
```

### Gross Profit PY
**Purpose:** Gross Profit for the same period one year earlier.
```dax
Gross Profit PY = 
CALCULATE ( [Gross Profit], SAMEPERIODLASTYEAR ( 'Date'[Date] ) )
```

### Margin % PY
**Purpose:** Margin for the same period one year earlier, so margin trends can be read alongside revenue trends rather than in isolation.
```dax
Margin % PY = 
CALCULATE ( [Margin], SAMEPERIODLASTYEAR ( 'Date'[Date] ) )
```

---

## Notes on the measure set

- **Most measures build on other measures rather than the fact table directly** (e.g. `Avg Order Value` depends on `Net Sales` and `Transactions`; every Time Intelligence measure depends on a Sales or Profitability measure). This keeps the logic DRY, but it also means the dependency order matters — `Net Sales` and `Transactions` are effectively the two foundational measures everything else is built from.
- **`Discount` vs `Discount Amount` vs the `Discount Rate` column in Fact_Sales** — three similarly-named things doing different jobs (a $ column, a $ measure, and a % measure). Worth a quick sanity note if a reviewer asks: `Discount Rate` is the as-recorded per-line rate, `Discount Amount` is the summed dollar total, and `Discount` is the recalculated aggregate percentage — they're all consistent, but the naming is close enough that it's worth being able to explain the distinction on the spot.
- **All Time Intelligence measures rely on the `Date` table being marked as the model's official date table** (continuous, one row per day, correctly related to Fact_Sales) — this is what makes `SAMEPERIODLASTYEAR` behave correctly rather than silently returning blanks.