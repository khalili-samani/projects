# Business requirements

## Business context

A hypothetical Australian property analytics team receives monthly synthetic listing and sale extracts from multiple source systems. The extracts are structurally stable but operationally messy. Analysts need a governed warehouse layer before building BI reports.

## Stakeholders

| Stakeholder | Need |
|---|---|
| Head of Analytics | Trustworthy trend and KPI reporting with visible caveats |
| BI analyst | Stable dimensions, measures and reporting views |
| Data quality lead | Defect counts, severity, resolution status and reconciliation |
| Operations manager | Source, agency and days-on-market performance comparisons |
| Analytics engineer | Idempotent transformations, tests and incremental logic |

## Key questions

1. How many source rows are accepted, rejected, deduplicated or unresolved each load?
2. How do median price, transaction volume and days on market change month over month?
3. Which states, regions, property types and bedroom segments drive changes?
4. Which cohorts sell faster or slower than comparable listings?
5. What is the indicative gross rental yield, and how does it vary by segment?
6. Which source systems or agencies have the highest data-quality issue rates?
7. How sensitive are reported metrics to exclusions caused by invalid prices or dates?

## Functional requirements

- Retain immutable source records.
- Support repeatable batch loads with a unique batch identifier and file checksum.
- Normalise known missing markers before parsing.
- Store both parsed values and parse-status flags.
- Keep one canonical record per business event while preserving duplicate lineage.
- Publish marts at monthly geography and property-segment grains.
- Fail tests for broken keys, impossible dates, negative measures or reconciliation gaps.

## Non-functional requirements

- MySQL 8.0+ compatibility.
- SQL scripts runnable in documented order with `ON_ERROR_STOP`.
- Snake_case names, UTC ingestion timestamps and explicit numeric precision.
- No confidential or personally identifiable real-world data.
