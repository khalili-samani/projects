# Architecture

## Selected platform

MySQL 8.0+ is used because it is free, strong for analytical SQL, and supports regular expressions, generated identity keys, refreshable aggregate tables, indexing, procedures and transactional DDL.

## Layers

1. **raw**: append-only source rows and batch metadata; all source fields remain text.
2. **stg**: standardised and typed records with parse flags, quality scores and duplicate classifications.
3. **dw**: conformed dimensions, rejected-record audit and the canonical sale fact.
4. **mart**: business-facing aggregate tables and views.
5. **audit**: test results and load reconciliation.

## Data flow

```mermaid
flowchart LR
    A[Original Python generator] -->|synthetic CSV| B[Python loader]
    B --> C[(raw.ingestion_batch)]
    B --> D[(raw.housing_listing)]
    D --> E[stg.v_housing_standardised]
    E --> F[stg.housing_ranked]
    F --> G[(dw dimensions)]
    F --> H[(dw.fact_property_sale)]
    F --> I[(dw.rejected_record)]
    H --> J[(mart.monthly_market_summary)]
    H --> K[(mart.segment_performance)]
    G --> J
    G --> K
    J --> L[BI / portfolio analysis]
    K --> L
```

## Incremental strategy

The loader calculates a SHA-256 file checksum. A successful checksum is not loaded twice unless `--force` is used. Transformations upsert dimensions by natural key and insert fact rows using a deterministic `event_hash`. This makes reruns idempotent while allowing new batches.

## Privacy and governance

The source is synthetic. Agent and address fields are nevertheless treated as quasi-identifiers to demonstrate privacy-aware design. Reporting marts omit phone numbers and full addresses. The README warns against using outputs for valuation, investment or identification.
