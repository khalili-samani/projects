# Data dictionary

## `raw.ingestion_batch`

| Column | Type | Description |
|---|---|---|
| `batch_id` | bigint PK | Load identifier |
| `source_file_name` | text | Original CSV name |
| `source_file_sha256` | char(64) | Idempotency checksum |
| `source_system` | text | Generator/source label |
| `loaded_at_ts` | timestamp(6) | UTC load timestamp |
| `row_count` | integer | Rows read by loader |
| `status` | text | started, loaded or failed |
| `error_message` | text | Failure detail |

## `raw.housing_listing`

Grain: one physical CSV row. The 37 source columns are text, plus `raw_row_id`, `batch_id`, `source_row_number`, `source_row_hash` and `loaded_at_ts`.

## `stg.housing_ranked`

Grain: one raw row after standardisation and duplicate ranking. Important fields include parsed dates, numeric measures, canonical categories, rule flags, `quality_score`, `duplicate_class`, `survivorship_rank` and `record_disposition`.

## Dimensions

| Table | Grain | Key | Important attributes |
|---|---|---|---|
| `dw.dim_date` | one calendar date | `date_key` integer YYYYMMDD | month, quarter, financial year, month-end flags |
| `dw.dim_geography` | one canonical suburb/postcode/state combination | `geography_key` | suburb, postcode, council, region, state |
| `dw.dim_property_type` | one canonical type | `property_type_key` | code, display name, dwelling group |
| `dw.dim_source` | one source system label | `source_key` | source name |
| `dw.dim_agent` | one version of agent/agency attributes | `agent_key` | natural hash, name, agency, masked phone, SCD2 dates |

## `dw.fact_property_sale`

Grain: one accepted canonical property-sale/listing event after survivorship.

| Column | Type | Description |
|---|---|---|
| `sale_key` | bigint PK | Surrogate fact key |
| `event_hash` | char(32) unique | Idempotent event identifier |
| dimension keys | bigint/int FK | Date, geography, property type, source and agent |
| `listing_id` | text | Source listing identifier |
| `sale_price_aud` | numeric(14,2) | Preferred canonical sale price |
| `price_source` | text | Source used for price |
| `days_on_market` | integer | Listing duration |
| room/area fields | numeric/integer | Canonical property attributes |
| market context fields | numeric/text | Synthetic monthly settings from source |
| derived ratios | numeric | Rental yield and price-to-median ratio |
| quality fields | numeric/text | Score, warning count and duplicate lineage |
| `batch_id`, `raw_row_id` | bigint | Audit lineage |

## Marts

| Table | Grain | Measures |
|---|---|---|
| `mart.monthly_market_summary` | sale month × state × property type | sales, median/average price, median DOM, yield, MoM and rolling metrics |
| `mart.segment_performance` | sale month × state × bedroom segment × price band | sales, median price, fast-sale rate, above-median rate |
| `mart.data_quality_summary` | batch × disposition × duplicate class | rows, average score, issue counts and rates |

## Actual-file extension

| Table | Column | Type | Definition |
|---|---|---|---|
| `dw.fact_property_sale` | `sale_date_precision` | `ENUM('day','month','unknown')` | Indicates whether the source supplied a complete date or only month and year. Month precision values use day 1 solely for the date key. |
