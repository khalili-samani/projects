# Delivery Report: MySQL Conversion

The project has been converted from PostgreSQL to MySQL 8.0+. The conversion covers database setup, stored functions, raw ingestion, staging views, window-based survivorship, dimensional models, SCD Type 2 agent handling, fact loading, refreshable mart tables, reporting views, stored-procedure tests, optimisation examples, Python connectors, Docker Compose, Make targets and documentation.

## Key MySQL design decisions

- MySQL schemas are implemented as separate databases: `raw`, `stg`, `dw`, `mart` and `audit`.
- `AUTO_INCREMENT` replaces identity columns.
- `ON DUPLICATE KEY UPDATE` and `INSERT IGNORE` replace PostgreSQL conflict clauses.
- Refreshable physical mart tables replace materialised views.
- MySQL 8 window functions support deduplication, survivorship, rolling metrics and ranking.
- JSON stores rejection reasons because MySQL has no native array type.
- `mysql-connector-python` replaces `psycopg`.

## Validation performed

Python scripts were syntax checked. Repository references were scanned for PostgreSQL-specific runtime dependencies. SQL was reviewed for MySQL 8.0+ syntax and dependency order, but not executed against a live MySQL server in this environment.
