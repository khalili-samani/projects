-- =============================================================================
-- schema.sql
-- Sector-Based Stock Market Analysis
-- =============================================================================
--
-- Reference document defining the full database schema for this project.
--
-- The Python pipeline (scripts/load_data.py) handles database setup
-- automatically — you do not need to run this file manually to use the
-- project. It is provided here as a readable reference for the database
-- design, and can be used to inspect or recreate schema objects independently.
--
-- Objects defined:
--   1. Database
--   2. Dimension tables  — dim_sector, dim_company
--   3. Fact table        — fact_stock_prices
--   4. Reference data    — sector and company inserts
--   5. Analytical view   — vw_daily_returns
--   6. Analytical view   — vw_company_summary
--   7. Analytical view   — vw_sector_summary
--   8. Sample queries
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Database
-- -----------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS market_analysis
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE market_analysis;


-- -----------------------------------------------------------------------------
-- 2. Dimension tables
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_sector (
    sector_id   INT          NOT NULL,
    sector_name VARCHAR(50)  NOT NULL,
    PRIMARY KEY (sector_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


CREATE TABLE IF NOT EXISTS dim_company (
    company_id   INT          NOT NULL,
    company_name VARCHAR(100) NOT NULL,
    ticker       VARCHAR(10)  NOT NULL,
    sector_id    INT          NOT NULL,
    PRIMARY KEY (company_id),
    UNIQUE KEY uq_ticker (ticker),
    FOREIGN KEY (sector_id) REFERENCES dim_sector (sector_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


-- -----------------------------------------------------------------------------
-- 3. Fact table — fact_stock_prices
--
-- One row per company per trading day.
-- Populated by the Python pipeline after clean_data.py is run.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_stock_prices (
    price_id    INT            NOT NULL AUTO_INCREMENT,
    company_id  INT            NOT NULL,
    date        DATE           NOT NULL,
    open_price  DECIMAL(10, 4) NOT NULL,
    high_price  DECIMAL(10, 4) NOT NULL,
    low_price   DECIMAL(10, 4) NOT NULL,
    close_price DECIMAL(10, 4) NOT NULL,
    volume      BIGINT         NOT NULL,
    PRIMARY KEY (price_id),
    UNIQUE KEY uq_company_date (company_id, date),
    INDEX idx_date (date),
    FOREIGN KEY (company_id) REFERENCES dim_company (company_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4
  COMMENT = 'Daily OHLCV stock prices — one row per company per trading day';


-- -----------------------------------------------------------------------------
-- 4. Reference data
-- -----------------------------------------------------------------------------

INSERT INTO dim_sector (sector_id, sector_name) VALUES
    (1, 'Technology'),
    (2, 'Finance'),
    (3, 'Energy'),
    (4, 'Healthcare');

INSERT INTO dim_company (company_id, company_name, ticker, sector_id) VALUES
    (1,  'Apple',             'AAPL', 1),
    (2,  'Microsoft',         'MSFT', 1),
    (3,  'Nvidia',            'NVDA', 1),
    (4,  'JPMorgan Chase',    'JPM',  2),
    (5,  'Goldman Sachs',     'GS',   2),
    (6,  'Bank of America',   'BAC',  2),
    (7,  'ExxonMobil',        'XOM',  3),
    (8,  'Chevron',           'CVX',  3),
    (9,  'ConocoPhillips',    'COP',  3),
    (10, 'Johnson & Johnson', 'JNJ',  4),
    (11, 'Pfizer',            'PFE',  4),
    (12, 'Merck',             'MRK',  4);


-- -----------------------------------------------------------------------------
-- 5. Analytical view — vw_daily_returns
--
-- Calculates the daily percentage return for each company using the LAG
-- window function on closing price.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW vw_daily_returns AS
WITH price_changes AS (
    SELECT
        company_id,
        date,
        close_price,
        LAG(close_price) OVER (
            PARTITION BY company_id
            ORDER BY date
        ) AS previous_close
    FROM fact_stock_prices
)
SELECT
    company_id,
    date,
    close_price,
    previous_close,
    ROUND(
        (close_price - previous_close) / previous_close * 100,
        4
    ) AS daily_return_pct
FROM price_changes
WHERE previous_close IS NOT NULL;


-- -----------------------------------------------------------------------------
-- 6. Analytical view — vw_company_summary
--
-- Aggregates daily returns to company-level performance metrics.
-- Used by Python analysis scripts for chart generation.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW vw_company_summary AS
SELECT
    c.company_id,
    c.company_name,
    c.ticker,
    s.sector_name,
    ROUND(AVG(r.daily_return_pct),    4) AS avg_daily_return_pct,
    ROUND(STDDEV(r.daily_return_pct), 4) AS volatility_pct,
    ROUND(
        ((MAX(f.close_price) - MIN(f.close_price)) / MIN(f.close_price)) * 100,
        2
    )                                    AS total_return_pct,
    COUNT(DISTINCT r.date)               AS trading_days
FROM vw_daily_returns r
JOIN dim_company c
    ON r.company_id = c.company_id
JOIN dim_sector s
    ON c.sector_id = s.sector_id
JOIN fact_stock_prices f
    ON r.company_id = f.company_id
GROUP BY
    c.company_id,
    c.company_name,
    c.ticker,
    s.sector_name;


-- -----------------------------------------------------------------------------
-- 7. Analytical view — vw_sector_summary
--
-- Rolls company metrics up to sector level for cross-sector comparison.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW vw_sector_summary AS
SELECT
    s.sector_name,
    COUNT(DISTINCT c.company_id)          AS company_count,
    ROUND(AVG(r.daily_return_pct),    4)  AS avg_daily_return_pct,
    ROUND(STDDEV(r.daily_return_pct), 4)  AS volatility_pct
FROM vw_daily_returns r
JOIN dim_company c
    ON r.company_id = c.company_id
JOIN dim_sector s
    ON c.sector_id = s.sector_id
GROUP BY s.sector_name
ORDER BY avg_daily_return_pct DESC;


-- =============================================================================
-- 8. Sample queries
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Company performance ranked by average daily return
-- -----------------------------------------------------------------------------
-- SELECT
--     company_name,
--     ticker,
--     sector_name,
--     avg_daily_return_pct,
--     volatility_pct,
--     total_return_pct,
--     trading_days
-- FROM vw_company_summary
-- ORDER BY avg_daily_return_pct DESC;


-- -----------------------------------------------------------------------------
-- Sector-level risk and return comparison
-- -----------------------------------------------------------------------------
-- SELECT *
-- FROM vw_sector_summary
-- ORDER BY avg_daily_return_pct DESC;


-- -----------------------------------------------------------------------------
-- Top 10 single-day returns across all companies
-- -----------------------------------------------------------------------------
-- SELECT
--     r.date,
--     c.company_name,
--     c.ticker,
--     r.daily_return_pct
-- FROM vw_daily_returns r
-- JOIN dim_company c ON r.company_id = c.company_id
-- ORDER BY r.daily_return_pct DESC
-- LIMIT 10;


-- -----------------------------------------------------------------------------
-- Top 10 single-day losses across all companies
-- -----------------------------------------------------------------------------
-- SELECT
--     r.date,
--     c.company_name,
--     c.ticker,
--     r.daily_return_pct
-- FROM vw_daily_returns r
-- JOIN dim_company c ON r.company_id = c.company_id
-- ORDER BY r.daily_return_pct ASC
-- LIMIT 10;


-- -----------------------------------------------------------------------------
-- 30-day rolling average return per company (recent trend)
-- -----------------------------------------------------------------------------
-- SELECT
--     r.date,
--     c.company_name,
--     ROUND(
--         AVG(r.daily_return_pct) OVER (
--             PARTITION BY r.company_id
--             ORDER BY r.date
--             ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
--         ), 4
--     ) AS rolling_30d_avg_return
-- FROM vw_daily_returns r
-- JOIN dim_company c ON r.company_id = c.company_id
-- ORDER BY c.company_name, r.date;
