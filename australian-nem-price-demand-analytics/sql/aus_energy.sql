-- =============================================================================
-- schema.sql
-- Australian Energy Market Analysis Pipeline
-- =============================================================================
--
-- Reference document defining the full database schema for this project.
--
-- The Python pipeline (scripts/load_energy.py) handles database setup
-- automatically — you do not need to run this file manually to use the
-- project. It is provided here as a readable reference for the database
-- design, and can be used to inspect or recreate schema objects independently.
--
-- Objects defined:
--   1. Database
--   2. Fact table       — fact_nem_price_demand
--   3. Analytical view  — vw_nem_daily_summary
--   4. Analytical view  — vw_nem_regional_summary
--   5. Sample queries
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Database
-- -----------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS aus_energy_automation
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE aus_energy_automation;


-- -----------------------------------------------------------------------------
-- 2. Fact table — fact_nem_price_demand
--
-- Stores 5-minute settlement interval data for each NEM region.
-- Populated by the Python pipeline (scripts/load_energy.py).
--
-- Note on negative prices:
--   Negative RRP values are a genuine feature of the NEM, not data errors.
--   They occur when non-dispatchable generation (e.g. rooftop solar) exceeds
--   demand and dispatchable generators bid negatively to avoid curtailment.
--   They are retained in all aggregations.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_nem_price_demand (
    record_id            BIGINT         NOT NULL AUTO_INCREMENT,
    region_code          VARCHAR(10)    NOT NULL COMMENT 'NEM region identifier, e.g. NSW1, QLD1, VIC1',
    settlement_datetime  DATETIME       NOT NULL COMMENT '5-minute settlement interval timestamp',
    trading_date         DATE           NOT NULL COMMENT 'Calendar trading date',
    rrp_aud_mwh          DECIMAL(12, 4) NULL     COMMENT 'Regional Reference Price in AUD/MWh (can be negative)',
    total_demand_mw      DECIMAL(12, 4) NULL     COMMENT 'Operational demand in MW',
    period_type          VARCHAR(20)    NULL     COMMENT 'Settlement period type from AEMO',

    PRIMARY KEY (record_id),

    -- Composite index for region + date filtering (most common query pattern)
    INDEX idx_region_date        (region_code, trading_date),

    -- Index for time-series queries across all regions
    INDEX idx_settlement         (settlement_datetime),

    -- Index supporting per-region time-series queries
    INDEX idx_region_settlement  (region_code, settlement_datetime)

) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COMMENT = 'NEM 5-minute wholesale price and demand — one row per region per interval';


-- -----------------------------------------------------------------------------
-- 3. Analytical view — vw_nem_daily_summary
--
-- Aggregates 5-minute intervals to daily metrics per region.
-- Used by analyse_energy.py for chart generation.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW vw_nem_daily_summary AS
SELECT
    trading_date,
    region_code,
    ROUND(AVG(rrp_aud_mwh),    2) AS avg_price_aud_mwh,
    ROUND(MAX(rrp_aud_mwh),    2) AS max_price_aud_mwh,
    ROUND(MIN(rrp_aud_mwh),    2) AS min_price_aud_mwh,
    ROUND(STDDEV(rrp_aud_mwh), 2) AS price_volatility,
    ROUND(AVG(total_demand_mw),2) AS avg_demand_mw,
    COUNT(*)                       AS interval_count,

    -- Proportion of intervals with negative pricing (solar oversupply signal)
    ROUND(
        SUM(CASE WHEN rrp_aud_mwh < 0 THEN 1 ELSE 0 END) / COUNT(*) * 100,
        1
    ) AS negative_price_pct

FROM fact_nem_price_demand
GROUP BY
    trading_date,
    region_code;


-- -----------------------------------------------------------------------------
-- 4. Analytical view — vw_nem_regional_summary
--
-- Monthly summary rolled up by region — useful for cross-period comparisons
-- as more months of data are accumulated.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW vw_nem_regional_summary AS
SELECT
    region_code,
    DATE_FORMAT(trading_date, '%Y-%m')  AS year_month,
    ROUND(AVG(rrp_aud_mwh),    2)       AS avg_price_aud_mwh,
    ROUND(MAX(rrp_aud_mwh),    2)       AS max_price_aud_mwh,
    ROUND(MIN(rrp_aud_mwh),    2)       AS min_price_aud_mwh,
    ROUND(STDDEV(rrp_aud_mwh), 2)       AS price_volatility,
    ROUND(AVG(total_demand_mw),2)       AS avg_demand_mw,
    COUNT(*)                            AS total_intervals,

    -- Negative price frequency — useful BESS strategy signal
    ROUND(
        SUM(CASE WHEN rrp_aud_mwh < 0 THEN 1 ELSE 0 END) / COUNT(*) * 100,
        1
    )                                   AS negative_price_pct

FROM fact_nem_price_demand
GROUP BY
    region_code,
    DATE_FORMAT(trading_date, '%Y-%m');


-- =============================================================================
-- 5. Sample queries
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Daily average price and volatility for a specific month
-- -----------------------------------------------------------------------------
-- SELECT
--     trading_date,
--     region_code,
--     avg_price_aud_mwh,
--     max_price_aud_mwh,
--     min_price_aud_mwh,
--     price_volatility,
--     avg_demand_mw,
--     negative_price_pct
-- FROM vw_nem_daily_summary
-- WHERE trading_date BETWEEN '2026-03-01' AND '2026-03-31'
-- ORDER BY trading_date, region_code;


-- -----------------------------------------------------------------------------
-- Days with the highest average price per region
-- -----------------------------------------------------------------------------
-- SELECT
--     trading_date,
--     region_code,
--     avg_price_aud_mwh,
--     max_price_aud_mwh,
--     avg_demand_mw
-- FROM vw_nem_daily_summary
-- ORDER BY avg_price_aud_mwh DESC
-- LIMIT 20;


-- -----------------------------------------------------------------------------
-- Monthly summary by region — cross-period comparison
-- -----------------------------------------------------------------------------
-- SELECT
--     region_code,
--     year_month,
--     avg_price_aud_mwh,
--     price_volatility,
--     negative_price_pct
-- FROM vw_nem_regional_summary
-- ORDER BY region_code, year_month;


-- -----------------------------------------------------------------------------
-- Price spike intervals above $300/MWh — BESS strategy signal
-- -----------------------------------------------------------------------------
-- SELECT
--     settlement_datetime,
--     region_code,
--     rrp_aud_mwh,
--     total_demand_mw
-- FROM fact_nem_price_demand
-- WHERE rrp_aud_mwh > 300
-- ORDER BY rrp_aud_mwh DESC;


-- -----------------------------------------------------------------------------
-- Negative price intervals by region — solar oversupply signal
-- -----------------------------------------------------------------------------
-- SELECT
--     settlement_datetime,
--     region_code,
--     rrp_aud_mwh,
--     total_demand_mw
-- FROM fact_nem_price_demand
-- WHERE rrp_aud_mwh < 0
-- ORDER BY rrp_aud_mwh ASC;


-- -----------------------------------------------------------------------------
-- Hourly average price profile — time-of-day pattern by region
-- -----------------------------------------------------------------------------
-- SELECT
--     region_code,
--     HOUR(settlement_datetime)      AS hour_of_day,
--     ROUND(AVG(rrp_aud_mwh),    2) AS avg_price_aud_mwh,
--     ROUND(AVG(total_demand_mw),2) AS avg_demand_mw
-- FROM fact_nem_price_demand
-- GROUP BY region_code, HOUR(settlement_datetime)
-- ORDER BY region_code, hour_of_day;


-- -----------------------------------------------------------------------------
-- Peak vs off-peak average price comparison by region
-- -----------------------------------------------------------------------------
-- SELECT
--     region_code,
--     CASE
--         WHEN HOUR(settlement_datetime) BETWEEN 7 AND 21 THEN 'Peak (7am–10pm)'
--         ELSE 'Off-Peak (10pm–7am)'
--     END                            AS period,
--     ROUND(AVG(rrp_aud_mwh),    2) AS avg_price_aud_mwh,
--     ROUND(STDDEV(rrp_aud_mwh), 2) AS price_volatility
-- FROM fact_nem_price_demand
-- GROUP BY region_code, period
-- ORDER BY region_code, period;
