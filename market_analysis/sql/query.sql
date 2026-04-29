CREATE DATABASE market_analysis;
USE market_analysis;

CREATE TABLE dim_sector (
    sector_id INT PRIMARY KEY,
    sector_name VARCHAR(50)
);

CREATE TABLE dim_company (
    company_id INT PRIMARY KEY,
    company_name VARCHAR(100),
    ticker VARCHAR(10),
    sector_id INT,
    FOREIGN KEY (sector_id) REFERENCES dim_sector(sector_id)
);

CREATE TABLE fact_stock_prices (
    price_id INT PRIMARY KEY AUTO_INCREMENT,
    company_id INT,
    date DATE,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    FOREIGN KEY (company_id) REFERENCES dim_company(company_id)
);

INSERT INTO dim_sector VALUES
(1, 'Technology'),
(2, 'Finance'),
(3, 'Energy'),
(4, 'Healthcare');

INSERT INTO dim_company VALUES
(1, 'Apple', 'AAPL', 1),
(2, 'Microsoft', 'MSFT', 1),
(3, 'Nvidia', 'NVDA', 1),
(4, 'JPMorgan Chase', 'JPM', 2),
(5, 'Goldman Sachs', 'GS', 2),
(6, 'Bank of America', 'BAC', 2),
(7, 'ExxonMobil', 'XOM', 3),
(8, 'Chevron', 'CVX', 3),
(9, 'ConocoPhillips', 'COP', 3),
(10, 'Johnson & Johnson', 'JNJ', 4),
(11, 'Pfizer', 'PFE', 4),
(12, 'Merck', 'MRK', 4);

SELECT COUNT(*) AS total_rows
FROM fact_stock_prices;

SELECT *
FROM fact_stock_prices
ORDER BY date
LIMIT 10;

SELECT DISTINCT company_id
FROM fact_stock_prices;

SELECT COUNT(DISTINCT company_id) AS num_companies
FROM fact_stock_prices;

SELECT company_id, COUNT(*) AS total_rows
FROM fact_stock_prices
GROUP BY company_id
ORDER BY company_id;

SELECT company_id, MIN(date) AS start_date, MAX(date) AS end_date
FROM fact_stock_prices
GROUP BY company_id
ORDER BY company_id;

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
SELECT company_id, date, close_price, previous_close,
ROUND(((close_price - previous_close) / previous_close) * 100, 4) AS daily_return_pct
FROM price_changes
WHERE previous_close IS NOT NULL;

SELECT * FROM vw_daily_returns
ORDER BY company_id, date
LIMIT 20;

SELECT
    c.company_name,
    s.sector_name,
    ROUND(AVG(v.daily_return_pct), 4) AS avg_daily_return_pct,
    ROUND(STDDEV(v.daily_return_pct), 4) AS volatility_pct
FROM vw_daily_returns v
JOIN dim_company c
    ON v.company_id = c.company_id
JOIN dim_sector s
    ON c.sector_id = s.sector_id
GROUP BY c.company_name, s.sector_name
ORDER BY s.sector_name, avg_daily_return_pct DESC;

SELECT
    s.sector_name,
    ROUND(AVG(v.daily_return_pct), 4) AS sector_avg_return_pct,
    ROUND(STDDEV(v.daily_return_pct), 4) AS sector_volatility_pct
FROM vw_daily_returns v
JOIN dim_company c
    ON v.company_id = c.company_id
JOIN dim_sector s
    ON c.sector_id = s.sector_id
GROUP BY s.sector_name
ORDER BY sector_avg_return_pct DESC;

SELECT
    c.company_name,
    ROUND(
        ((MAX(f.close_price) - MIN(f.close_price)) / MIN(f.close_price)) * 100,
        2
    ) AS total_return_pct
FROM fact_stock_prices f
JOIN dim_company c
    ON f.company_id = c.company_id
GROUP BY c.company_name
ORDER BY total_return_pct DESC;

CREATE OR REPLACE VIEW vw_company_summary AS
SELECT
    c.company_id,
    c.company_name,
    s.sector_name,
    ROUND(AVG(v.daily_return_pct), 4) AS avg_daily_return_pct,
    ROUND(STDDEV(v.daily_return_pct), 4) AS volatility_pct
FROM vw_daily_returns v
JOIN dim_company c
    ON v.company_id = c.company_id
JOIN dim_sector s
    ON c.sector_id = s.sector_id
GROUP BY c.company_id, c.company_name, s.sector_name;

SELECT * FROM vw_company_summary
ORDER BY sector_name, avg_daily_return_pct DESC;