-- 1. Monthly performance with period-over-period and rolling metrics.
SELECT * FROM mart.monthly_market_summary
ORDER BY state_code, property_type_code, sale_month;

-- 2. State contribution to national monthly sales using window functions.
WITH monthly AS (
 SELECT d.month_start_date sale_month, g.state_code, count(*) sale_count,
        sum(f.sale_price_aud) gross_sales_value_aud
 FROM dw.fact_property_sale f
 JOIN dw.dim_date d ON d.date_key=f.sale_date_key
 JOIN dw.dim_geography g ON g.geography_key=f.geography_key
 GROUP BY 1,2
)
SELECT *, round(sale_count/nullif(sum(sale_count) OVER(PARTITION BY sale_month),0)*100,2) AS national_volume_share_pct
FROM monthly ORDER BY sale_month,state_code;

-- 3. Listing-month cohorts and speed-to-sale distribution.
WITH cohorts AS (
 SELECT d.month_start_date cohort_month, p.property_type_name,
        count(*) listings,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY f.days_on_market) median_days_on_market,
        avg((f.days_on_market<=30)) sold_30,
        avg((f.days_on_market<=60)) sold_60,
        avg((f.days_on_market<=90)) sold_90
 FROM dw.fact_property_sale f
 JOIN dw.dim_date d ON d.date_key=f.sale_date_key
 JOIN dw.dim_property_type p ON p.property_type_key=f.property_type_key
 WHERE f.days_on_market IS NOT NULL
 GROUP BY 1,2
)
SELECT cohort_month,property_type_name,listings,median_days_on_market,
 round(sold_30*100,2) sold_within_30_days_pct,
 round(sold_60*100,2) sold_within_60_days_pct,
 round(sold_90*100,2) sold_within_90_days_pct
FROM cohorts ORDER BY cohort_month,property_type_name;

-- 4. Agency data-quality ranking with minimum-volume protection.
WITH agency AS (
 SELECT coalesce(a.agency_name,'Unknown') agency_name,
        count(*) accepted_sales,
        avg(f.quality_score) avg_quality_score,
        avg((f.warning_count>0)) warning_rate
 FROM dw.fact_property_sale f LEFT JOIN dw.dim_agent a ON a.agent_key=f.agent_key
 GROUP BY 1
)
SELECT *, dense_rank() OVER(ORDER BY warning_rate, avg_quality_score DESC) quality_rank
FROM agency WHERE accepted_sales>=10 ORDER BY quality_rank,accepted_sales DESC;

-- 5. Price segmentation and conditional aggregation.
SELECT g.state_code,
 count(*) FILTER(WHERE f.sale_price_aud<500000) under_500k_sales,
 count(*) FILTER(WHERE f.sale_price_aud BETWEEN 500000 AND 999999.99) mid_market_sales,
 count(*) FILTER(WHERE f.sale_price_aud>=1000000) million_plus_sales,
 round(avg(f.gross_rental_yield_pct),2) average_gross_yield_pct
FROM dw.fact_property_sale f JOIN dw.dim_geography g ON g.geography_key=f.geography_key
GROUP BY g.state_code ORDER BY g.state_code;

-- 6. Potential under/over-performance relative to synthetic suburb reference price.
SELECT g.state_code,g.suburb_name,p.property_type_name,count(*) sale_count,
 percentile_cont(0.5) WITHIN GROUP(ORDER BY f.price_to_suburb_median_ratio) median_price_to_reference_ratio,
 avg((f.price_to_suburb_median_ratio>1.10)) materially_above_reference_rate
FROM dw.fact_property_sale f
JOIN dw.dim_geography g ON g.geography_key=f.geography_key
JOIN dw.dim_property_type p ON p.property_type_key=f.property_type_key
WHERE f.price_to_suburb_median_ratio IS NOT NULL
GROUP BY 1,2,3 HAVING count(*)>=5 ORDER BY median_price_to_reference_ratio DESC;

-- 7. Reconciliation summary.
SELECT * FROM mart.v_data_quality_batch_reconciliation ORDER BY batch_id;
