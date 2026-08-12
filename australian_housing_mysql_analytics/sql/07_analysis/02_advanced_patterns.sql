-- Year-over-year comparison using a self-join on month number and prior year.
WITH monthly AS (
 SELECT d.calendar_year,d.month_number,g.state_code,
        percentile_cont(0.5) WITHIN GROUP(ORDER BY f.sale_price_aud) median_price
 FROM dw.fact_property_sale f JOIN dw.dim_date d ON d.date_key=f.sale_date_key
 JOIN dw.dim_geography g ON g.geography_key=f.geography_key
 GROUP BY 1,2,3
)
SELECT cur.calendar_year,cur.month_number,cur.state_code,cur.median_price,
 prev.median_price prior_year_median_price,
 round((cur.median_price-prev.median_price)/nullif(prev.median_price,0)*100,2) yoy_change_pct
FROM monthly cur LEFT JOIN monthly prev
 ON prev.calendar_year=cur.calendar_year-1 AND prev.month_number=cur.month_number AND prev.state_code=cur.state_code
ORDER BY cur.state_code,cur.calendar_year,cur.month_number;

-- Quartile segmentation within state and property type.
WITH ranked AS (
 SELECT f.sale_key,g.state_code,p.property_type_name,f.sale_price_aud,
 ntile(4) OVER(PARTITION BY g.state_code,p.property_type_name ORDER BY f.sale_price_aud) price_quartile
 FROM dw.fact_property_sale f
 JOIN dw.dim_geography g ON g.geography_key=f.geography_key
 JOIN dw.dim_property_type p ON p.property_type_key=f.property_type_key
)
SELECT state_code,property_type_name,price_quartile,count(*) sales,
 min(sale_price_aud) min_price_aud,max(sale_price_aud) max_price_aud
FROM ranked GROUP BY 1,2,3 ORDER BY 1,2,3;

-- Month-level anomaly candidates based on z-score of sales volume.
WITH counts AS (
 SELECT d.month_start_date,g.state_code,count(*) sale_count
 FROM dw.fact_property_sale f JOIN dw.dim_date d ON d.date_key=f.sale_date_key
 JOIN dw.dim_geography g ON g.geography_key=f.geography_key GROUP BY 1,2
), scored AS (
 SELECT *, avg(sale_count) OVER(PARTITION BY state_code) mean_sales,
 stddev_samp(sale_count) OVER(PARTITION BY state_code) sd_sales
 FROM counts
)
SELECT *, round((sale_count-mean_sales)/nullif(sd_sales,0),2) volume_z_score
FROM scored WHERE abs((sale_count-mean_sales)/nullif(sd_sales,0))>=2 ORDER BY abs((sale_count-mean_sales)/nullif(sd_sales,0)) DESC;
