CREATE OR REPLACE VIEW mart.v_executive_monthly_kpis AS
SELECT sale_month, state_code,
 sum(sale_count) AS sale_count,
 round(sum(median_sale_price_aud*sale_count)/nullif(sum(sale_count),0),2) AS weighted_segment_median_price_aud,
 round(avg(median_days_on_market),2) AS average_segment_median_days_on_market,
 round(avg(mom_median_price_change_pct),2) AS average_segment_mom_price_change_pct,
 round(avg(average_quality_score),2) AS average_quality_score
FROM mart.monthly_market_summary
GROUP BY sale_month,state_code;

CREATE OR REPLACE VIEW mart.v_property_sale_detail AS
SELECT f.sale_key, d.full_date AS sale_date, d.financial_year,
 g.state_code, g.suburb_name, g.postcode, g.region_name,
 p.property_type_name, src.source_name,
 a.agent_name, a.agency_name,
 f.listing_id, f.sale_price_aud, f.bedrooms, f.bathrooms,
 f.land_size_sqm, f.days_on_market, f.gross_rental_yield_pct,
 f.price_to_suburb_median_ratio, f.quality_score, f.warning_count
FROM dw.fact_property_sale f
JOIN dw.dim_date d ON d.date_key=f.sale_date_key
JOIN dw.dim_geography g ON g.geography_key=f.geography_key
JOIN dw.dim_property_type p ON p.property_type_key=f.property_type_key
JOIN dw.dim_source src ON src.source_key=f.source_key
LEFT JOIN dw.dim_agent a ON a.agent_key=f.agent_key;

CREATE OR REPLACE VIEW mart.v_data_quality_batch_reconciliation AS
SELECT b.batch_id, b.source_file_name, b.row_count AS raw_file_rows,
 count(s.raw_row_id) AS staged_rows,
 SUM(s.record_disposition IN ('accepted','accepted_with_warning') AND s.survivorship_rank=1) AS candidate_fact_rows,
 SUM(s.record_disposition='rejected') AS rejected_rows,
 SUM(s.record_disposition='duplicate_exact') AS exact_duplicate_rows,
 SUM(s.record_disposition='duplicate_variant') AS variant_duplicate_rows
FROM raw.ingestion_batch b
LEFT JOIN stg.housing_ranked s ON s.batch_id=b.batch_id
GROUP BY b.batch_id,b.source_file_name,b.row_count;
