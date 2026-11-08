CREATE INDEX idx_raw_housing_batch_hash ON raw.housing_listing(batch_id,source_row_hash);
CREATE INDEX idx_raw_housing_listing_id ON raw.housing_listing(listing_id(100));
CREATE INDEX idx_fact_sale_date_geo ON dw.fact_property_sale(sale_date_key,geography_key);
CREATE INDEX idx_fact_property_type_date ON dw.fact_property_sale(property_type_key,sale_date_key);
CREATE INDEX idx_fact_batch ON dw.fact_property_sale(batch_id);
CREATE INDEX idx_dim_geo_state_suburb ON dw.dim_geography(state_code,suburb_name);
ANALYZE TABLE raw.housing_listing,stg.housing_ranked,dw.fact_property_sale;
EXPLAIN ANALYZE SELECT d.month_start_date,g.state_code,COUNT(*),AVG(f.sale_price_aud) FROM dw.fact_property_sale f JOIN dw.dim_date d ON d.date_key=f.sale_date_key JOIN dw.dim_geography g ON g.geography_key=f.geography_key WHERE d.full_date BETWEEN '2022-01-01' AND '2023-12-31' AND g.state_code IN('NSW','VIC','QLD') GROUP BY 1,2;
