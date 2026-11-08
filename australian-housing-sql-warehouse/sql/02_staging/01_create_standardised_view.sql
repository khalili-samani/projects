CREATE OR REPLACE VIEW stg.v_housing_standardised AS
WITH base AS (
 SELECT r.*, stg.nullify_marker(r.listing_id) listing_id_clean,
 LOWER(COALESCE(stg.nullify_marker(r.source),'unknown')) source_clean,
 REGEXP_REPLACE(stg.nullify_marker(r.address),'[[:space:]]+',' ') address_clean,
 REGEXP_REPLACE(stg.nullify_marker(r.suburb),'[[:space:]]+',' ') suburb_clean,
 stg.normalise_state(r.state) state_code,
 CASE WHEN REGEXP_REPLACE(COALESCE(r.postcode,''),'[^0-9]','') REGEXP '^[0-9]{4}$' THEN REGEXP_REPLACE(r.postcode,'[^0-9]','') END postcode_clean,
 REGEXP_REPLACE(stg.nullify_marker(r.council_area),'[[:space:]]+',' ') council_area_clean,
 REGEXP_REPLACE(stg.nullify_marker(r.region),'[[:space:]]+',' ') region_clean,
 stg.safe_numeric(r.distance_to_cbd_km) distance_to_cbd_km_num, stg.safe_numeric(r.lat) latitude, stg.safe_numeric(r.lon) longitude,
 stg.normalise_property_type(r.property_type) property_type_code,
 CASE LOWER(COALESCE(stg.nullify_marker(r.bedrooms),'')) WHEN 'one' THEN 1 WHEN 'two' THEN 2 WHEN 'three' THEN 3 WHEN 'four' THEN 4 WHEN 'five' THEN 5 WHEN 'six' THEN 6 ELSE stg.safe_integer(r.bedrooms) END bedrooms_num,
 stg.safe_integer(r.bathrooms) bathrooms_num, stg.safe_integer(r.car_spaces) car_spaces_num, stg.safe_integer(r.toilets) toilets_num,
 stg.parse_land_sqm(r.land_size) land_size_sqm, stg.safe_numeric(r.building_area) building_area_sqm, stg.safe_integer(r.year_built) year_built_num_raw,
 stg.parse_boolean(r.has_pool) has_pool_bool, stg.parse_boolean(r.has_garage) has_garage_bool,
 stg.parse_price_aud(r.sale_price) sale_price_display_aud, stg.safe_numeric(r.price_raw_aud) price_raw_aud_num,
 stg.parse_sale_date(r.sale_date) sale_date_parsed, CASE WHEN stg.nullify_marker(r.sale_date) REGEXP '^[A-Za-z]{3}-[0-9]{2}$' THEN 'month' WHEN stg.nullify_marker(r.sale_date) IS NULL THEN 'unknown' ELSE 'day' END sale_date_precision, stg.nullify_marker(r.sale_method) sale_method_clean, stg.safe_integer(r.days_on_market) days_on_market_num,
 stg.nullify_marker(r.inspection_note) inspection_note_clean, stg.nullify_marker(r.agent_name) agent_name_clean, stg.nullify_marker(r.agency_name) agency_name_clean,
 REGEXP_REPLACE(COALESCE(stg.nullify_marker(r.agent_phone),''),'[^0-9+]','') agent_phone_clean,
 stg.safe_numeric(r.rba_cash_rate_pct) rba_cash_rate_pct_num, stg.nullify_marker(r.market_sentiment) market_sentiment_clean,
 stg.nullify_marker(r.market_context) market_context_clean, stg.safe_numeric(r.suburb_median_price) suburb_median_price_aud,
 stg.safe_numeric(r.auction_clearance_rate_pct) auction_clearance_rate_pct_num, stg.safe_numeric(r.weekly_rent_aud) weekly_rent_aud_num,
 stg.safe_integer(r.property_count_suburb) property_count_suburb_num
 FROM raw.housing_listing r
), typed AS (
 SELECT b.*, CASE WHEN price_raw_aud_num BETWEEN 50000 AND 50000000 THEN ROUND(price_raw_aud_num,2) WHEN sale_price_display_aud BETWEEN 50000 AND 50000000 THEN sale_price_display_aud END sale_price_aud,
 CASE WHEN price_raw_aud_num BETWEEN 50000 AND 50000000 THEN 'price_raw_aud' WHEN sale_price_display_aud BETWEEN 50000 AND 50000000 THEN 'sale_price_display' ELSE 'unresolved' END price_source,
 CASE WHEN year_built_num_raw BETWEEN 1800 AND YEAR(COALESCE(sale_date_parsed,CURRENT_DATE)) THEN year_built_num_raw END year_built_num,
 CASE WHEN distance_to_cbd_km_num BETWEEN 0 AND 500 THEN distance_to_cbd_km_num END distance_to_cbd_km_valid,
 IF(latitude BETWEEN -44 AND -10 AND longitude BETWEEN 112 AND 154,1,0) is_coordinate_plausible,
 IF(CHAR_LENGTH(agent_phone_clean) BETWEEN 8 AND 13,1,0) is_phone_plausible FROM base b
), flags AS (
 SELECT t.*, listing_id_clean IS NULL issue_missing_listing_id,
 (sale_date_parsed IS NULL OR sale_date_parsed>DATE(loaded_at_ts)) issue_invalid_sale_date,
 sale_price_aud IS NULL issue_invalid_sale_price, state_code IS NULL issue_invalid_state,
 (postcode_clean IS NULL OR NOT ((state_code='NSW' AND LEFT(postcode_clean,1) IN ('1','2')) OR (state_code='VIC' AND LEFT(postcode_clean,1)='3') OR (state_code='QLD' AND LEFT(postcode_clean,1)='4') OR (state_code='SA' AND LEFT(postcode_clean,1)='5') OR (state_code='WA' AND LEFT(postcode_clean,1)='6') OR (state_code='TAS' AND LEFT(postcode_clean,1)='7') OR (state_code='NT' AND LEFT(postcode_clean,1)='0') OR (state_code='ACT' AND LEFT(postcode_clean,1)='2'))) issue_postcode_state_mismatch,
 (year_built_num_raw IS NOT NULL AND year_built_num IS NULL) issue_invalid_year_built,
 sale_date_precision='month' issue_imputed_sale_day,
 property_type_code='unknown' issue_unknown_property_type,
 ((has_pool IS NOT NULL AND has_pool_bool IS NULL) OR (has_garage IS NOT NULL AND has_garage_bool IS NULL)) issue_invalid_boolean,
 ((bedrooms IS NOT NULL AND bedrooms_num IS NULL) OR (bathrooms IS NOT NULL AND bathrooms_num IS NULL) OR (days_on_market IS NOT NULL AND days_on_market_num IS NULL)) issue_invalid_numeric
 FROM typed t
)
SELECT f.*, GREATEST(0,100-30*issue_missing_listing_id-25*issue_invalid_sale_date-20*issue_invalid_sale_price-15*issue_invalid_state-10*issue_postcode_state_mismatch-10*issue_invalid_year_built-3*issue_imputed_sale_day-5*issue_unknown_property_type-5*issue_invalid_boolean-5*issue_invalid_numeric) quality_score,
 MD5(CONCAT_WS('|',listing_id_clean,CAST(sale_date_parsed AS CHAR),LOWER(COALESCE(address_clean,'')),COALESCE(CAST(sale_price_aud AS CHAR),''))) event_hash,
 MD5(CONCAT_WS('|',LOWER(COALESCE(agent_name_clean,'')),LOWER(COALESCE(agency_name_clean,'')))) agent_natural_key,
 MD5(CONCAT_WS('|',LOWER(COALESCE(suburb_clean,'')),COALESCE(postcode_clean,''),COALESCE(state_code,''),LOWER(COALESCE(council_area_clean,'')))) geography_natural_key
FROM flags f;
