-- Batch-level reconciliation.
SELECT * FROM mart.v_data_quality_batch_reconciliation ORDER BY batch_id;

-- Accepted candidates should equal distinct fact lineage rows for loaded batches.
SELECT
 (SELECT count(*) FROM stg.housing_ranked WHERE record_disposition IN('accepted','accepted_with_warning') AND survivorship_rank=1) AS accepted_candidates,
 (SELECT count(*) FROM dw.fact_property_sale) AS fact_rows;

-- Issue prevalence profile.
SELECT
 count(*) total_rows,
 avg(issue_missing_listing_id) missing_listing_id_rate,
 avg(issue_invalid_sale_date) invalid_sale_date_rate,
 avg(issue_invalid_sale_price) invalid_sale_price_rate,
 avg(issue_invalid_state) invalid_state_rate,
 avg(issue_postcode_state_mismatch) postcode_state_mismatch_rate
FROM stg.housing_ranked;
