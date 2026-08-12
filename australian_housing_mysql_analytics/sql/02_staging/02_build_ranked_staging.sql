DROP TABLE IF EXISTS stg.housing_ranked;
CREATE TABLE stg.housing_ranked AS
WITH ranked AS (
 SELECT s.*, ROW_NUMBER() OVER(PARTITION BY batch_id,source_row_hash ORDER BY raw_row_id) exact_duplicate_rank,
 COUNT(*) OVER(PARTITION BY batch_id,source_row_hash) exact_duplicate_count,
 COUNT(*) OVER(PARTITION BY batch_id,listing_id_clean) listing_id_count,
 ROW_NUMBER() OVER(PARTITION BY batch_id,listing_id_clean ORDER BY quality_score DESC, sale_date_parsed DESC, raw_row_id) survivorship_rank
 FROM stg.v_housing_standardised s
)
SELECT r.*, CASE WHEN exact_duplicate_rank>1 THEN 'duplicate_exact' WHEN listing_id_clean IS NOT NULL AND listing_id_count>1 AND survivorship_rank>1 THEN 'duplicate_variant' ELSE 'unique' END duplicate_class,
 CASE WHEN issue_missing_listing_id OR issue_invalid_sale_date OR issue_invalid_sale_price OR issue_invalid_state THEN 'rejected' WHEN exact_duplicate_rank>1 THEN 'duplicate_exact' WHEN listing_id_clean IS NOT NULL AND listing_id_count>1 AND survivorship_rank>1 THEN 'duplicate_variant' WHEN quality_score<100 THEN 'accepted_with_warning' ELSE 'accepted' END record_disposition
FROM ranked r;
ALTER TABLE stg.housing_ranked ADD PRIMARY KEY(raw_row_id), ADD INDEX idx_stg_ranked_batch(batch_id), ADD INDEX idx_stg_ranked_event_hash(event_hash), ADD INDEX idx_stg_ranked_disposition(record_disposition);
