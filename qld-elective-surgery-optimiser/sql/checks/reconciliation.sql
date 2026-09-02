-- Warehouse reconciliation checks.
--
-- Each query should return zero problematic rows or a logically
-- reconcilable count.

-- 1. Every fact row must have a facility.
SELECT
    COUNT(*) AS missing_facility_keys
FROM fact_elective_surgery_performance
WHERE facility_key IS NULL;


-- 2. Every fact row must have a reporting period.
SELECT
    COUNT(*) AS missing_reporting_periods
FROM fact_elective_surgery_performance
WHERE reporting_period_key IS NULL;


-- 3. Every fact row must retain source lineage.
SELECT
    COUNT(*) AS missing_source_keys
FROM fact_elective_surgery_performance
WHERE source_resource_key IS NULL;


-- 4. Specialty rows require a specialty key.
SELECT
    COUNT(*) AS specialty_rows_without_specialty
FROM fact_elective_surgery_performance
WHERE resource_kind = 'specialty'
  AND specialty_key IS NULL;


-- 5. Category rows require an urgency-category key.
SELECT
    COUNT(*) AS category_rows_without_category
FROM fact_elective_surgery_performance
WHERE resource_kind = 'category'
  AND urgency_category_key IS NULL;


-- 6. Specialty rows must not contain an urgency-category key.
SELECT
    COUNT(*) AS specialty_rows_with_category
FROM fact_elective_surgery_performance
WHERE resource_kind = 'specialty'
  AND urgency_category_key IS NOT NULL;


-- 7. Category rows must not contain a specialty key.
SELECT
    COUNT(*) AS category_rows_with_specialty
FROM fact_elective_surgery_performance
WHERE resource_kind = 'category'
  AND specialty_key IS NOT NULL;


-- 8. Long waits cannot exceed total waiting volume.
SELECT
    COUNT(*) AS invalid_long_wait_relationships
FROM fact_elective_surgery_performance
WHERE vol_long_waits > vol_waiting;


-- 9. Fact primary keys must remain unique.
SELECT
    record_id,
    COUNT(*) AS record_count
FROM fact_elective_surgery_performance
GROUP BY record_id
HAVING COUNT(*) > 1;