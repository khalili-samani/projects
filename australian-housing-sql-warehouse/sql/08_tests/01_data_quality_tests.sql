CREATE TABLE IF NOT EXISTS audit.test_result(test_run_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,tested_at_ts TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),test_name VARCHAR(255) NOT NULL,severity ENUM('critical','warning') NOT NULL,failure_count BIGINT NOT NULL,passed TINYINT(1) NOT NULL,details TEXT) ENGINE=InnoDB;
DELIMITER $$
DROP PROCEDURE IF EXISTS audit.run_data_quality_tests$$
CREATE PROCEDURE audit.run_data_quality_tests()
BEGIN
 DECLARE failures BIGINT DEFAULT 0; DECLARE critical_total BIGINT DEFAULT 0;
 DELETE FROM audit.test_result WHERE tested_at_ts<DATE_SUB(NOW(),INTERVAL 90 DAY);
 SELECT COUNT(*) INTO failures FROM dw.fact_property_sale WHERE listing_id IS NULL OR sale_price_aud<=0; INSERT INTO audit.test_result(test_name,severity,failure_count,passed,details) VALUES('fact_required_values','critical',failures,failures=0,'Listing ID and positive price are required'); SET critical_total=critical_total+failures;
 SELECT COUNT(*)-COUNT(DISTINCT event_hash) INTO failures FROM dw.fact_property_sale; INSERT INTO audit.test_result(test_name,severity,failure_count,passed,details) VALUES('fact_event_hash_unique','critical',failures,failures=0,'Event hash must be unique'); SET critical_total=critical_total+failures;
 SELECT COUNT(*) INTO failures FROM dw.fact_property_sale f LEFT JOIN dw.dim_date d ON d.date_key=f.sale_date_key LEFT JOIN dw.dim_geography g ON g.geography_key=f.geography_key LEFT JOIN dw.dim_property_type p ON p.property_type_key=f.property_type_key LEFT JOIN dw.dim_source s ON s.source_key=f.source_key WHERE d.date_key IS NULL OR g.geography_key IS NULL OR p.property_type_key IS NULL OR s.source_key IS NULL; INSERT INTO audit.test_result(test_name,severity,failure_count,passed,details) VALUES('fact_foreign_keys_resolve','critical',failures,failures=0,'All mandatory foreign keys must resolve'); SET critical_total=critical_total+failures;
 SELECT ABS((SELECT COUNT(*) FROM raw.housing_listing)-(SELECT COUNT(*) FROM stg.housing_ranked)) INTO failures; INSERT INTO audit.test_result(test_name,severity,failure_count,passed,details) VALUES('raw_to_staging_reconciliation','critical',failures,failures=0,'Every raw row must reach staging'); SET critical_total=critical_total+failures;
 SELECT COUNT(*) INTO failures FROM dw.fact_property_sale WHERE quality_score<40; INSERT INTO audit.test_result(test_name,severity,failure_count,passed,details) VALUES('low_quality_fact_rows','warning',failures,1,'Monitored only');
 IF critical_total>0 THEN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='Critical data quality tests failed'; END IF;
END$$
DELIMITER ;
CALL audit.run_data_quality_tests();
SELECT * FROM audit.test_result ORDER BY test_run_id DESC;
