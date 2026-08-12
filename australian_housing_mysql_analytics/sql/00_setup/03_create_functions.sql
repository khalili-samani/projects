DELIMITER $$
DROP FUNCTION IF EXISTS stg.nullify_marker$$
CREATE FUNCTION stg.nullify_marker(p_value TEXT) RETURNS TEXT DETERMINISTIC
BEGIN
  DECLARE v TEXT;
  IF p_value IS NULL THEN RETURN NULL; END IF;
  SET v=TRIM(p_value);
  IF v='' OR LOWER(v) IN ('n/a','na','-','unknown','?','<blank>','null','none') THEN RETURN NULL; END IF;
  RETURN v;
END$$
DROP FUNCTION IF EXISTS stg.safe_numeric$$
CREATE FUNCTION stg.safe_numeric(p_value TEXT) RETURNS DECIMAL(18,4) DETERMINISTIC
BEGIN
  DECLARE v TEXT;
  SET v=stg.nullify_marker(p_value);
  IF v IS NULL THEN RETURN NULL; END IF;
  SET v=REGEXP_REPLACE(v,'[^0-9.\\-]','');
  IF v IN ('','-','.','-.') OR v NOT REGEXP '^-?[0-9]+(\\.[0-9]+)?$' THEN RETURN NULL; END IF;
  RETURN CAST(v AS DECIMAL(18,4));
END$$
DROP FUNCTION IF EXISTS stg.safe_integer$$
CREATE FUNCTION stg.safe_integer(p_value TEXT) RETURNS BIGINT DETERMINISTIC
BEGIN
  DECLARE n DECIMAL(18,4);
  SET n=stg.safe_numeric(p_value);
  IF n IS NULL OR n<>TRUNCATE(n,0) THEN RETURN NULL; END IF;
  RETURN CAST(n AS SIGNED);
END$$
DROP FUNCTION IF EXISTS stg.parse_boolean$$
CREATE FUNCTION stg.parse_boolean(p_value TEXT) RETURNS TINYINT DETERMINISTIC
BEGIN
  DECLARE v TEXT;
  SET v=LOWER(REGEXP_REPLACE(COALESCE(stg.nullify_marker(p_value),''),'[^a-z0-9]',''));
  RETURN CASE WHEN v IN ('yes','y','true','1') THEN 1 WHEN v IN ('no','n','false','0') THEN 0 ELSE NULL END;
END$$
DROP FUNCTION IF EXISTS stg.normalise_state$$
CREATE FUNCTION stg.normalise_state(p_value TEXT) RETURNS VARCHAR(3) DETERMINISTIC
BEGIN
  DECLARE v TEXT;
  SET v=UPPER(REGEXP_REPLACE(COALESCE(stg.nullify_marker(p_value),''),'[^A-Z]',''));
  RETURN CASE v WHEN 'NSW' THEN 'NSW' WHEN 'NEWSOUTHWALES' THEN 'NSW' WHEN 'VIC' THEN 'VIC' WHEN 'VICTORIA' THEN 'VIC' WHEN 'QLD' THEN 'QLD' WHEN 'QUEENSLAND' THEN 'QLD' WHEN 'SA' THEN 'SA' WHEN 'SOUTHAUSTRALIA' THEN 'SA' WHEN 'WA' THEN 'WA' WHEN 'WESTERNAUSTRALIA' THEN 'WA' WHEN 'TAS' THEN 'TAS' WHEN 'TASMANIA' THEN 'TAS' WHEN 'NT' THEN 'NT' WHEN 'NORTHERNTERRITORY' THEN 'NT' WHEN 'ACT' THEN 'ACT' WHEN 'AUSTRALIANCAPITALTERRITORY' THEN 'ACT' ELSE NULL END;
END$$
DROP FUNCTION IF EXISTS stg.normalise_property_type$$
CREATE FUNCTION stg.normalise_property_type(p_value TEXT) RETURNS VARCHAR(30) DETERMINISTIC
BEGIN
  DECLARE v TEXT; SET v=LOWER(COALESCE(p_value,''));
  RETURN CASE WHEN v REGEXP 'house|detached' THEN 'house' WHEN v REGEXP 'unit|apartment|flat' THEN 'unit_apartment' WHEN v REGEXP 'town' THEN 'townhouse' WHEN v REGEXP 'villa' THEN 'villa' WHEN v REGEXP 'duplex|semi' THEN 'duplex' WHEN v REGEXP 'land|vacant' THEN 'land' ELSE 'unknown' END;
END$$
DROP FUNCTION IF EXISTS stg.parse_sale_date$$
CREATE FUNCTION stg.parse_sale_date(p_value TEXT) RETURNS DATE DETERMINISTIC
BEGIN
  DECLARE v TEXT; DECLARE d DATE;
  SET v=stg.nullify_marker(p_value); IF v IS NULL THEN RETURN NULL; END IF;
  SET d=STR_TO_DATE(v,'%Y-%m-%d'); IF d IS NOT NULL AND DATE_FORMAT(d,'%Y-%m-%d')=v THEN RETURN d; END IF;
  SET d=STR_TO_DATE(v,'%d/%m/%Y'); IF d IS NOT NULL AND DATE_FORMAT(d,'%d/%m/%Y')=v THEN RETURN d; END IF;
  SET d=STR_TO_DATE(v,'%d-%m-%Y'); IF d IS NOT NULL AND DATE_FORMAT(d,'%d-%m-%Y')=v THEN RETURN d; END IF;
  SET d=STR_TO_DATE(v,'%m/%d/%Y'); IF d IS NOT NULL AND DATE_FORMAT(d,'%m/%d/%Y')=v THEN RETURN d; END IF;
  SET d=STR_TO_DATE(v,'%d %b %Y'); IF d IS NOT NULL THEN RETURN d; END IF;
  SET d=STR_TO_DATE(v,'%b %d, %Y'); IF d IS NOT NULL THEN RETURN d; END IF;
  SET d=STR_TO_DATE(CONCAT('01-',v),'%d-%b-%y'); RETURN d;
END$$
DROP FUNCTION IF EXISTS stg.parse_price_aud$$
CREATE FUNCTION stg.parse_price_aud(p_value TEXT) RETURNS DECIMAL(14,2) DETERMINISTIC
BEGIN
  DECLARE v TEXT; DECLARE n DECIMAL(18,4);
  SET v=LOWER(COALESCE(stg.nullify_marker(p_value),''));
  IF v='' OR v REGEXP 'poa|contact' THEN RETURN NULL; END IF;
  SET n=stg.safe_numeric(v); IF n IS NULL THEN RETURN NULL; END IF;
  IF v REGEXP '[0-9](\\.[0-9]+)?[[:space:]]*m' THEN SET n=n*1000000; END IF;
  RETURN ROUND(n,2);
END$$
DROP FUNCTION IF EXISTS stg.parse_land_sqm$$
CREATE FUNCTION stg.parse_land_sqm(p_value TEXT) RETURNS DECIMAL(14,2) DETERMINISTIC
BEGIN
  DECLARE v TEXT; DECLARE n DECIMAL(18,4);
  SET v=LOWER(COALESCE(stg.nullify_marker(p_value),'')); SET n=stg.safe_numeric(v);
  IF n IS NULL THEN RETURN NULL; END IF; IF v REGEXP 'ha|hectare' THEN SET n=n*10000; END IF;
  IF n<0 OR n>10000000 THEN RETURN NULL; END IF; RETURN ROUND(n,2);
END$$
DELIMITER ;
