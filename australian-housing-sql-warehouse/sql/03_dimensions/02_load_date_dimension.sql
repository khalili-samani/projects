SET SESSION cte_max_recursion_depth=10000;
WITH RECURSIVE dates AS (SELECT DATE('2020-01-01') d UNION ALL SELECT DATE_ADD(d,INTERVAL 1 DAY) FROM dates WHERE d<'2026-12-31')
INSERT INTO dw.dim_date(date_key,full_date,day_of_month,day_name,week_of_year,month_number,month_name,quarter_number,calendar_year,financial_year,financial_quarter,month_start_date,month_end_date,is_month_end,is_weekend)
SELECT CAST(DATE_FORMAT(d,'%Y%m%d') AS UNSIGNED),d,DAY(d),DAYNAME(d),WEEK(d,3),MONTH(d),MONTHNAME(d),QUARTER(d),YEAR(d),YEAR(d)+IF(MONTH(d)>=7,1,0),FLOOR(MOD(MONTH(d)+5,12)/3)+1,DATE_SUB(d,INTERVAL DAY(d)-1 DAY),LAST_DAY(d),d=LAST_DAY(d),WEEKDAY(d)>=5 FROM dates
ON DUPLICATE KEY UPDATE full_date=VALUES(full_date);
