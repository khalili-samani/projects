CREATE TABLE IF NOT EXISTS dim_facility (
    facility_key VARCHAR PRIMARY KEY,
    facility_code VARCHAR NOT NULL,
    facility_name VARCHAR NOT NULL,
    hhs VARCHAR,
    region VARCHAR,
    resolution_status VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_specialty (
    specialty_key VARCHAR PRIMARY KEY,
    specialty_code VARCHAR,
    specialty_name VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_urgency_category (
    urgency_category_key VARCHAR PRIMARY KEY,
    urgency_category_name VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_reporting_period (
    reporting_period_key DATE PRIMARY KEY,
    calendar_year INTEGER NOT NULL,
    calendar_quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    quarter_label VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_source_resource (
    source_resource_key VARCHAR PRIMARY KEY,
    resource_id VARCHAR,
    source_sha256 VARCHAR,
    source_url VARCHAR,
    source_file VARCHAR,
    retrieved_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS fact_elective_surgery_performance (
    record_id VARCHAR PRIMARY KEY,

    facility_key VARCHAR NOT NULL,
    reporting_period_key DATE NOT NULL,

    resource_kind VARCHAR NOT NULL,

    specialty_key VARCHAR,
    urgency_category_key VARCHAR,

    vol_treated DOUBLE,
    pct_treated_in_time DOUBLE,
    pct_variation_treated_prior_year DOUBLE,

    vol_waiting DOUBLE,
    vol_long_waits DOUBLE,

    pct_waiting_in_time_total DOUBLE,

    vol_long_waits_rfs DOUBLE,
    vol_long_waits_nrfs DOUBLE,
    pct_waiting_in_time_rfs DOUBLE,

    previous_vol_waiting DOUBLE,
    backlog_change DOUBLE,

    previous_vol_long_waits DOUBLE,
    long_wait_change DOUBLE,

    long_wait_share DOUBLE,
    treatment_to_waiting_ratio DOUBLE,

    data_last_update TIMESTAMP,

    source_resource_key VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_data_quality_event (
    event_id VARCHAR PRIMARY KEY,
    source_path VARCHAR,
    resource_kind VARCHAR,
    rule_id VARCHAR,
    severity VARCHAR,
    row_index BIGINT,
    column_name VARCHAR,
    observed_value VARCHAR,
    message VARCHAR
);