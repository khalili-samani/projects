CREATE DATABASE IF NOT EXISTS nem_forecasting;
USE nem_forecasting;

CREATE TABLE IF NOT EXISTS raw_nem_data (
    settlement_date DATETIME NOT NULL,
    region_id VARCHAR(10) NOT NULL,
    demand_mw DECIMAL(12, 2) NOT NULL,
    dispatch_price_mwh DECIMAL(12, 2) NOT NULL,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (settlement_date, region_id),
    INDEX idx_raw_region_time (region_id, settlement_date)
);

CREATE TABLE IF NOT EXISTS raw_weather_data (
    observation_date DATETIME NOT NULL,
    station_name VARCHAR(100) NOT NULL,
    temperature_c DECIMAL(5, 2),
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (observation_date, station_name),
    INDEX idx_weather_station_time (station_name, observation_date)
);

CREATE TABLE IF NOT EXISTS weather_station_region_map (
    station_name VARCHAR(100) NOT NULL,
    region_id VARCHAR(10) NOT NULL,

    PRIMARY KEY (station_name, region_id)
);

INSERT INTO weather_station_region_map (station_name, region_id)
VALUES ('Melbourne Olympic Park', 'VIC1')
ON DUPLICATE KEY UPDATE region_id = VALUES(region_id);

CREATE TABLE IF NOT EXISTS cleaned_nem_weather_data (
    settlement_date DATETIME NOT NULL,
    region_id VARCHAR(10) NOT NULL,
    demand_mw DECIMAL(12, 2) NOT NULL,
    dispatch_price_mwh DECIMAL(12, 2) NOT NULL,
    temperature_c DECIMAL(5, 2),
    weather_station_name VARCHAR(100),

    PRIMARY KEY (settlement_date, region_id),
    INDEX idx_clean_region_time (region_id, settlement_date)
);

CREATE TABLE IF NOT EXISTS engineered_features (
    settlement_date DATETIME NOT NULL,
    region_id VARCHAR(10) NOT NULL,

    demand_mw DECIMAL(12, 2) NOT NULL,
    dispatch_price_mwh DECIMAL(12, 2) NOT NULL,
    target_price_mwh DECIMAL(12, 2) NOT NULL,

    temperature_c DECIMAL(5, 2),

    hour INT NOT NULL,
    day_of_week INT NOT NULL,
    month INT NOT NULL,
    is_weekend TINYINT(1) NOT NULL,

    hour_sin DECIMAL(10, 8),
    hour_cos DECIMAL(10, 8),
    day_sin DECIMAL(10, 8),
    day_cos DECIMAL(10, 8),

    demand_lag_1h DECIMAL(12, 2),
    demand_lag_24h DECIMAL(12, 2),

    price_lag_1h DECIMAL(12, 2),
    price_lag_24h DECIMAL(12, 2),

    demand_rolling_mean_4h DECIMAL(12, 2),
    price_rolling_mean_4h DECIMAL(12, 2),

    PRIMARY KEY (settlement_date, region_id),
    INDEX idx_features_region_time (region_id, settlement_date)
);