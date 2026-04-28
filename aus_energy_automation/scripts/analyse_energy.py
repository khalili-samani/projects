import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from config import (
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_DATABASE,
    CHARTS_DIR,
)


if not MYSQL_PASSWORD:
    raise ValueError("MYSQL_PASSWORD environment variable is not set.")


engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
)


def load_daily_summary() -> pd.DataFrame:
    query = """
    SELECT
        trading_date,
        region_code,
        avg_price_aud_mwh,
        max_price_aud_mwh,
        min_price_aud_mwh,
        price_volatility,
        avg_demand_mw
    FROM vw_nem_daily_summary
    ORDER BY trading_date, region_code
    """

    df = pd.read_sql(query, engine)
    df["trading_date"] = pd.to_datetime(df["trading_date"])

    return df


def load_raw_prices() -> pd.DataFrame:
    query = """
    SELECT
        region_code,
        rrp_aud_mwh
    FROM fact_nem_price_demand
    """

    return pd.read_sql(query, engine)


def chart_daily_avg_price(df: pd.DataFrame, year_month: str) -> None:
    plt.figure(figsize=(12, 6))

    for region in df["region_code"].unique():
        subset = df[df["region_code"] == region]
        plt.plot(subset["trading_date"], subset["avg_price_aud_mwh"], label=region)

    plt.xlabel("Date")
    plt.ylabel("Average RRP (AUD/MWh)")
    plt.title(f"Daily Average Wholesale Electricity Price by Region - {year_month}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = CHARTS_DIR / f"daily_avg_rrp_by_region_{year_month}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def chart_price_vs_demand(df: pd.DataFrame, year_month: str) -> None:
    plt.figure(figsize=(8, 6))

    for region in df["region_code"].unique():
        subset = df[df["region_code"] == region]
        plt.scatter(
            subset["avg_demand_mw"],
            subset["avg_price_aud_mwh"],
            label=region,
            alpha=0.8,
        )

    plt.xlabel("Average Demand (MW)")
    plt.ylabel("Average Price (AUD/MWh)")
    plt.title(f"Price vs Demand Relationship - {year_month}")
    plt.legend()
    plt.tight_layout()

    output_path = CHARTS_DIR / f"price_vs_demand_{year_month}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def chart_price_distribution_by_region(df_raw: pd.DataFrame, year_month: str) -> None:
    for region in df_raw["region_code"].unique():
        subset = df_raw[df_raw["region_code"] == region]

        plt.figure(figsize=(8, 6))
        plt.hist(subset["rrp_aud_mwh"], bins=100)

        plt.xlabel("RRP (AUD/MWh)")
        plt.ylabel("Frequency")
        plt.title(f"Electricity Price Distribution - {region} - {year_month}")
        plt.tight_layout()

        output_path = CHARTS_DIR / f"price_distribution_{region.lower()}_{year_month}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")


def chart_volatility_by_region(df: pd.DataFrame, year_month: str) -> None:
    df_vol = (
        df.groupby("region_code", as_index=False)["price_volatility"]
        .mean()
        .sort_values("price_volatility", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(df_vol["region_code"], df_vol["price_volatility"])

    plt.xlabel("Region")
    plt.ylabel("Average Daily Price Volatility")
    plt.title(f"Average Daily Price Volatility by Region - {year_month}")
    plt.tight_layout()

    output_path = CHARTS_DIR / f"avg_daily_price_volatility_by_region_{year_month}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main(year_month: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df_daily = load_daily_summary()
    df_raw = load_raw_prices()

    chart_daily_avg_price(df_daily, year_month)
    chart_price_vs_demand(df_daily, year_month)
    chart_price_distribution_by_region(df_raw, year_month)
    chart_volatility_by_region(df_daily, year_month)

    print("Analysis step completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/analyse_energy.py YYYYMM")

    main(sys.argv[1])
