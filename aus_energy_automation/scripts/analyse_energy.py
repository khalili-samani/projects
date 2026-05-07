"""
analyse_energy.py
-----------------
Generates analytical charts from the NEM price and demand database.

Charts produced for a given month:
    - Daily average wholesale price by region (line chart)
    - Price vs demand relationship by region (scatter chart)
    - Electricity price distribution by region (histogram)
    - Average daily price volatility by region (bar chart)

Note on negative prices: negative RRP values are a known feature of the NEM,
caused by oversupply from non-dispatchable generation such as rooftop solar.
They are displayed as-is and annotated in the relevant charts.

Usage:
    python scripts/analyse_energy.py YYYYMM
"""

import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import (
    CHART_DPI,
    CHARTS_DIR,
    HISTOGRAM_BINS,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_USER,
    TABLE_NAME,
    VIEW_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Consistent colour palette across all charts.
REGION_COLOURS: dict[str, str] = {
    "NSW1": "#1f77b4",
    "QLD1": "#ff7f0e",
    "VIC1": "#2ca02c",
}


def _check_password() -> None:
    """Raise immediately if the database password is not configured."""
    if not MYSQL_PASSWORD:
        raise EnvironmentError(
            "MYSQL_PASSWORD environment variable is not set. "
            "Export it before running the pipeline."
        )


def _engine() -> Engine:
    """Return a SQLAlchemy engine connected to the target database."""
    return create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}",
        pool_pre_ping=True,
    )


def load_daily_summary(year_month: str) -> pd.DataFrame:
    """Load the daily summary view filtered to the target month.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".

    Returns:
        DataFrame with daily aggregated price and demand metrics.
    """
    year = int(year_month[:4])
    month = int(year_month[4:])

    query = f"""
    SELECT
        trading_date,
        region_code,
        avg_price_aud_mwh,
        max_price_aud_mwh,
        min_price_aud_mwh,
        price_volatility,
        avg_demand_mw
    FROM `{VIEW_NAME}`
    WHERE YEAR(trading_date)  = {year}
      AND MONTH(trading_date) = {month}
    ORDER BY trading_date, region_code
    """
    df = pd.read_sql(query, _engine())
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


def load_raw_prices(year_month: str) -> pd.DataFrame:
    """Load raw 5-minute interval prices filtered to the target month.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".

    Returns:
        DataFrame with region_code and rrp_aud_mwh columns.
    """
    year = int(year_month[:4])
    month = int(year_month[4:])

    query = f"""
    SELECT
        region_code,
        rrp_aud_mwh
    FROM `{TABLE_NAME}`
    WHERE YEAR(trading_date)  = {year}
      AND MONTH(trading_date) = {month}
    """
    return pd.read_sql(query, _engine())


def _save_chart(output_path: object) -> None:
    """Save and close the current matplotlib figure."""
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path)


def chart_daily_avg_price(df: pd.DataFrame, year_month: str) -> None:
    """Plot daily average wholesale price by region.

    Args:
        df: Daily summary DataFrame from load_daily_summary().
        year_month: Period label used in the chart title and filename.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for region in sorted(df["region_code"].unique()):
        subset = df[df["region_code"] == region]
        ax.plot(
            subset["trading_date"],
            subset["avg_price_aud_mwh"],
            label=region,
            color=REGION_COLOURS.get(region),
            linewidth=1.5,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average RRP (AUD/MWh)")
    ax.set_title(
        f"Daily Average Wholesale Electricity Price by Region — {year_month}"
    )
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    _save_chart(CHARTS_DIR / f"daily_avg_rrp_by_region_{year_month}.png")


def chart_price_vs_demand(df: pd.DataFrame, year_month: str) -> None:
    """Plot daily average price vs average demand as a scatter chart.

    Args:
        df: Daily summary DataFrame from load_daily_summary().
        year_month: Period label used in the chart title and filename.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for region in sorted(df["region_code"].unique()):
        subset = df[df["region_code"] == region]
        ax.scatter(
            subset["avg_demand_mw"],
            subset["avg_price_aud_mwh"],
            label=region,
            color=REGION_COLOURS.get(region),
            alpha=0.8,
            s=50,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Average Demand (MW)")
    ax.set_ylabel("Average Price (AUD/MWh)")
    ax.set_title(f"Price vs Demand Relationship — {year_month}")
    ax.legend()
    fig.tight_layout()

    _save_chart(CHARTS_DIR / f"price_vs_demand_{year_month}.png")


def chart_price_distribution_by_region(
    df_raw: pd.DataFrame, year_month: str
) -> None:
    """Plot price frequency distribution for each region.

    Negative prices are retained and annotated — they reflect genuine NEM
    market conditions (oversupply from non-dispatchable generation).

    Args:
        df_raw: Raw interval price DataFrame from load_raw_prices().
        year_month: Period label used in the chart title and filename.
    """
    for region in sorted(df_raw["region_code"].unique()):
        subset = df_raw[df_raw["region_code"] == region]["rrp_aud_mwh"].dropna()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(
            subset,
            bins=HISTOGRAM_BINS,
            color=REGION_COLOURS.get(region, "#1f77b4"),
            edgecolor="white",
            linewidth=0.4,
        )

        negative_pct = (subset < 0).mean() * 100
        if negative_pct > 0:
            ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)

            # Place the label in the negative region (left of zero) at mid-chart
            # height, with the arrow pointing right to the zero line.
            y_mid = ax.get_ylim()[1] * 0.55
            x_label = subset.min() + (0 - subset.min()) * 0.35

            ax.annotate(
                f"{negative_pct:.1f}% of intervals\npriced below zero",
                xy=(0, y_mid),
                xytext=(x_label, y_mid),
                fontsize=9,
                color="red",
                ha="center",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color="red",
                    lw=0.8,
                    connectionstyle="arc3,rad=0.0",
                ),
            )

        ax.set_xlabel("RRP (AUD/MWh)")
        ax.set_ylabel("Frequency (5-min intervals)")
        ax.set_title(
            f"Electricity Price Distribution — {region} — {year_month}"
        )
        fig.tight_layout()

        _save_chart(
            CHARTS_DIR / f"price_distribution_{region.lower()}_{year_month}.png"
        )


def chart_volatility_by_region(df: pd.DataFrame, year_month: str) -> None:
    """Plot average daily price volatility (std dev) by region.

    Args:
        df: Daily summary DataFrame from load_daily_summary().
        year_month: Period label used in the chart title and filename.
    """
    df_vol = (
        df.groupby("region_code", as_index=False)["price_volatility"]
        .mean()
        .sort_values("price_volatility", ascending=False)
    )

    colours = [REGION_COLOURS.get(r, "#aec7e8") for r in df_vol["region_code"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        df_vol["region_code"],
        df_vol["price_volatility"],
        color=colours,
        edgecolor="white",
        linewidth=0.4,
    )

    # Add value labels above each bar.
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Region")
    ax.set_ylabel("Avg Daily Price Volatility (AUD/MWh std dev)")
    ax.set_title(f"Average Daily Price Volatility by Region — {year_month}")
    ax.set_ylim(0, df_vol["price_volatility"].max() * 1.2)
    fig.tight_layout()

    _save_chart(
        CHARTS_DIR / f"avg_daily_price_volatility_by_region_{year_month}.png"
    )


def main(year_month: str) -> None:
    """Generate all charts for the given period.

    Args:
        year_month: Period in YYYYMM format, e.g. "202603".
    """
    _check_password()

    logger.info("Starting analysis step for period: %s", year_month)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df_daily = load_daily_summary(year_month)
    df_raw = load_raw_prices(year_month)

    if df_daily.empty:
        logger.warning(
            "No daily summary data found for %s. "
            "Ensure the load step has been run for this period.",
            year_month,
        )
        return

    chart_daily_avg_price(df_daily, year_month)
    chart_price_vs_demand(df_daily, year_month)
    chart_price_distribution_by_region(df_raw, year_month)
    chart_volatility_by_region(df_daily, year_month)

    logger.info(
        "Analysis step completed — %d chart(s) saved to %s.",
        4,
        CHARTS_DIR,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python scripts/analyse_energy.py YYYYMM")
        sys.exit(1)

    main(sys.argv[1])
