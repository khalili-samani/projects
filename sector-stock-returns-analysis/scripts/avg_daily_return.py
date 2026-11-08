"""
avg_daily_return.py
-------------------
Generates a bar chart of average daily return (%) by company.

Bars are coloured green for positive returns and dark red for negative returns,
sorted from highest to lowest. Data is sourced from the vw_company_summary
view in MySQL.

Usage:
    python scripts/avg_daily_return.py
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from config import CHART_DPI, CHARTS_DIR, MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_USER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

POSITIVE_COLOUR: str = "#006400"
NEGATIVE_COLOUR: str = "#8B0000"


def _check_password() -> None:
    """Raise immediately if the database password is not configured."""
    if not MYSQL_PASSWORD:
        raise EnvironmentError(
            "MYSQL_PASSWORD environment variable is not set. "
            "Export it before running this script."
        )


def load_company_summary() -> pd.DataFrame:
    """Load the company summary view from MySQL.

    Returns:
        DataFrame with company_name, sector_name, avg_daily_return_pct,
        and volatility_pct columns.
    """
    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}",
        pool_pre_ping=True,
    )
    df = pd.read_sql("SELECT * FROM vw_company_summary", engine)
    logger.info("Loaded %d companies from vw_company_summary.", len(df))
    return df


def chart_avg_daily_return(df: pd.DataFrame) -> None:
    """Plot and save a bar chart of average daily return by company.

    Bars are sorted descending and coloured by return sign.

    Args:
        df: Company summary DataFrame from load_company_summary().
    """
    df = df.sort_values("avg_daily_return_pct", ascending=False)

    colours = [
        POSITIVE_COLOUR if x >= 0 else NEGATIVE_COLOUR
        for x in df["avg_daily_return_pct"]
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(df["company_name"], df["avg_daily_return_pct"], color=colours)

    # Add value labels above/below each bar.
    for bar, val in zip(bars, df["avg_daily_return_pct"]):
        y_pos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.006
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:.3f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Company")
    ax.set_ylabel("Average Daily Return (%)")
    ax.set_title("Average Daily Return by Company")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_DIR / "avg_daily_return_by_company.png"
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()

    logger.info("Saved: %s", output_path)


def main() -> None:
    """Load data and generate the average daily return chart."""
    _check_password()

    df = load_company_summary()
    chart_avg_daily_return(df)

    logger.info("avg_daily_return step completed.")


if __name__ == "__main__":
    main()
