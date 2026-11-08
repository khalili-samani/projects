"""
risk_vs_return.py
-----------------
Generates a scatter chart comparing volatility (risk) vs average daily return
for all companies, coloured by sector.

Each point is labelled with the company name. A zero-return reference line
is included to distinguish positive from negative performers.

Usage:
    python scripts/risk_vs_return.py
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from config import (
    CHART_DPI,
    CHARTS_DIR,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_USER,
    SECTOR_COLOURS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def chart_risk_vs_return(df: pd.DataFrame) -> None:
    """Plot and save a risk vs return scatter chart coloured by sector.

    Company names are annotated next to each point with basic offset to
    reduce overlap.

    Args:
        df: Company summary DataFrame from load_company_summary().
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    for sector, group in df.groupby("sector_name"):
        ax.scatter(
            group["volatility_pct"],
            group["avg_daily_return_pct"],
            s=100,
            label=sector,
            color=SECTOR_COLOURS.get(sector, "#555555"),
            zorder=3,
        )

    # Annotate each point with the company name.
    for _, row in df.iterrows():
        ax.annotate(
            row["company_name"],
            xy=(row["volatility_pct"], row["avg_daily_return_pct"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Volatility — Std Dev of Daily Return (%)")
    ax.set_ylabel("Average Daily Return (%)")
    ax.set_title("Risk vs Return by Company")
    ax.legend(title="Sector", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_DIR / "risk_vs_return.png"
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()

    logger.info("Saved: %s", output_path)


def main() -> None:
    """Load data and generate the risk vs return chart."""
    _check_password()

    df = load_company_summary()
    chart_risk_vs_return(df)

    logger.info("risk_vs_return step completed.")


if __name__ == "__main__":
    main()
