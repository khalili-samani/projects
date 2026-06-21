"""
run_pipeline.py
---------------
Orchestrates the end-to-end NEM price and demand pipeline.

Usage:
    python scripts/run_pipeline.py
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE_STEPS: list[str] = [
    "fetch_energy.py",
    "clean_energy.py",
    "load_energy.py",
    "analyse_energy.py",
]


def get_year_month() -> str:
    year = input("Enter year (e.g. 2026): ").strip()
    month = input("Enter month (e.g. 03): ").strip()

    if not year.isdigit() or len(year) != 4:
        raise ValueError(f"Invalid year '{year}' — must be 4 digits, e.g. 2026.")

    if not month.isdigit():
        raise ValueError(f"Invalid month '{month}' — must be numeric, e.g. 03.")

    month_int = int(month)

    if not 1 <= month_int <= 12:
        raise ValueError(f"Invalid month '{month}' — must be between 1 and 12.")

    return f"{year}{month_int:02d}"


def run_step(script_name: str, year_month: str) -> None:
    script_path = SCRIPTS_DIR / script_name

    logger.info("Running: %s %s", script_path, year_month)

    result = subprocess.run(
        [sys.executable, str(script_path), year_month],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed at: {script_path} "
            f"(exit code {result.returncode})"
        )


def main() -> None:
    print("\n── Australian Energy Market Pipeline ──\n")

    try:
        year_month = get_year_month()
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        sys.exit(1)

    logger.info("Pipeline started for period: %s", year_month)

    try:
        for step in PIPELINE_STEPS:
            run_step(step, year_month)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Pipeline completed successfully for period: %s", year_month)
    print(f"\n── All steps completed for {year_month} ──\n")


if __name__ == "__main__":
    main()