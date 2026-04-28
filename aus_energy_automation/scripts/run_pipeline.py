import subprocess
import sys


def get_year_month() -> str:
    year = input("Enter year, e.g. 2026: ").strip()
    month = input("Enter month, e.g. 03: ").strip()

    if not year.isdigit() or len(year) != 4:
        raise ValueError("Year must be 4 digits, e.g. 2026")

    if not month.isdigit():
        raise ValueError("Month must be numeric, e.g. 03")

    month_int = int(month)

    if month_int < 1 or month_int > 12:
        raise ValueError("Month must be between 1 and 12")

    return f"{year}{month_int:02d}"


def run_step(script_path: str, year_month: str) -> None:
    print(f"\nRunning: {script_path}")
    result = subprocess.run(
        [sys.executable, script_path, year_month],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed at: {script_path}")


def main() -> None:
    year_month = get_year_month()

    run_step("scripts/fetch_energy.py", year_month)
    run_step("scripts/clean_energy.py", year_month)
    run_step("scripts/load_energy.py", year_month)
    run_step("scripts/analyse_energy.py", year_month)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
