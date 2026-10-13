"""Tests for Phase 5 optimisation CLI integration."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from typer.testing import CliRunner

from qld_surgery_optimiser.cli import app

runner = CliRunner()


def _write_analytical_data(
    path: Path,
) -> None:
    """Write a small optimisation-ready analytical dataset."""
    dataframe = pd.DataFrame(
        {
            "facility_key": [
                "facility-a",
                "facility-b",
            ],
            "facility_name": [
                "Facility A",
                "Facility B",
            ],
            "resource_kind": [
                "specialty",
                "specialty",
            ],
            "report_month": [
                pd.Timestamp(
                    "2025-06-01"
                ),
                pd.Timestamp(
                    "2025-06-01"
                ),
            ],
            "service_code": [
                "general-surgery",
                "orthopaedics",
            ],
            "service_name": [
                "General Surgery",
                "Orthopaedics",
            ],
            "vol_waiting": [
                20,
                20,
            ],
            "vol_long_waits": [
                15,
                2,
            ],
        }
    )

    dataframe.to_parquet(
        path,
        index=False,
    )


def _write_multi_period_analytical_data(
    path: Path,
) -> None:
    """Write optimisation-ready data covering two reporting periods."""
    dataframe = pd.DataFrame(
        {
            "facility_key": [
                "facility-a",
                "facility-a",
            ],
            "facility_name": [
                "Facility A",
                "Facility A",
            ],
            "resource_kind": [
                "specialty",
                "specialty",
            ],
            "report_month": [
                pd.Timestamp(
                    "2025-05-01"
                ),
                pd.Timestamp(
                    "2025-06-01"
                ),
            ],
            "service_code": [
                "general-surgery",
                "general-surgery",
            ],
            "service_name": [
                "General Surgery",
                "General Surgery",
            ],
            "vol_waiting": [
                5,
                20,
            ],
            "vol_long_waits": [
                1,
                15,
            ],
        }
    )

    dataframe.to_parquet(
        path,
        index=False,
    )


def _write_scenario(
    path: Path,
) -> None:
    """Write a deterministic synthetic optimisation scenario."""
    path.write_text(
        """
scenario:
  name: cli_test
  description: >
    Synthetic command-line optimisation scenario.
  resource_kind: specialty
  synthetic_assumptions: true
  notes: >
    Test-only capacity assumptions.

capacity:
  total_additional_cases: 10
  default_max_additional_cases_per_service: 10

objective:
  waiting_weight: 1
  long_wait_weight: 3

constraints:
  enforce_waiting_volume_ceiling: true
  enforce_service_capacity_ceiling: true

solver:
  max_time_seconds: 5
  num_search_workers: 1
  random_seed: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_cli_help_exposes_optimise_command() -> None:
    """Top-level CLI help should advertise the optimisation command."""
    result = runner.invoke(
        app,
        [
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "optimise" in result.output


def test_optimise_command_runs_pipeline(
    tmp_path: Path,
) -> None:
    """The CLI should solve a valid scenario and persist outputs."""
    analytical_data_path = (
        tmp_path
        / "analytical.parquet"
    )

    scenario_path = (
        tmp_path
        / "scenario.yml"
    )

    output_directory = (
        tmp_path
        / "outputs"
    )

    _write_analytical_data(
        analytical_data_path
    )

    _write_scenario(
        scenario_path
    )

    result = runner.invoke(
        app,
        [
            "optimise",
            "--analytical-data",
            str(
                analytical_data_path
            ),
            "--scenario",
            str(
                scenario_path
            ),
            "--output-dir",
            str(
                output_directory
            ),
        ],
    )

    assert result.exit_code == 0, result.output

    assert (
        "Optimisation completed successfully."
        in result.output
    )

    assert (
        "Scenario: cli_test"
        in result.output
    )

    assert (
        "Solver status: optimal"
        in result.output
    )

    assert (
        "Additional cases allocated: 10"
        in result.output
    )

    assert (
        output_directory
        / "cli_test"
        / "optimisation_allocations.csv"
    ).exists()


def test_optimise_command_accepts_reporting_period(
    tmp_path: Path,
) -> None:
    """An explicit CLI reporting period should select that period."""
    analytical_data_path = (
        tmp_path
        / "analytical.parquet"
    )

    scenario_path = (
        tmp_path
        / "scenario.yml"
    )

    output_directory = (
        tmp_path
        / "outputs"
    )

    _write_multi_period_analytical_data(
        analytical_data_path
    )

    _write_scenario(
        scenario_path
    )

    result = runner.invoke(
        app,
        [
            "optimise",
            "--analytical-data",
            str(
                analytical_data_path
            ),
            "--scenario",
            str(
                scenario_path
            ),
            "--output-dir",
            str(
                output_directory
            ),
            "--reporting-period",
            "2025-05-01",
        ],
    )

    assert result.exit_code == 0, result.output

    allocations = pd.read_csv(
        output_directory
        / "cli_test"
        / "optimisation_allocations.csv"
    )

    assert len(
        allocations
    ) == 1

    assert allocations.loc[
        0,
        "reporting_period",
    ] == "2025-05-01"

    additional_cases = cast(
    int,
    allocations.loc[
        0,
        "additional_cases",
    ],
    )
    assert additional_cases == 5


def test_optimise_command_rejects_invalid_reporting_period(
    tmp_path: Path,
) -> None:
    """An invalid reporting period should fail with a clear message."""
    analytical_data_path = (
        tmp_path
        / "analytical.parquet"
    )

    scenario_path = (
        tmp_path
        / "scenario.yml"
    )

    _write_analytical_data(
        analytical_data_path
    )

    _write_scenario(
        scenario_path
    )

    result = runner.invoke(
        app,
        [
            "optimise",
            "--analytical-data",
            str(
                analytical_data_path
            ),
            "--scenario",
            str(
                scenario_path
            ),
            "--reporting-period",
            "June-2025",
        ],
    )

    assert result.exit_code == 1

    assert (
        "Reporting period must use ISO date format "
        "YYYY-MM-DD."
        in result.output
    )