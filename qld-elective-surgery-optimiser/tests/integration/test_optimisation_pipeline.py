"""Integration tests for the Phase 5 optimisation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qld_surgery_optimiser.optimisation.pipeline import (
    run_optimisation_pipeline,
)


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


def _write_scenario(
    path: Path,
) -> None:
    """Write a deterministic synthetic optimisation scenario."""
    path.write_text(
        """
scenario:
  name: integration_test
  description: >
    Synthetic integration-test optimisation scenario.
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


def test_run_optimisation_pipeline_writes_outputs(
    tmp_path: Path,
) -> None:
    """A valid scenario should solve and persist reconciled outputs."""
    analytical_data_path = (
        tmp_path
        / "longitudinal_performance.parquet"
    )

    scenario_path = (
        tmp_path
        / "optimisation_baseline.yml"
    )

    output_directory = (
        tmp_path
        / "reports"
        / "outputs"
        / "optimisation"
    )

    _write_analytical_data(
        analytical_data_path
    )

    _write_scenario(
        scenario_path
    )

    result = run_optimisation_pipeline(
        analytical_data_path=(
            analytical_data_path
        ),
        scenario_path=scenario_path,
        output_directory=(
            output_directory
        ),
    )

    assert result.summary.status == "optimal"
    assert result.summary.total_capacity_available == 10
    assert result.summary.total_additional_cases == 10
    assert result.summary.unused_capacity == 0

    assert result.output_directory == (
        output_directory
        / "integration_test"
    )

    assert result.summary_path.exists()
    assert result.allocations_path.exists()
    assert result.diagnostics_path.exists()

    allocations = pd.read_csv(
        result.allocations_path
    )

    assert len(
        allocations
    ) == 2

    assert (
        allocations[
            "additional_cases"
        ].sum()
        == 10
    )

    general_surgery = allocations.loc[
        allocations[
            "service_key"
        ]
        == "general-surgery",
        "additional_cases",
    ]

    assert len(
        general_surgery
    ) == 1

    assert int(
        general_surgery.iloc[
            0
        ]
    ) == 10


def test_pipeline_json_outputs_are_reconciled(
    tmp_path: Path,
) -> None:
    """Persisted JSON should reconcile with the returned model result."""
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

    result = run_optimisation_pipeline(
        analytical_data_path=(
            analytical_data_path
        ),
        scenario_path=scenario_path,
        output_directory=(
            output_directory
        ),
    )

    summary_payload = json.loads(
        result.summary_path.read_text(
            encoding="utf-8"
        )
    )

    diagnostics_payload = json.loads(
        result.diagnostics_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary_payload[
        "scenario_name"
    ] == "integration_test"

    assert summary_payload[
        "total_additional_cases"
    ] == 10

    assert diagnostics_payload[
        "total_additional_cases"
    ] == 10

    assert diagnostics_payload[
        "capacity_utilisation"
    ] == 1.0


def test_pipeline_selects_latest_reporting_period(
    tmp_path: Path,
) -> None:
    """Omitting the reporting period should use the latest valid period."""
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
        analytical_data_path,
        index=False,
    )

    _write_scenario(
        scenario_path
    )

    result = run_optimisation_pipeline(
        analytical_data_path=(
            analytical_data_path
        ),
        scenario_path=scenario_path,
        output_directory=(
            output_directory
        ),
    )

    assert len(
        result.summary.allocations
    ) == 1

    allocation = result.summary.allocations[
        0
    ]

    assert allocation.reporting_period.isoformat() == (
        "2025-06-01"
    )

    assert allocation.additional_cases == 10


def test_pipeline_rejects_missing_analytical_file(
    tmp_path: Path,
) -> None:
    """The pipeline should fail clearly when analytical data is absent."""
    scenario_path = (
        tmp_path
        / "scenario.yml"
    )

    _write_scenario(
        scenario_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="Analytical data file does not exist",
    ):
        run_optimisation_pipeline(
            analytical_data_path=(
                tmp_path
                / "missing.parquet"
            ),
            scenario_path=scenario_path,
            output_directory=(
                tmp_path
                / "outputs"
            ),
        )


def test_pipeline_rejects_missing_scenario_file(
    tmp_path: Path,
) -> None:
    """The pipeline should fail clearly when scenario config is absent."""
    analytical_data_path = (
        tmp_path
        / "analytical.parquet"
    )

    _write_analytical_data(
        analytical_data_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="Optimisation scenario file does not exist",
    ):
        run_optimisation_pipeline(
            analytical_data_path=(
                analytical_data_path
            ),
            scenario_path=(
                tmp_path
                / "missing.yml"
            ),
            output_directory=(
                tmp_path
                / "outputs"
            ),
        )