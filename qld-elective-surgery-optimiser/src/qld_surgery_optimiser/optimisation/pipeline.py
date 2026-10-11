"""End-to-end orchestration for elective surgery optimisation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from qld_surgery_optimiser.optimisation.contract import (
    build_optimisation_inputs,
)
from qld_surgery_optimiser.optimisation.diagnostics import (
    OptimisationDiagnostics,
    build_optimisation_diagnostics,
)
from qld_surgery_optimiser.optimisation.models import (
    OptimisationSummary,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
    load_scenario,
)
from qld_surgery_optimiser.optimisation.solver import (
    solve_capacity_allocation,
)


@dataclass(
    frozen=True,
    slots=True,
)
class OptimisationPipelineResult:
    """Paths and model outputs produced by one optimisation run."""

    scenario: OptimisationScenario
    summary: OptimisationSummary
    diagnostics: OptimisationDiagnostics

    output_directory: Path
    summary_path: Path
    allocations_path: Path
    diagnostics_path: Path


def run_optimisation_pipeline(
    *,
    analytical_data_path: Path,
    scenario_path: Path,
    output_directory: Path,
    reporting_period: date | None = None,
) -> OptimisationPipelineResult:
    """Execute one optimisation scenario from analytical Parquet data.

    The pipeline keeps four information classes separate:

    * observed analytical inputs;
    * synthetic scenario assumptions;
    * solver allocation decisions; and
    * derived diagnostic metrics.

    Parameters
    ----------
    analytical_data_path:
        Path to canonical or longitudinal analytical Parquet data.
    scenario_path:
        Path to the dedicated optimisation scenario YAML file.
    output_directory:
        Directory where scenario-specific optimisation outputs are written.
    reporting_period:
        Optional reporting period. When omitted, the latest valid period
        for the scenario's resource family is selected by the input
        contract.

    Returns
    -------
    OptimisationPipelineResult
        Scenario, solver result, diagnostics and persisted output paths.

    Raises
    ------
    FileNotFoundError
        If the analytical dataset or scenario file does not exist.
    ValueError
        If the analytical dataset, scenario or optimisation contract is
        invalid.
    """
    _validate_input_path(
        analytical_data_path,
        label="Analytical data",
    )

    _validate_input_path(
        scenario_path,
        label="Optimisation scenario",
    )

    scenario = load_scenario(
        scenario_path
    )

    analytical_data = pd.read_parquet(
        analytical_data_path
    )

    inputs = build_optimisation_inputs(
        analytical_data,
        resource_kind=(
            scenario.scenario.resource_kind
        ),
        reporting_period=reporting_period,
    )

    summary = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    if summary.status not in {
        "optimal",
        "feasible",
    }:
        raise ValueError(
            "Optimisation pipeline did not produce an allocatable "
            f"solution. Solver status: {summary.status!r}."
        )

    diagnostics = build_optimisation_diagnostics(
        inputs=inputs,
        summary=summary,
        scenario=scenario,
    )

    scenario_output_directory = (
        output_directory
        / _safe_directory_name(
            scenario.scenario.name
        )
    )

    scenario_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        scenario_output_directory
        / "optimisation_summary.json"
    )

    allocations_path = (
        scenario_output_directory
        / "optimisation_allocations.csv"
    )

    diagnostics_path = (
        scenario_output_directory
        / "optimisation_diagnostics.json"
    )

    _write_json(
        summary_path,
        asdict(
            summary
        ),
    )

    _write_allocations_csv(
        allocations_path,
        summary=summary,
    )

    _write_json(
        diagnostics_path,
        asdict(
            diagnostics
        ),
    )

    return OptimisationPipelineResult(
        scenario=scenario,
        summary=summary,
        diagnostics=diagnostics,
        output_directory=(
            scenario_output_directory
        ),
        summary_path=summary_path,
        allocations_path=allocations_path,
        diagnostics_path=diagnostics_path,
    )


def _write_allocations_csv(
    path: Path,
    *,
    summary: OptimisationSummary,
) -> None:
    """Persist solver allocation decisions as a flat CSV file."""
    rows = [
        {
            "facility_key": allocation.facility_key,
            "facility_name": allocation.facility_name,
            "service_key": allocation.service_key,
            "service_name": allocation.service_name,
            "resource_kind": allocation.resource_kind,
            "reporting_period": (
                allocation.reporting_period.isoformat()
            ),
            "additional_cases": (
                allocation.additional_cases
            ),
        }
        for allocation in summary.allocations
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "facility_key",
            "facility_name",
            "service_key",
            "service_name",
            "resource_kind",
            "reporting_period",
            "additional_cases",
        ],
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a deterministic human-readable JSON artefact."""
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_default(
    value: object,
) -> str:
    """Serialise supported non-JSON-native values."""
    if isinstance(
        value,
        date,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    raise TypeError(
        "Unsupported value for optimisation JSON output: "
        f"{type(value).__name__}."
    )


def _safe_directory_name(
    value: str,
) -> str:
    """Return a deterministic filesystem-safe scenario directory name."""
    cleaned = "".join(
        character.lower()
        if character.isalnum()
        else "_"
        for character in value.strip()
    )

    collapsed = "_".join(
        part
        for part in cleaned.split(
            "_"
        )
        if part
    )

    return (
        collapsed
        or "optimisation_scenario"
    )


def _validate_input_path(
    path: Path,
    *,
    label: str,
) -> None:
    """Validate an input file before pipeline execution."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{label} path is not a file: {path}"
        )