"""Tests for optimisation scenario loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
    load_scenario,
    parse_scenario,
)


def _valid_scenario_data() -> dict[str, object]:
    """Return one valid baseline scenario mapping."""
    return {
        "scenario": {
            "name": "baseline_capacity_recovery",
            "description": (
                "Synthetic baseline scenario for testing "
                "elective surgery capacity allocation."
            ),
            "resource_kind": "specialty",
            "synthetic_assumptions": True,
            "notes": (
                "Capacity assumptions are synthetic and are "
                "not observed Queensland Health values."
            ),
        },
        "capacity": {
            "total_additional_cases": 100,
            "default_max_additional_cases_per_service": 25,
        },
        "objective": {
            "waiting_weight": 1,
            "long_wait_weight": 3,
        },
        "constraints": {
            "enforce_waiting_volume_ceiling": True,
            "enforce_service_capacity_ceiling": True,
        },
        "solver": {
            "max_time_seconds": 30,
            "num_search_workers": 1,
            "random_seed": 0,
        },
    }


def test_parse_valid_scenario() -> None:
    """A valid mapping should become a typed scenario."""
    scenario = parse_scenario(
        _valid_scenario_data()
    )

    assert isinstance(
        scenario,
        OptimisationScenario,
    )

    assert (
        scenario.scenario.name
        == "baseline_capacity_recovery"
    )

    assert (
        scenario.scenario.resource_kind
        == "specialty"
    )

    assert (
        scenario.scenario.synthetic_assumptions
        is True
    )

    assert (
        scenario.capacity.total_additional_cases
        == 100
    )

    assert (
        scenario.capacity.default_max_additional_cases_per_service
        == 25
    )

    assert (
        scenario.objective.waiting_weight
        == 1
    )

    assert (
        scenario.objective.long_wait_weight
        == 3
    )

    assert (
        scenario.constraints.enforce_waiting_volume_ceiling
        is True
    )

    assert (
        scenario.constraints.enforce_service_capacity_ceiling
        is True
    )

    assert (
        scenario.solver.max_time_seconds
        == 30.0
    )

    assert (
        scenario.solver.num_search_workers
        == 1
    )

    assert (
        scenario.solver.random_seed
        == 0
    )


def test_load_scenario_from_yaml(
    tmp_path: Path,
) -> None:
    """A valid YAML file should load into the typed scenario model."""
    scenario_path = (
        tmp_path
        / "scenario.yml"
    )

    scenario_path.write_text(
        """
scenario:
  name: baseline_capacity_recovery
  description: >
    Synthetic baseline scenario for testing elective surgery
    capacity allocation.
  resource_kind: specialty
  synthetic_assumptions: true
  notes: >
    Capacity assumptions are synthetic and are not observed
    Queensland Health values.

capacity:
  total_additional_cases: 100
  default_max_additional_cases_per_service: 25

objective:
  waiting_weight: 1
  long_wait_weight: 3

constraints:
  enforce_waiting_volume_ceiling: true
  enforce_service_capacity_ceiling: true

solver:
  max_time_seconds: 30
  num_search_workers: 1
  random_seed: 0
""".strip(),
        encoding="utf-8",
    )

    scenario = load_scenario(
        scenario_path
    )

    assert (
        scenario.scenario.name
        == "baseline_capacity_recovery"
    )

    assert (
        scenario.capacity.total_additional_cases
        == 100
    )


def test_loads_repository_baseline_scenario() -> None:
    """The committed baseline scenario should remain valid."""
    scenario_path = (
        Path(__file__).parents[2]
        / "configs"
        / "scenarios"
        / "baseline.yml"
    )

    scenario = load_scenario(
        scenario_path
    )

    assert (
        scenario.scenario.name
        == "baseline_capacity_recovery"
    )

    assert (
        scenario.scenario.resource_kind
        == "specialty"
    )

    assert (
        scenario.scenario.synthetic_assumptions
        is True
    )

    assert (
        scenario.capacity.total_additional_cases
        == 100
    )


def test_rejects_missing_scenario_file(
    tmp_path: Path,
) -> None:
    """Missing scenario files should fail explicitly."""
    scenario_path = (
        tmp_path
        / "missing.yml"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Scenario file does not exist",
    ):
        load_scenario(
            scenario_path
        )


def test_rejects_directory_as_scenario_path(
    tmp_path: Path,
) -> None:
    """A directory is not a valid scenario file."""
    with pytest.raises(
        ValueError,
        match="Scenario path is not a file",
    ):
        load_scenario(
            tmp_path
        )


def test_rejects_invalid_yaml(
    tmp_path: Path,
) -> None:
    """Malformed YAML should fail with a scenario-specific error."""
    scenario_path = (
        tmp_path
        / "invalid.yml"
    )

    scenario_path.write_text(
        """
scenario:
  name: baseline
  broken: [
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Scenario YAML could not be parsed",
    ):
        load_scenario(
            scenario_path
        )


def test_rejects_non_mapping_yaml_root(
    tmp_path: Path,
) -> None:
    """Scenario YAML must use a mapping at the document root."""
    scenario_path = (
        tmp_path
        / "scenario.yml"
    )

    scenario_path.write_text(
        """
- one
- two
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a mapping at the root",
    ):
        load_scenario(
            scenario_path
        )


def test_rejects_missing_required_section() -> None:
    """Every scenario configuration section is mandatory."""
    data = _valid_scenario_data()

    del data[
        "capacity"
    ]

    with pytest.raises(
        ValueError,
        match="missing section 'capacity'",
    ):
        parse_scenario(
            data
        )


def test_rejects_non_mapping_section() -> None:
    """Scenario sections must themselves be mappings."""
    data = _valid_scenario_data()

    data[
        "capacity"
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="section 'capacity' must be a mapping",
    ):
        parse_scenario(
            data
        )


def test_rejects_missing_scenario_name() -> None:
    """Scenario name is mandatory."""
    data = _valid_scenario_data()

    scenario_section = data[
        "scenario"
    ]

    assert isinstance(
        scenario_section,
        dict,
    )

    del scenario_section[
        "name"
    ]

    with pytest.raises(
        ValueError,
        match="Missing required scenario field 'name'",
    ):
        parse_scenario(
            data
        )


def test_rejects_empty_scenario_name() -> None:
    """Scenario names must contain useful text."""
    data = _valid_scenario_data()

    scenario_section = data[
        "scenario"
    ]

    assert isinstance(
        scenario_section,
        dict,
    )

    scenario_section[
        "name"
    ] = "   "

    with pytest.raises(
        ValueError,
        match="field 'name' must not be empty",
    ):
        parse_scenario(
            data
        )


def test_rejects_unsupported_resource_kind() -> None:
    """Only published source families supported by Phase 5 are valid."""
    data = _valid_scenario_data()

    scenario_section = data[
        "scenario"
    ]

    assert isinstance(
        scenario_section,
        dict,
    )

    scenario_section[
        "resource_kind"
    ] = "combined"

    with pytest.raises(
        ValueError,
        match=(
            "resource_kind must be either "
            "'specialty' or 'category'"
        ),
    ):
        parse_scenario(
            data
        )


def test_accepts_category_resource_kind() -> None:
    """Category-level optimisation remains a supported source family."""
    data = _valid_scenario_data()

    scenario_section = data[
        "scenario"
    ]

    assert isinstance(
        scenario_section,
        dict,
    )

    scenario_section[
        "resource_kind"
    ] = "category"

    scenario = parse_scenario(
        data
    )

    assert (
        scenario.scenario.resource_kind
        == "category"
    )


def test_rejects_false_synthetic_assumption_flag() -> None:
    """Unverified capacity scenarios must identify assumptions as synthetic."""
    data = _valid_scenario_data()

    scenario_section = data[
        "scenario"
    ]

    assert isinstance(
        scenario_section,
        dict,
    )

    scenario_section[
        "synthetic_assumptions"
    ] = False

    with pytest.raises(
        ValueError,
        match="synthetic_assumptions: true",
    ):
        parse_scenario(
            data
        )


def test_rejects_non_boolean_synthetic_flag() -> None:
    """Synthetic-assumption metadata must be explicitly boolean."""
    data = _valid_scenario_data()

    scenario_section = data[
        "scenario"
    ]

    assert isinstance(
        scenario_section,
        dict,
    )

    scenario_section[
        "synthetic_assumptions"
    ] = "true"

    with pytest.raises(
        ValueError,
        match=(
            "field 'synthetic_assumptions' "
            "must be a boolean"
        ),
    ):
        parse_scenario(
            data
        )


def test_rejects_negative_total_capacity() -> None:
    """Total additional capacity cannot be negative."""
    data = _valid_scenario_data()

    capacity_section = data[
        "capacity"
    ]

    assert isinstance(
        capacity_section,
        dict,
    )

    capacity_section[
        "total_additional_cases"
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            "field 'total_additional_cases' "
            "must not be negative"
        ),
    ):
        parse_scenario(
            data
        )


def test_allows_zero_total_capacity() -> None:
    """A zero-capacity scenario is valid for baseline testing."""
    data = _valid_scenario_data()

    capacity_section = data[
        "capacity"
    ]

    assert isinstance(
        capacity_section,
        dict,
    )

    capacity_section[
        "total_additional_cases"
    ] = 0

    scenario = parse_scenario(
        data
    )

    assert (
        scenario.capacity.total_additional_cases
        == 0
    )


def test_rejects_negative_service_capacity() -> None:
    """Per-service synthetic capacity cannot be negative."""
    data = _valid_scenario_data()

    capacity_section = data[
        "capacity"
    ]

    assert isinstance(
        capacity_section,
        dict,
    )

    capacity_section[
        "default_max_additional_cases_per_service"
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            "field "
            "'default_max_additional_cases_per_service' "
            "must not be negative"
        ),
    ):
        parse_scenario(
            data
        )


def test_rejects_boolean_capacity_value() -> None:
    """Boolean values must not be accepted as integer capacities."""
    data = _valid_scenario_data()

    capacity_section = data[
        "capacity"
    ]

    assert isinstance(
        capacity_section,
        dict,
    )

    capacity_section[
        "total_additional_cases"
    ] = True

    with pytest.raises(
        ValueError,
        match=(
            "field 'total_additional_cases' "
            "must be an integer"
        ),
    ):
        parse_scenario(
            data
        )


def test_rejects_negative_waiting_weight() -> None:
    """Objective weights cannot be negative."""
    data = _valid_scenario_data()

    objective_section = data[
        "objective"
    ]

    assert isinstance(
        objective_section,
        dict,
    )

    objective_section[
        "waiting_weight"
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            "field 'waiting_weight' "
            "must not be negative"
        ),
    ):
        parse_scenario(
            data
        )


def test_rejects_negative_long_wait_weight() -> None:
    """Long-wait objective weight cannot be negative."""
    data = _valid_scenario_data()

    objective_section = data[
        "objective"
    ]

    assert isinstance(
        objective_section,
        dict,
    )

    objective_section[
        "long_wait_weight"
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            "field 'long_wait_weight' "
            "must not be negative"
        ),
    ):
        parse_scenario(
            data
        )


def test_rejects_all_zero_objective_weights() -> None:
    """At least one optimisation objective term must matter."""
    data = _valid_scenario_data()

    objective_section = data[
        "objective"
    ]

    assert isinstance(
        objective_section,
        dict,
    )

    objective_section[
        "waiting_weight"
    ] = 0

    objective_section[
        "long_wait_weight"
    ] = 0

    with pytest.raises(
        ValueError,
        match=(
            "At least one objective weight "
            "must be greater than zero"
        ),
    ):
        parse_scenario(
            data
        )


def test_allows_zero_waiting_weight_when_long_wait_weight_positive() -> None:
    """A scenario may optimise purely for long-wait priority."""
    data = _valid_scenario_data()

    objective_section = data[
        "objective"
    ]

    assert isinstance(
        objective_section,
        dict,
    )

    objective_section[
        "waiting_weight"
    ] = 0

    scenario = parse_scenario(
        data
    )

    assert (
        scenario.objective.waiting_weight
        == 0
    )

    assert (
        scenario.objective.long_wait_weight
        == 3
    )


def test_rejects_non_boolean_constraint_flag() -> None:
    """Constraint switches must be explicit booleans."""
    data = _valid_scenario_data()

    constraints_section = data[
        "constraints"
    ]

    assert isinstance(
        constraints_section,
        dict,
    )

    constraints_section[
        "enforce_waiting_volume_ceiling"
    ] = "yes"

    with pytest.raises(
        ValueError,
        match=(
            "field 'enforce_waiting_volume_ceiling' "
            "must be a boolean"
        ),
    ):
        parse_scenario(
            data
        )


def test_rejects_zero_solver_time_limit() -> None:
    """Solver time limit must be greater than zero."""
    data = _valid_scenario_data()

    solver_section = data[
        "solver"
    ]

    assert isinstance(
        solver_section,
        dict,
    )

    solver_section[
        "max_time_seconds"
    ] = 0

    with pytest.raises(
        ValueError,
        match=(
            "field 'max_time_seconds' "
            "must be greater than zero"
        ),
    ):
        parse_scenario(
            data
        )


def test_accepts_fractional_solver_time_limit() -> None:
    """The solver time limit may use fractional seconds."""
    data = _valid_scenario_data()

    solver_section = data[
        "solver"
    ]

    assert isinstance(
        solver_section,
        dict,
    )

    solver_section[
        "max_time_seconds"
    ] = 2.5

    scenario = parse_scenario(
        data
    )

    assert (
        scenario.solver.max_time_seconds
        == 2.5
    )


def test_rejects_zero_search_workers() -> None:
    """At least one CP-SAT search worker is required."""
    data = _valid_scenario_data()

    solver_section = data[
        "solver"
    ]

    assert isinstance(
        solver_section,
        dict,
    )

    solver_section[
        "num_search_workers"
    ] = 0

    with pytest.raises(
        ValueError,
        match=(
            "field 'num_search_workers' "
            "must be greater than zero"
        ),
    ):
        parse_scenario(
            data
        )


def test_rejects_negative_random_seed() -> None:
    """The configured deterministic random seed cannot be negative."""
    data = _valid_scenario_data()

    solver_section = data[
        "solver"
    ]

    assert isinstance(
        solver_section,
        dict,
    )

    solver_section[
        "random_seed"
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            "field 'random_seed' "
            "must not be negative"
        ),
    ):
        parse_scenario(
            data
        )