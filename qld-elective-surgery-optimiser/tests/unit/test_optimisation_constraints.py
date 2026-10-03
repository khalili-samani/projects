"""Tests for baseline optimisation capacity constraints."""

from __future__ import annotations

from datetime import date

import pytest
from ortools.sat.python import cp_model

from qld_surgery_optimiser.optimisation.constraints import (
    AllocationKey,
    add_capacity_constraints,
    allocation_key,
    allocation_upper_bound,
)
from qld_surgery_optimiser.optimisation.models import (
    OptimisationInputRow,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
    parse_scenario,
)


def _scenario(
    *,
    total_additional_cases: int = 100,
    default_max_additional_cases_per_service: int = 25,
    enforce_waiting_volume_ceiling: bool = True,
    enforce_service_capacity_ceiling: bool = True,
) -> OptimisationScenario:
    """Return a valid optimisation scenario for constraint tests."""
    return parse_scenario(
        {
            "scenario": {
                "name": "constraint_test",
                "description": (
                    "Synthetic optimisation scenario used for "
                    "constraint unit tests."
                ),
                "resource_kind": "specialty",
                "synthetic_assumptions": True,
                "notes": (
                    "All capacity assumptions are synthetic."
                ),
            },
            "capacity": {
                "total_additional_cases": (
                    total_additional_cases
                ),
                "default_max_additional_cases_per_service": (
                    default_max_additional_cases_per_service
                ),
            },
            "objective": {
                "waiting_weight": 1,
                "long_wait_weight": 3,
            },
            "constraints": {
                "enforce_waiting_volume_ceiling": (
                    enforce_waiting_volume_ceiling
                ),
                "enforce_service_capacity_ceiling": (
                    enforce_service_capacity_ceiling
                ),
            },
            "solver": {
                "max_time_seconds": 5,
                "num_search_workers": 1,
                "random_seed": 0,
            },
        }
    )


def _input_row(
    *,
    facility_key: str,
    service_key: str,
    vol_waiting: int,
    vol_long_waits: int = 0,
) -> OptimisationInputRow:
    """Return one valid optimisation input row."""
    return OptimisationInputRow(
        facility_key=facility_key,
        facility_name=f"Facility {facility_key}",
        service_key=service_key,
        service_name=f"Service {service_key}",
        resource_kind="specialty",
        reporting_period=date(
            2025,
            6,
            1,
        ),
        vol_waiting=vol_waiting,
        vol_long_waits=vol_long_waits,
    )


def _allocation_variables(
    model: cp_model.CpModel,
    inputs: list[OptimisationInputRow],
    *,
    upper_bound: int = 1000,
) -> dict[
    AllocationKey,
    cp_model.IntVar,
]:
    """Create integer allocation variables for test inputs."""
    return {
        allocation_key(row): model.new_int_var(
            0,
            upper_bound,
            (
                "allocation_"
                f"{row.facility_key}_"
                f"{row.service_key}"
            ),
        )
        for row in inputs
    }


def _solve_maximising_total(
    *,
    inputs: list[OptimisationInputRow],
    scenario: OptimisationScenario,
) -> tuple[
    cp_model.CpSolver,
    dict[
        AllocationKey,
        cp_model.IntVar,
    ],
]:
    """Build constraints and maximise total allocation."""
    model = cp_model.CpModel()

    variables = _allocation_variables(
        model,
        inputs,
    )

    add_capacity_constraints(
        model,
        inputs=inputs,
        allocation_variables=variables,
        scenario=scenario,
    )

    model.maximize(
        sum(
            variables.values()
        )
    )

    solver = cp_model.CpSolver()

    status = solver.solve(
        model
    )

    assert status in {
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    }

    return (
        solver,
        variables,
    )


def test_allocation_key_uses_facility_and_service() -> None:
    """Allocation keys should match the optimisation grain."""
    row = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=100,
    )

    assert allocation_key(
        row
    ) == (
        "facility-1",
        "GS",
    )


def test_upper_bound_uses_total_capacity() -> None:
    """No row can exceed the total scenario capacity."""
    row = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=500,
    )

    scenario = _scenario(
        total_additional_cases=40,
        default_max_additional_cases_per_service=100,
    )

    assert allocation_upper_bound(
        row,
        scenario=scenario,
    ) == 40


def test_upper_bound_uses_service_capacity_ceiling() -> None:
    """Per-service capacity should tighten a row's upper bound."""
    row = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=500,
    )

    scenario = _scenario(
        total_additional_cases=100,
        default_max_additional_cases_per_service=25,
    )

    assert allocation_upper_bound(
        row,
        scenario=scenario,
    ) == 25


def test_upper_bound_uses_waiting_volume_ceiling() -> None:
    """Allocation cannot exceed observed waiting volume when enabled."""
    row = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=12,
    )

    scenario = _scenario(
        total_additional_cases=100,
        default_max_additional_cases_per_service=25,
    )

    assert allocation_upper_bound(
        row,
        scenario=scenario,
    ) == 12


def test_upper_bound_ignores_service_ceiling_when_disabled() -> None:
    """Disabled service ceilings should not restrict the row."""
    row = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=80,
    )

    scenario = _scenario(
        total_additional_cases=100,
        default_max_additional_cases_per_service=10,
        enforce_service_capacity_ceiling=False,
    )

    assert allocation_upper_bound(
        row,
        scenario=scenario,
    ) == 80


def test_upper_bound_ignores_waiting_ceiling_when_disabled() -> None:
    """Disabled waiting ceilings should not restrict the row."""
    row = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=5,
    )

    scenario = _scenario(
        total_additional_cases=100,
        default_max_additional_cases_per_service=25,
        enforce_waiting_volume_ceiling=False,
    )

    assert allocation_upper_bound(
        row,
        scenario=scenario,
    ) == 25


def test_total_allocation_cannot_exceed_global_capacity() -> None:
    """Combined allocations must respect the scenario-wide capacity."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
            vol_waiting=100,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=30,
        default_max_additional_cases_per_service=100,
    )

    solver, variables = _solve_maximising_total(
        inputs=inputs,
        scenario=scenario,
    )

    total = sum(
        solver.value(
            variable
        )
        for variable
        in variables.values()
    )

    assert total == 30


def test_allocation_cannot_exceed_service_capacity() -> None:
    """A service-level allocation must respect its synthetic ceiling."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
        )
    ]

    scenario = _scenario(
        total_additional_cases=100,
        default_max_additional_cases_per_service=17,
    )

    solver, variables = _solve_maximising_total(
        inputs=inputs,
        scenario=scenario,
    )

    variable = variables[
        (
            "facility-1",
            "GS",
        )
    ]

    assert solver.value(
        variable
    ) == 17


def test_allocation_cannot_exceed_waiting_volume() -> None:
    """Allocation must not exceed observed backlog when enabled."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=9,
        )
    ]

    scenario = _scenario(
        total_additional_cases=100,
        default_max_additional_cases_per_service=100,
    )

    solver, variables = _solve_maximising_total(
        inputs=inputs,
        scenario=scenario,
    )

    variable = variables[
        (
            "facility-1",
            "GS",
        )
    ]

    assert solver.value(
        variable
    ) == 9


def test_zero_waiting_volume_forces_zero_allocation() -> None:
    """A zero-backlog service should receive no allocation."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=0,
        )
    ]

    scenario = _scenario(
        total_additional_cases=100,
        default_max_additional_cases_per_service=100,
    )

    solver, variables = _solve_maximising_total(
        inputs=inputs,
        scenario=scenario,
    )

    variable = variables[
        (
            "facility-1",
            "GS",
        )
    ]

    assert solver.value(
        variable
    ) == 0


def test_zero_total_capacity_forces_zero_allocation() -> None:
    """A zero-capacity scenario should allocate no cases."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
            vol_waiting=100,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=0,
        default_max_additional_cases_per_service=25,
    )

    solver, variables = _solve_maximising_total(
        inputs=inputs,
        scenario=scenario,
    )

    assert all(
        solver.value(
            variable
        )
        == 0
        for variable
        in variables.values()
    )


def test_service_ceiling_can_be_disabled() -> None:
    """The service ceiling switch should change feasible allocation."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=80,
        )
    ]

    scenario = _scenario(
        total_additional_cases=50,
        default_max_additional_cases_per_service=10,
        enforce_service_capacity_ceiling=False,
    )

    solver, variables = _solve_maximising_total(
        inputs=inputs,
        scenario=scenario,
    )

    variable = variables[
        (
            "facility-1",
            "GS",
        )
    ]

    assert solver.value(
        variable
    ) == 50


def test_waiting_ceiling_can_be_disabled() -> None:
    """The waiting-volume ceiling switch should change feasibility."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=5,
        )
    ]

    scenario = _scenario(
        total_additional_cases=40,
        default_max_additional_cases_per_service=40,
        enforce_waiting_volume_ceiling=False,
    )

    solver, variables = _solve_maximising_total(
        inputs=inputs,
        scenario=scenario,
    )

    variable = variables[
        (
            "facility-1",
            "GS",
        )
    ]

    assert solver.value(
        variable
    ) == 40


def test_rejects_missing_allocation_variable() -> None:
    """Every optimisation row must have a corresponding decision variable."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
            vol_waiting=100,
        ),
    ]

    model = cp_model.CpModel()

    variables = {
        (
            "facility-1",
            "GS",
        ): model.new_int_var(
            0,
            100,
            "allocation_facility_1_gs",
        )
    }

    with pytest.raises(
        ValueError,
        match="missing allocation variables",
    ):
        add_capacity_constraints(
            model,
            inputs=inputs,
            allocation_variables=variables,
            scenario=_scenario(),
        )


def test_rejects_unexpected_allocation_variable() -> None:
    """Solver variables must not introduce services absent from inputs."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
        )
    ]

    model = cp_model.CpModel()

    variables = _allocation_variables(
        model,
        inputs,
    )

    variables[
        (
            "facility-99",
            "UNKNOWN",
        )
    ] = model.new_int_var(
        0,
        100,
        "unexpected_allocation",
    )

    with pytest.raises(
        ValueError,
        match="unexpected allocation variables",
    ):
        add_capacity_constraints(
            model,
            inputs=inputs,
            allocation_variables=variables,
            scenario=_scenario(),
        )


def test_rejects_missing_and_unexpected_variables_together() -> None:
    """Contract errors should report both kinds of variable mismatch."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
            vol_waiting=100,
        ),
    ]

    model = cp_model.CpModel()

    variables = {
        (
            "facility-1",
            "GS",
        ): model.new_int_var(
            0,
            100,
            "allocation_facility_1_gs",
        ),
        (
            "facility-99",
            "UNKNOWN",
        ): model.new_int_var(
            0,
            100,
            "unexpected_allocation",
        ),
    }

    with pytest.raises(
        ValueError,
        match=(
            r"missing allocation variables.*"
            r"unexpected allocation variables"
        ),
    ):
        add_capacity_constraints(
            model,
            inputs=inputs,
            allocation_variables=variables,
            scenario=_scenario(),
        )


def test_empty_input_set_is_valid() -> None:
    """Constraint construction should tolerate an empty input collection."""
    model = cp_model.CpModel()

    add_capacity_constraints(
        model,
        inputs=[],
        allocation_variables={},
        scenario=_scenario(),
    )

    solver = cp_model.CpSolver()

    status = solver.solve(
        model
    )

    assert status in {
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    }