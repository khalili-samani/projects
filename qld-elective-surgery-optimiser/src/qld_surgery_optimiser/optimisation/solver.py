"""CP-SAT solver orchestration for elective surgery capacity allocation."""

from __future__ import annotations

from collections.abc import Sequence

from ortools.sat.python import cp_model

from qld_surgery_optimiser.optimisation.constraints import (
    AllocationKey,
    add_capacity_constraints,
    allocation_key,
    allocation_upper_bound,
)
from qld_surgery_optimiser.optimisation.models import (
    AllocationDecision,
    OptimisationInputRow,
    OptimisationSummary,
    SolverStatus,
)
from qld_surgery_optimiser.optimisation.objective import (
    add_waitlist_recovery_objective,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
)


def solve_capacity_allocation(
    *,
    inputs: Sequence[OptimisationInputRow],
    scenario: OptimisationScenario,
) -> OptimisationSummary:
    """Solve one elective surgery capacity-allocation scenario.

    Parameters
    ----------
    inputs:
        Validated optimisation input rows.
    scenario:
        Validated scenario containing synthetic capacity assumptions,
        objective weights, constraint switches and solver settings.

    Returns
    -------
    OptimisationSummary
        Typed optimisation result including solver status, objective
        value and service-level allocation decisions.

    Raises
    ------
    ValueError
        If the optimisation inputs are inconsistent with the scenario.
    """
    _validate_solver_inputs(
        inputs=inputs,
        scenario=scenario,
    )

    model = cp_model.CpModel()

    allocation_variables = _create_allocation_variables(
        model=model,
        inputs=inputs,
        scenario=scenario,
    )

    add_capacity_constraints(
        model,
        inputs=inputs,
        allocation_variables=allocation_variables,
        scenario=scenario,
    )

    add_waitlist_recovery_objective(
        model,
        inputs=inputs,
        allocation_variables=allocation_variables,
        scenario=scenario,
    )

    solver = cp_model.CpSolver()

    _configure_solver(
        solver=solver,
        scenario=scenario,
    )

    raw_status = solver.solve(
        model
    )

    status = _normalise_solver_status(
        int(
            raw_status
        )
    )

    if status not in {
        "optimal",
        "feasible",
    }:
        return OptimisationSummary(
            scenario_name=scenario.scenario.name,
            status=status,
            objective_value=None,
            total_capacity_available=(
                scenario.capacity.total_additional_cases
            ),
            total_additional_cases=0,
            unused_capacity=(
                scenario.capacity.total_additional_cases
            ),
            allocations=(),
        )

    allocations = _extract_allocations(
        solver=solver,
        inputs=inputs,
        allocation_variables=allocation_variables,
    )

    total_additional_cases = sum(
        allocation.additional_cases
        for allocation in allocations
    )

    total_capacity_available = (
        scenario.capacity.total_additional_cases
    )

    unused_capacity = (
        total_capacity_available
        - total_additional_cases
    )

    return OptimisationSummary(
        scenario_name=scenario.scenario.name,
        status=status,
        objective_value=float(
            solver.objective_value
        ),
        total_capacity_available=total_capacity_available,
        total_additional_cases=total_additional_cases,
        unused_capacity=unused_capacity,
        allocations=allocations,
    )


def _create_allocation_variables(
    *,
    model: cp_model.CpModel,
    inputs: Sequence[OptimisationInputRow],
    scenario: OptimisationScenario,
) -> dict[
    AllocationKey,
    cp_model.IntVar,
]:
    """Create bounded integer decision variables for optimisation rows."""
    variables: dict[
        AllocationKey,
        cp_model.IntVar,
    ] = {}

    for row in inputs:
        key = allocation_key(
            row
        )

        if key in variables:
            raise ValueError(
                "Duplicate optimisation key encountered while "
                "creating solver variables: "
                f"{key[0]} / {key[1]}."
            )

        upper_bound = allocation_upper_bound(
            row,
            scenario=scenario,
        )

        variables[
            key
        ] = model.new_int_var(
            0,
            upper_bound,
            _allocation_variable_name(
                key
            ),
        )

    return variables


def _allocation_variable_name(
    key: AllocationKey,
) -> str:
    """Return a deterministic CP-SAT variable name."""
    facility_key, service_key = key

    safe_facility = _normalise_variable_name_part(
        facility_key
    )

    safe_service = _normalise_variable_name_part(
        service_key
    )

    return (
        "additional_cases__"
        f"{safe_facility}__"
        f"{safe_service}"
    )


def _normalise_variable_name_part(
    value: str,
) -> str:
    """Normalise one identifier for use in a solver variable name."""
    cleaned = "".join(
        character
        if character.isalnum()
        else "_"
        for character in value.strip()
    )

    cleaned = cleaned.strip(
        "_"
    )

    return (
        cleaned
        or "unknown"
    )


def _configure_solver(
    *,
    solver: cp_model.CpSolver,
    scenario: OptimisationScenario,
) -> None:
    """Apply deterministic scenario solver settings."""
    solver.parameters.max_time_in_seconds = (
        scenario.solver.max_time_seconds
    )

    solver.parameters.num_search_workers = (
        scenario.solver.num_search_workers
    )

    solver.parameters.random_seed = (
        scenario.solver.random_seed
    )


def _extract_allocations(
    *,
    solver: cp_model.CpSolver,
    inputs: Sequence[OptimisationInputRow],
    allocation_variables: dict[
        AllocationKey,
        cp_model.IntVar,
    ],
) -> tuple[
    AllocationDecision,
    ...,
]:
    """Translate solver values into typed allocation decisions."""
    allocations: list[
        AllocationDecision
    ] = []

    for row in inputs:
        key = allocation_key(
            row
        )

        variable = allocation_variables[
            key
        ]

        additional_cases = int(
            solver.value(
                variable
            )
        )

        allocations.append(
            AllocationDecision(
                facility_key=row.facility_key,
                facility_name=row.facility_name,
                service_key=row.service_key,
                service_name=row.service_name,
                resource_kind=row.resource_kind,
                reporting_period=row.reporting_period,
                additional_cases=additional_cases,
            )
        )

    return tuple(
        allocations
    )


def _normalise_solver_status(
    raw_status: int,
) -> SolverStatus:
    """Map OR-Tools status codes to the public domain model."""
    if raw_status == cp_model.OPTIMAL:
        return "optimal"

    if raw_status == cp_model.FEASIBLE:
        return "feasible"

    if raw_status == cp_model.INFEASIBLE:
        return "infeasible"

    if raw_status == cp_model.MODEL_INVALID:
        return "model_invalid"

    return "unknown"


def _validate_solver_inputs(
    *,
    inputs: Sequence[OptimisationInputRow],
    scenario: OptimisationScenario,
) -> None:
    """Validate cross-row invariants required by the solver."""
    resource_kinds = {
        row.resource_kind
        for row in inputs
    }

    if len(
        resource_kinds
    ) > 1:
        raise ValueError(
            "Optimisation inputs must contain exactly one "
            "resource_kind."
        )

    if (
        resource_kinds
        and scenario.scenario.resource_kind
        not in resource_kinds
    ):
        actual = next(
            iter(
                resource_kinds
            )
        )

        raise ValueError(
            "Scenario resource_kind does not match optimisation "
            f"inputs: scenario={scenario.scenario.resource_kind!r}, "
            f"inputs={actual!r}."
        )

    reporting_periods = {
        row.reporting_period
        for row in inputs
    }

    if len(
        reporting_periods
    ) > 1:
        raise ValueError(
            "Optimisation inputs must contain exactly one "
            "reporting period."
        )

    optimisation_keys = [
        allocation_key(
            row
        )
        for row in inputs
    ]

    if len(
        optimisation_keys
    ) != len(
        set(
            optimisation_keys
        )
    ):
        raise ValueError(
            "Optimisation inputs contain duplicate "
            "facility-service keys."
        )