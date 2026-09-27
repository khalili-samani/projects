"""Constraint construction for elective surgery capacity optimisation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from ortools.sat.python import cp_model

from qld_surgery_optimiser.optimisation.models import (
    OptimisationInputRow,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
)


AllocationKey: TypeAlias = tuple[str, str]

AllocationVariables: TypeAlias = Mapping[
    AllocationKey,
    cp_model.IntVar,
]


def allocation_key(
    row: OptimisationInputRow,
) -> AllocationKey:
    """Return the unique optimisation key for one input row.

    The current optimisation grain is:

        facility_key × service_key

    The contract layer is responsible for ensuring that this grain is
    unique before rows reach the solver.
    """
    return (
        row.facility_key,
        row.service_key,
    )


def allocation_upper_bound(
    row: OptimisationInputRow,
    *,
    scenario: OptimisationScenario,
) -> int:
    """Return the maximum feasible allocation for one input row.

    The returned value reflects enabled scenario constraints.

    The total scenario capacity is always used as a hard upper bound
    because no individual service can receive more capacity than is
    available across the entire optimisation problem.
    """
    upper_bound = (
        scenario.capacity.total_additional_cases
    )

    if (
        scenario.constraints
        .enforce_service_capacity_ceiling
    ):
        upper_bound = min(
            upper_bound,
            scenario.capacity
            .default_max_additional_cases_per_service,
        )

    if (
        scenario.constraints
        .enforce_waiting_volume_ceiling
    ):
        upper_bound = min(
            upper_bound,
            row.vol_waiting,
        )

    return max(
        0,
        upper_bound,
    )


def add_capacity_constraints(
    model: cp_model.CpModel,
    *,
    inputs: Sequence[
        OptimisationInputRow
    ],
    allocation_variables: AllocationVariables,
    scenario: OptimisationScenario,
) -> None:
    """Add baseline capacity-allocation constraints to a CP-SAT model.

    Constraints currently include:

    - non-negative allocations;
    - total allocation no greater than scenario capacity;
    - optional per-service capacity ceilings; and
    - optional waiting-volume ceilings.

    Parameters
    ----------
    model:
        OR-Tools CP-SAT model receiving the constraints.
    inputs:
        Validated optimisation input rows.
    allocation_variables:
        Integer decision variables keyed by
        ``(facility_key, service_key)``.
    scenario:
        Validated optimisation scenario containing synthetic capacity
        assumptions and constraint switches.

    Raises
    ------
    ValueError
        If decision variables do not correspond exactly to the supplied
        optimisation inputs.
    """
    _validate_allocation_variables(
        inputs=inputs,
        allocation_variables=(
            allocation_variables
        ),
    )

    variables = [
        allocation_variables[
            allocation_key(
                row
            )
        ]
        for row
        in inputs
    ]

    for row in inputs:
        key = allocation_key(
            row
        )

        variable = (
            allocation_variables[
                key
            ]
        )

        model.add(
            variable
            >= 0
        )

        if (
            scenario.constraints
            .enforce_service_capacity_ceiling
        ):
            model.add(
                variable
                <= (
                    scenario.capacity
                    .default_max_additional_cases_per_service
                )
            )

        if (
            scenario.constraints
            .enforce_waiting_volume_ceiling
        ):
            model.add(
                variable
                <= row.vol_waiting
            )

    if variables:
        model.add(
            sum(
                variables
            )
            <= (
                scenario.capacity
                .total_additional_cases
            )
        )


def _validate_allocation_variables(
    *,
    inputs: Sequence[
        OptimisationInputRow
    ],
    allocation_variables: AllocationVariables,
) -> None:
    """Ensure solver variables exactly match optimisation input rows."""
    expected_keys = {
        allocation_key(
            row
        )
        for row
        in inputs
    }

    actual_keys = set(
        allocation_variables
    )

    missing_keys = (
        expected_keys
        - actual_keys
    )

    unexpected_keys = (
        actual_keys
        - expected_keys
    )

    if (
        not missing_keys
        and not unexpected_keys
    ):
        return

    messages: list[
        str
    ] = []

    if missing_keys:
        formatted_missing = (
            _format_keys(
                missing_keys
            )
        )

        messages.append(
            "missing allocation variables for: "
            f"{formatted_missing}"
        )

    if unexpected_keys:
        formatted_unexpected = (
            _format_keys(
                unexpected_keys
            )
        )

        messages.append(
            "unexpected allocation variables for: "
            f"{formatted_unexpected}"
        )

    raise ValueError(
        "Allocation-variable contract mismatch: "
        + "; ".join(
            messages
        )
        + "."
    )


def _format_keys(
    keys: set[
        AllocationKey
    ],
) -> str:
    """Return deterministic text for allocation keys."""
    return ", ".join(
        (
            f"{facility_key} / "
            f"{service_key}"
        )
        for (
            facility_key,
            service_key,
        )
        in sorted(
            keys
        )
    )