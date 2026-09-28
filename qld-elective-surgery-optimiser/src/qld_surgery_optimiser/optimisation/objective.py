"""Objective construction for elective surgery capacity optimisation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from ortools.sat.python import cp_model

from qld_surgery_optimiser.optimisation.constraints import (
    AllocationKey,
    allocation_key,
)
from qld_surgery_optimiser.optimisation.models import (
    OptimisationInputRow,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
)


ObjectiveCoefficients: TypeAlias = Mapping[
    AllocationKey,
    int,
]


LONG_WAIT_SHARE_SCALE = 1000


def build_objective_coefficients(
    inputs: Sequence[
        OptimisationInputRow
    ],
    *,
    scenario: OptimisationScenario,
) -> dict[
    AllocationKey,
    int,
]:
    """Build integer objective coefficients for each allocation row.

    The baseline objective prioritises two signals:

    1. general waitlist recovery; and
    2. relative long-wait burden.

    Each additional allocated case receives a base score from the
    configured waiting weight. A scaled long-wait-share score is then
    added using the configured long-wait weight.

    CP-SAT requires integer coefficients, so long-wait share is scaled
    to an integer between zero and ``LONG_WAIT_SHARE_SCALE``.

    Parameters
    ----------
    inputs:
        Validated optimisation input rows.
    scenario:
        Validated optimisation scenario containing objective weights.

    Returns
    -------
    dict[AllocationKey, int]
        Integer objective coefficient for each optimisation row.
    """
    coefficients: dict[
        AllocationKey,
        int,
    ] = {}

    for row in inputs:
        key = allocation_key(
            row
        )

        if key in coefficients:
            raise ValueError(
                "Duplicate optimisation key encountered while "
                "building objective coefficients: "
                f"{key[0]} / {key[1]}."
            )

        coefficients[
            key
        ] = objective_coefficient(
            row,
            scenario=scenario,
        )

    return coefficients


def objective_coefficient(
    row: OptimisationInputRow,
    *,
    scenario: OptimisationScenario,
) -> int:
    """Return the integer benefit score for one allocated case.

    The coefficient is calculated as:

    ``waiting component + long-wait component``

    where:

    ``waiting component =
        waiting_weight * LONG_WAIT_SHARE_SCALE``

    and:

    ``long-wait component =
        long_wait_weight * scaled_long_wait_share``

    Scaling the base waiting component to the same magnitude as the
    long-wait share keeps the configured weights interpretable.

    Rows with no waiting population have a long-wait-share value of
    zero.
    """
    waiting_component = (
        scenario.objective.waiting_weight
        * LONG_WAIT_SHARE_SCALE
    )

    long_wait_component = (
        scenario.objective.long_wait_weight
        * scaled_long_wait_share(
            row
        )
    )

    return (
        waiting_component
        + long_wait_component
    )


def scaled_long_wait_share(
    row: OptimisationInputRow,
) -> int:
    """Return a deterministic integer-scaled long-wait share.

    A row with no waiting population returns zero.

    Integer arithmetic is used instead of floating-point arithmetic so
    objective coefficients remain deterministic and CP-SAT compatible.
    """
    if row.vol_waiting == 0:
        return 0

    numerator = (
        row.vol_long_waits
        * LONG_WAIT_SHARE_SCALE
    )

    return (
        numerator
        // row.vol_waiting
    )


def add_waitlist_recovery_objective(
    model: cp_model.CpModel,
    *,
    inputs: Sequence[
        OptimisationInputRow
    ],
    allocation_variables: Mapping[
        AllocationKey,
        cp_model.IntVar,
    ],
    scenario: OptimisationScenario,
) -> dict[
    AllocationKey,
    int,
]:
    """Add the baseline waitlist-recovery objective to a CP-SAT model.

    The model maximises the weighted value of allocated capacity.

    Parameters
    ----------
    model:
        OR-Tools CP-SAT model receiving the objective.
    inputs:
        Validated optimisation input rows.
    allocation_variables:
        Integer decision variables keyed by
        ``(facility_key, service_key)``.
    scenario:
        Validated optimisation scenario.

    Returns
    -------
    dict[AllocationKey, int]
        Objective coefficients used in the model.

    Raises
    ------
    ValueError
        If allocation variables do not correspond exactly to the
        supplied optimisation rows.
    """
    coefficients = (
        build_objective_coefficients(
            inputs,
            scenario=scenario,
        )
    )

    _validate_objective_variables(
        coefficients=coefficients,
        allocation_variables=(
            allocation_variables
        ),
    )

    objective_terms = [
        allocation_variables[
            key
        ]
        * coefficient
        for (
            key,
            coefficient,
        )
        in coefficients.items()
    ]

    if objective_terms:
        model.maximize(
            sum(
                objective_terms
            )
        )
    else:
        model.maximize(
            0
        )

    return coefficients


def _validate_objective_variables(
    *,
    coefficients: ObjectiveCoefficients,
    allocation_variables: Mapping[
        AllocationKey,
        cp_model.IntVar,
    ],
) -> None:
    """Ensure objective terms and allocation variables share one grain."""
    expected_keys = set(
        coefficients
    )

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
        messages.append(
            "missing allocation variables for: "
            + _format_keys(
                missing_keys
            )
        )

    if unexpected_keys:
        messages.append(
            "unexpected allocation variables for: "
            + _format_keys(
                unexpected_keys
            )
        )

    raise ValueError(
        "Objective-variable contract mismatch: "
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
    """Return deterministic text for optimisation keys."""
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