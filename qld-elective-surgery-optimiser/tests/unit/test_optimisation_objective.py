"""Tests for the baseline waitlist-recovery objective."""

from __future__ import annotations

from datetime import date

import pytest
from ortools.sat.python import cp_model

from qld_surgery_optimiser.optimisation.constraints import (
    AllocationKey,
    allocation_key,
)
from qld_surgery_optimiser.optimisation.models import (
    OptimisationInputRow,
)
from qld_surgery_optimiser.optimisation.objective import (
    LONG_WAIT_SHARE_SCALE,
    add_waitlist_recovery_objective,
    build_objective_coefficients,
    objective_coefficient,
    scaled_long_wait_share,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
    parse_scenario,
)


def _scenario(
    *,
    waiting_weight: int = 1,
    long_wait_weight: int = 3,
) -> OptimisationScenario:
    """Return a valid optimisation scenario for objective tests."""
    return parse_scenario(
        {
            "scenario": {
                "name": "objective_test",
                "description": (
                    "Synthetic optimisation scenario used for "
                    "objective unit tests."
                ),
                "resource_kind": "specialty",
                "synthetic_assumptions": True,
                "notes": (
                    "All capacity assumptions are synthetic."
                ),
            },
            "capacity": {
                "total_additional_cases": 100,
                "default_max_additional_cases_per_service": 100,
            },
            "objective": {
                "waiting_weight": waiting_weight,
                "long_wait_weight": long_wait_weight,
            },
            "constraints": {
                "enforce_waiting_volume_ceiling": True,
                "enforce_service_capacity_ceiling": True,
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
    facility_key: str = "facility-1",
    service_key: str = "GS",
    vol_waiting: int = 100,
    vol_long_waits: int = 20,
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
) -> dict[
    AllocationKey,
    cp_model.IntVar,
]:
    """Create integer allocation variables for objective tests."""
    return {
        allocation_key(row): model.new_int_var(
            0,
            100,
            (
                "allocation_"
                f"{row.facility_key}_"
                f"{row.service_key}"
            ),
        )
        for row in inputs
    }


def test_scaled_long_wait_share_returns_zero_for_zero_waiting() -> None:
    """Zero waiting volume should produce a zero long-wait share."""
    row = _input_row(
        vol_waiting=0,
        vol_long_waits=0,
    )

    assert scaled_long_wait_share(
        row
    ) == 0


def test_scaled_long_wait_share_returns_expected_integer_value() -> None:
    """Long-wait share should use deterministic integer scaling."""
    row = _input_row(
        vol_waiting=100,
        vol_long_waits=25,
    )

    assert scaled_long_wait_share(
        row
    ) == 250


def test_scaled_long_wait_share_uses_floor_integer_arithmetic() -> None:
    """Fractional scaled shares should be deterministically floored."""
    row = _input_row(
        vol_waiting=3,
        vol_long_waits=1,
    )

    assert scaled_long_wait_share(
        row
    ) == 333


def test_objective_coefficient_with_no_long_waits() -> None:
    """A row with no long waits should receive only the waiting term."""
    row = _input_row(
        vol_waiting=100,
        vol_long_waits=0,
    )

    scenario = _scenario(
        waiting_weight=1,
        long_wait_weight=3,
    )

    assert objective_coefficient(
        row,
        scenario=scenario,
    ) == LONG_WAIT_SHARE_SCALE


def test_objective_coefficient_combines_waiting_and_long_wait_terms() -> None:
    """The coefficient should combine base and long-wait priority."""
    row = _input_row(
        vol_waiting=100,
        vol_long_waits=50,
    )

    scenario = _scenario(
        waiting_weight=1,
        long_wait_weight=3,
    )

    expected = (
        1 * LONG_WAIT_SHARE_SCALE
        + 3 * 500
    )

    assert objective_coefficient(
        row,
        scenario=scenario,
    ) == expected


def test_objective_coefficient_supports_zero_waiting_weight() -> None:
    """The objective may prioritise long-wait burden exclusively."""
    row = _input_row(
        vol_waiting=100,
        vol_long_waits=40,
    )

    scenario = _scenario(
        waiting_weight=0,
        long_wait_weight=2,
    )

    assert objective_coefficient(
        row,
        scenario=scenario,
    ) == 800


def test_objective_coefficient_supports_zero_long_wait_weight() -> None:
    """The objective may use only general waiting-list recovery."""
    row = _input_row(
        vol_waiting=100,
        vol_long_waits=75,
    )

    scenario = _scenario(
        waiting_weight=2,
        long_wait_weight=0,
    )

    assert objective_coefficient(
        row,
        scenario=scenario,
    ) == (
        2 * LONG_WAIT_SHARE_SCALE
    )


def test_build_objective_coefficients_for_multiple_rows() -> None:
    """Each optimisation row should receive its own coefficient."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
            vol_long_waits=50,
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
            vol_waiting=100,
            vol_long_waits=10,
        ),
    ]

    scenario = _scenario(
        waiting_weight=1,
        long_wait_weight=3,
    )

    coefficients = build_objective_coefficients(
        inputs,
        scenario=scenario,
    )

    assert coefficients == {
        (
            "facility-1",
            "GS",
        ): 2500,
        (
            "facility-2",
            "ORTH",
        ): 1300,
    }


def test_higher_long_wait_share_receives_higher_coefficient() -> None:
    """Higher relative long-wait burden should increase priority."""
    high_burden = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=100,
        vol_long_waits=60,
    )

    low_burden = _input_row(
        facility_key="facility-2",
        service_key="ORTH",
        vol_waiting=100,
        vol_long_waits=10,
    )

    scenario = _scenario()

    high_score = objective_coefficient(
        high_burden,
        scenario=scenario,
    )

    low_score = objective_coefficient(
        low_burden,
        scenario=scenario,
    )

    assert high_score > low_score


def test_equal_long_wait_share_produces_equal_coefficient() -> None:
    """Absolute backlog size alone should not change the coefficient."""
    smaller_backlog = _input_row(
        facility_key="facility-1",
        service_key="GS",
        vol_waiting=100,
        vol_long_waits=20,
    )

    larger_backlog = _input_row(
        facility_key="facility-2",
        service_key="ORTH",
        vol_waiting=500,
        vol_long_waits=100,
    )

    scenario = _scenario()

    assert objective_coefficient(
        smaller_backlog,
        scenario=scenario,
    ) == objective_coefficient(
        larger_backlog,
        scenario=scenario,
    )


def test_rejects_duplicate_optimisation_keys() -> None:
    """Objective coefficients must have one row per optimisation key."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
        ),
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=200,
            vol_long_waits=50,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate optimisation key",
    ):
        build_objective_coefficients(
            inputs,
            scenario=_scenario(),
        )


def test_add_objective_returns_used_coefficients() -> None:
    """Objective construction should expose its scoring coefficients."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
            vol_long_waits=50,
        )
    ]

    model = cp_model.CpModel()

    variables = _allocation_variables(
        model,
        inputs,
    )

    coefficients = add_waitlist_recovery_objective(
        model,
        inputs=inputs,
        allocation_variables=variables,
        scenario=_scenario(),
    )

    assert coefficients == {
        (
            "facility-1",
            "GS",
        ): 2500,
    }


def test_objective_prefers_higher_long_wait_share() -> None:
    """The solved objective should favour the higher-burden service."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
            vol_long_waits=60,
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
            vol_waiting=100,
            vol_long_waits=10,
        ),
    ]

    model = cp_model.CpModel()

    variables = _allocation_variables(
        model,
        inputs,
    )

    model.add(
        sum(
            variables.values()
        )
        <= 10
    )

    add_waitlist_recovery_objective(
        model,
        inputs=inputs,
        allocation_variables=variables,
        scenario=_scenario(),
    )

    solver = cp_model.CpSolver()

    status = solver.solve(
        model
    )

    assert status == cp_model.OPTIMAL

    high_priority = variables[
        (
            "facility-1",
            "GS",
        )
    ]

    low_priority = variables[
        (
            "facility-2",
            "ORTH",
        )
    ]

    assert solver.value(
        high_priority
    ) == 10

    assert solver.value(
        low_priority
    ) == 0


def test_objective_is_indifferent_when_coefficients_are_equal() -> None:
    """Equal burden should produce equal model objective value per case."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
            vol_waiting=100,
            vol_long_waits=20,
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
            vol_waiting=200,
            vol_long_waits=40,
        ),
    ]

    coefficients = build_objective_coefficients(
        inputs,
        scenario=_scenario(),
    )

    assert coefficients[
        (
            "facility-1",
            "GS",
        )
    ] == coefficients[
        (
            "facility-2",
            "ORTH",
        )
    ]


def test_rejects_missing_objective_variable() -> None:
    """Every scored optimisation row requires a decision variable."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
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
        add_waitlist_recovery_objective(
            model,
            inputs=inputs,
            allocation_variables=variables,
            scenario=_scenario(),
        )


def test_rejects_unexpected_objective_variable() -> None:
    """Decision variables must not introduce unscored rows."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
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
        add_waitlist_recovery_objective(
            model,
            inputs=inputs,
            allocation_variables=variables,
            scenario=_scenario(),
        )


def test_rejects_missing_and_unexpected_objective_variables() -> None:
    """Objective contract errors should report both mismatch types."""
    inputs = [
        _input_row(
            facility_key="facility-1",
            service_key="GS",
        ),
        _input_row(
            facility_key="facility-2",
            service_key="ORTH",
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
            "missing allocation variables.*"
            "unexpected allocation variables"
        ),
    ):
        add_waitlist_recovery_objective(
            model,
            inputs=inputs,
            allocation_variables=variables,
            scenario=_scenario(),
        )


def test_empty_input_set_builds_zero_objective() -> None:
    """An empty optimisation set should still produce a valid model."""
    model = cp_model.CpModel()

    coefficients = add_waitlist_recovery_objective(
        model,
        inputs=[],
        allocation_variables={},
        scenario=_scenario(),
    )

    assert coefficients == {}

    solver = cp_model.CpSolver()

    status = solver.solve(
        model
    )

    assert status == cp_model.OPTIMAL