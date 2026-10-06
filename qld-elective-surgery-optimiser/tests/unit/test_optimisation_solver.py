"""Tests for CP-SAT elective surgery capacity allocation."""

from __future__ import annotations

from datetime import date

import pytest

from qld_surgery_optimiser.optimisation.models import (
    OptimisationInputRow,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    CapacityConfig,
    ConstraintConfig,
    ObjectiveConfig,
    OptimisationScenario,
    ScenarioMetadata,
    SolverConfig,
)
from qld_surgery_optimiser.optimisation.solver import (
    solve_capacity_allocation,
)


REPORTING_PERIOD = date(
    2025,
    6,
    1,
)


def _scenario(
    *,
    resource_kind: str = "specialty",
    total_additional_cases: int = 10,
    max_additional_cases_per_service: int = 10,
    waiting_weight: int = 1,
    long_wait_weight: int = 3,
    enforce_waiting_volume_ceiling: bool = True,
    enforce_service_capacity_ceiling: bool = True,
) -> OptimisationScenario:
    """Build a deterministic optimisation scenario for solver tests."""
    if resource_kind not in {
        "specialty",
        "category",
    }:
        raise ValueError(
            "Unsupported resource_kind in test helper."
        )

    return OptimisationScenario(
        scenario=ScenarioMetadata(
            name="solver_test",
            description=(
                "Synthetic scenario used by solver unit tests."
            ),
            resource_kind=resource_kind,
            synthetic_assumptions=True,
            notes=(
                "Test-only synthetic capacity assumptions."
            ),
        ),
        capacity=CapacityConfig(
            total_additional_cases=(
                total_additional_cases
            ),
            default_max_additional_cases_per_service=(
                max_additional_cases_per_service
            ),
        ),
        objective=ObjectiveConfig(
            waiting_weight=waiting_weight,
            long_wait_weight=long_wait_weight,
        ),
        constraints=ConstraintConfig(
            enforce_waiting_volume_ceiling=(
                enforce_waiting_volume_ceiling
            ),
            enforce_service_capacity_ceiling=(
                enforce_service_capacity_ceiling
            ),
        ),
        solver=SolverConfig(
            max_time_seconds=5.0,
            num_search_workers=1,
            random_seed=0,
        ),
    )


def _row(
    *,
    facility_key: str,
    service_key: str,
    vol_waiting: int,
    vol_long_waits: int,
    resource_kind: str = "specialty",
    reporting_period: date = REPORTING_PERIOD,
) -> OptimisationInputRow:
    """Build one valid optimisation row for solver tests."""
    if resource_kind not in {
        "specialty",
        "category",
    }:
        raise ValueError(
            "Unsupported resource_kind in test helper."
        )

    return OptimisationInputRow(
        facility_key=facility_key,
        facility_name=f"Facility {facility_key}",
        service_key=service_key,
        service_name=f"Service {service_key}",
        resource_kind=resource_kind,
        reporting_period=reporting_period,
        vol_waiting=vol_waiting,
        vol_long_waits=vol_long_waits,
    )


def test_solver_returns_optimal_summary() -> None:
    """A straightforward bounded problem should solve optimally."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=20,
            vol_long_waits=10,
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=20,
            vol_long_waits=5,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=10,
        max_additional_cases_per_service=10,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    assert result.status == "optimal"
    assert result.scenario_name == "solver_test"
    assert result.objective_value is not None
    assert result.total_capacity_available == 10
    assert result.total_additional_cases == 10
    assert result.unused_capacity == 0

    assert len(
        result.allocations
    ) == 2

    assert sum(
        allocation.additional_cases
        for allocation in result.allocations
    ) == 10


def test_solver_prioritises_higher_long_wait_burden() -> None:
    """Higher long-wait share should receive capacity first."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="high-burden",
            vol_waiting=10,
            vol_long_waits=8,
        ),
        _row(
            facility_key="facility-b",
            service_key="low-burden",
            vol_waiting=10,
            vol_long_waits=1,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=10,
        max_additional_cases_per_service=10,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    allocations = {
        allocation.service_key: (
            allocation.additional_cases
        )
        for allocation in result.allocations
    }

    assert result.status == "optimal"

    assert allocations[
        "high-burden"
    ] == 10

    assert allocations[
        "low-burden"
    ] == 0


def test_solver_respects_global_capacity_ceiling() -> None:
    """Allocated activity must not exceed total scenario capacity."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=50,
            vol_long_waits=25,
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=50,
            vol_long_waits=20,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=7,
        max_additional_cases_per_service=20,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    assert result.status == "optimal"
    assert result.total_additional_cases == 7
    assert result.total_additional_cases <= 7
    assert result.unused_capacity == 0


def test_solver_respects_service_capacity_ceiling() -> None:
    """Each allocation must respect the configured service ceiling."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=100,
            vol_long_waits=80,
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=100,
            vol_long_waits=10,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=10,
        max_additional_cases_per_service=6,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    assert result.status == "optimal"
    assert result.total_additional_cases == 10

    assert all(
        allocation.additional_cases <= 6
        for allocation in result.allocations
    )


def test_solver_respects_waiting_volume_ceiling() -> None:
    """Allocation must not exceed the observed waiting volume."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=3,
            vol_long_waits=3,
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=20,
            vol_long_waits=1,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=10,
        max_additional_cases_per_service=10,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    allocations = {
        allocation.service_key: (
            allocation.additional_cases
        )
        for allocation in result.allocations
    }

    assert result.status == "optimal"

    assert allocations[
        "service-a"
    ] == 3

    assert allocations[
        "service-b"
    ] == 7


def test_solver_reports_unused_capacity_when_demand_is_lower() -> None:
    """Unused capacity should be retained when demand limits allocation."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=2,
            vol_long_waits=1,
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=3,
            vol_long_waits=1,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=10,
        max_additional_cases_per_service=10,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    assert result.status == "optimal"
    assert result.total_capacity_available == 10
    assert result.total_additional_cases == 5
    assert result.unused_capacity == 5


def test_solver_handles_zero_capacity() -> None:
    """A zero-capacity scenario should produce zero allocations."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=20,
            vol_long_waits=10,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=0,
        max_additional_cases_per_service=10,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    assert result.status == "optimal"
    assert result.total_capacity_available == 0
    assert result.total_additional_cases == 0
    assert result.unused_capacity == 0

    assert result.allocations[
        0
    ].additional_cases == 0


def test_solver_preserves_allocation_identity_fields() -> None:
    """Solver output should retain source row identity and context."""
    input_row = _row(
        facility_key="facility-a",
        service_key="general-surgery",
        vol_waiting=10,
        vol_long_waits=5,
    )

    scenario = _scenario(
        total_additional_cases=4,
        max_additional_cases_per_service=4,
    )

    result = solve_capacity_allocation(
        inputs=[input_row],
        scenario=scenario,
    )

    allocation = result.allocations[
        0
    ]

    assert allocation.facility_key == (
        input_row.facility_key
    )
    assert allocation.facility_name == (
        input_row.facility_name
    )
    assert allocation.service_key == (
        input_row.service_key
    )
    assert allocation.service_name == (
        input_row.service_name
    )
    assert allocation.resource_kind == (
        input_row.resource_kind
    )
    assert allocation.reporting_period == (
        input_row.reporting_period
    )


def test_solver_rejects_scenario_resource_kind_mismatch() -> None:
    """Scenario and input source families must agree."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=10,
            vol_long_waits=2,
            resource_kind="specialty",
        ),
    ]

    scenario = _scenario(
        resource_kind="category",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Scenario resource_kind does not match "
            "optimisation inputs"
        ),
    ):
        solve_capacity_allocation(
            inputs=inputs,
            scenario=scenario,
        )


def test_solver_rejects_mixed_resource_kinds() -> None:
    """One solve must operate on one published source family."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=10,
            vol_long_waits=2,
            resource_kind="specialty",
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=10,
            vol_long_waits=2,
            resource_kind="category",
        ),
    ]

    scenario = _scenario(
        resource_kind="specialty",
    )

    with pytest.raises(
        ValueError,
        match=(
            "must contain exactly one resource_kind"
        ),
    ):
        solve_capacity_allocation(
            inputs=inputs,
            scenario=scenario,
        )


def test_solver_rejects_mixed_reporting_periods() -> None:
    """One solve must operate on one reporting period."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=10,
            vol_long_waits=2,
            reporting_period=date(
                2025,
                6,
                1,
            ),
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=10,
            vol_long_waits=2,
            reporting_period=date(
                2025,
                7,
                1,
            ),
        ),
    ]

    scenario = _scenario()

    with pytest.raises(
        ValueError,
        match=(
            "must contain exactly one reporting period"
        ),
    ):
        solve_capacity_allocation(
            inputs=inputs,
            scenario=scenario,
        )


def test_solver_rejects_duplicate_allocation_keys() -> None:
    """Facility-service keys must be unique before solving."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=10,
            vol_long_waits=2,
        ),
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=20,
            vol_long_waits=3,
        ),
    ]

    scenario = _scenario()

    with pytest.raises(
        ValueError,
        match=(
            "duplicate facility-service keys"
        ),
    ):
        solve_capacity_allocation(
            inputs=inputs,
            scenario=scenario,
        )


def test_solver_summary_reconciles_allocations() -> None:
    """Summary totals must reconcile to allocation decisions."""
    inputs = [
        _row(
            facility_key="facility-a",
            service_key="service-a",
            vol_waiting=20,
            vol_long_waits=15,
        ),
        _row(
            facility_key="facility-b",
            service_key="service-b",
            vol_waiting=20,
            vol_long_waits=10,
        ),
        _row(
            facility_key="facility-c",
            service_key="service-c",
            vol_waiting=20,
            vol_long_waits=5,
        ),
    ]

    scenario = _scenario(
        total_additional_cases=12,
        max_additional_cases_per_service=5,
    )

    result = solve_capacity_allocation(
        inputs=inputs,
        scenario=scenario,
    )

    allocation_total = sum(
        allocation.additional_cases
        for allocation in result.allocations
    )

    assert result.status == "optimal"

    assert (
        result.total_additional_cases
        == allocation_total
    )

    assert (
        result.total_capacity_available
        == result.total_additional_cases
        + result.unused_capacity
    )