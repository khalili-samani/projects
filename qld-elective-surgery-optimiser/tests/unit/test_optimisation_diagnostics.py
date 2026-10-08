"""Tests for optimisation diagnostics and allocation metrics."""

from __future__ import annotations

from datetime import date

import pytest

from qld_surgery_optimiser.optimisation.diagnostics import (
    build_optimisation_diagnostics,
)
from qld_surgery_optimiser.optimisation.models import (
    AllocationDecision,
    OptimisationInputRow,
    OptimisationSummary,
    ResourceKind,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    CapacityConfig,
    ConstraintConfig,
    ObjectiveConfig,
    OptimisationScenario,
    ScenarioMetadata,
    SolverConfig,
)


REPORTING_PERIOD = date(
    2025,
    6,
    1,
)


def _scenario(
    *,
    resource_kind: ResourceKind = "specialty",
    total_capacity: int = 10,
    service_capacity: int = 6,
    enforce_waiting_ceiling: bool = True,
    enforce_service_ceiling: bool = True,
) -> OptimisationScenario:
    """Build a deterministic test scenario."""
    return OptimisationScenario(
        scenario=ScenarioMetadata(
            name="diagnostic_test",
            description=(
                "Synthetic optimisation diagnostics test."
            ),
            resource_kind=resource_kind,
            synthetic_assumptions=True,
            notes="Test-only assumptions.",
        ),
        capacity=CapacityConfig(
            total_additional_cases=(
                total_capacity
            ),
            default_max_additional_cases_per_service=(
                service_capacity
            ),
        ),
        objective=ObjectiveConfig(
            waiting_weight=1,
            long_wait_weight=3,
        ),
        constraints=ConstraintConfig(
            enforce_waiting_volume_ceiling=(
                enforce_waiting_ceiling
            ),
            enforce_service_capacity_ceiling=(
                enforce_service_ceiling
            ),
        ),
        solver=SolverConfig(
            max_time_seconds=5.0,
            num_search_workers=1,
            random_seed=0,
        ),
    )


def _input_row(
    *,
    facility_key: str,
    service_key: str,
    vol_waiting: int,
    vol_long_waits: int,
    resource_kind: ResourceKind = "specialty",
    reporting_period: date = REPORTING_PERIOD,
) -> OptimisationInputRow:
    """Build one optimisation input row."""
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


def _allocation(
    *,
    row: OptimisationInputRow,
    additional_cases: int,
) -> AllocationDecision:
    """Build one allocation matching an optimisation input."""
    return AllocationDecision(
        facility_key=row.facility_key,
        facility_name=row.facility_name,
        service_key=row.service_key,
        service_name=row.service_name,
        resource_kind=row.resource_kind,
        reporting_period=row.reporting_period,
        additional_cases=additional_cases,
    )


def _summary(
    *,
    allocations: tuple[
        AllocationDecision,
        ...,
    ],
    total_capacity: int,
    scenario_name: str = "diagnostic_test",
    status: str = "optimal",
) -> OptimisationSummary:
    """Build a reconciled optimisation summary."""
    total_additional_cases = sum(
        allocation.additional_cases
        for allocation in allocations
    )

    return OptimisationSummary(
        scenario_name=scenario_name,
        status=status,
        objective_value=100.0,
        total_capacity_available=(
            total_capacity
        ),
        total_additional_cases=(
            total_additional_cases
        ),
        unused_capacity=(
            total_capacity
            - total_additional_cases
        ),
        allocations=allocations,
    )


def test_builds_aggregate_diagnostics() -> None:
    """Aggregate diagnostics should reconcile with solver output."""
    row_a = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=10,
        vol_long_waits=5,
    )

    row_b = _input_row(
        facility_key="facility-b",
        service_key="service-b",
        vol_waiting=20,
        vol_long_waits=4,
    )

    allocations = (
        _allocation(
            row=row_a,
            additional_cases=6,
        ),
        _allocation(
            row=row_b,
            additional_cases=4,
        ),
    )

    summary = _summary(
        allocations=allocations,
        total_capacity=10,
    )

    diagnostics = build_optimisation_diagnostics(
        inputs=[
            row_a,
            row_b,
        ],
        summary=summary,
        scenario=_scenario(
            total_capacity=10,
            service_capacity=6,
        ),
    )

    assert diagnostics.scenario_name == (
        "diagnostic_test"
    )
    assert diagnostics.status == "optimal"
    assert diagnostics.total_capacity_available == 10
    assert diagnostics.total_additional_cases == 10
    assert diagnostics.unused_capacity == 0
    assert diagnostics.capacity_utilisation == 1.0
    assert diagnostics.services_considered == 2
    assert diagnostics.services_receiving_capacity == 2


def test_calculates_service_ratios() -> None:
    """Service diagnostics should expose descriptive allocation ratios."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=20,
        vol_long_waits=5,
    )

    allocation = _allocation(
        row=row,
        additional_cases=10,
    )

    diagnostics = build_optimisation_diagnostics(
        inputs=[row],
        summary=_summary(
            allocations=(
                allocation,
            ),
            total_capacity=10,
        ),
        scenario=_scenario(
            total_capacity=10,
            service_capacity=10,
        ),
    )

    service = diagnostics.service_diagnostics[
        0
    ]

    assert service.long_wait_share == 0.25
    assert service.allocation_share_of_waiting == 0.5
    assert service.receives_capacity is True


def test_zero_waiting_produces_zero_ratios() -> None:
    """Zero waiting volumes should not cause division errors."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=0,
        vol_long_waits=0,
    )

    allocation = _allocation(
        row=row,
        additional_cases=0,
    )

    diagnostics = build_optimisation_diagnostics(
        inputs=[row],
        summary=_summary(
            allocations=(
                allocation,
            ),
            total_capacity=0,
        ),
        scenario=_scenario(
            total_capacity=0,
            service_capacity=0,
        ),
    )

    service = diagnostics.service_diagnostics[
        0
    ]

    assert service.long_wait_share == 0.0
    assert service.allocation_share_of_waiting == 0.0
    assert diagnostics.capacity_utilisation == 0.0


def test_counts_services_receiving_capacity() -> None:
    """Only positive allocations should count as receiving capacity."""
    row_a = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=10,
        vol_long_waits=5,
    )

    row_b = _input_row(
        facility_key="facility-b",
        service_key="service-b",
        vol_waiting=10,
        vol_long_waits=2,
    )

    allocations = (
        _allocation(
            row=row_a,
            additional_cases=5,
        ),
        _allocation(
            row=row_b,
            additional_cases=0,
        ),
    )

    diagnostics = build_optimisation_diagnostics(
        inputs=[
            row_a,
            row_b,
        ],
        summary=_summary(
            allocations=allocations,
            total_capacity=10,
        ),
        scenario=_scenario(
            total_capacity=10,
        ),
    )

    assert diagnostics.services_receiving_capacity == 1
    assert diagnostics.capacity_utilisation == 0.5


def test_identifies_waiting_ceiling_binding() -> None:
    """Allocation equal to waiting volume should flag its ceiling."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=4,
        vol_long_waits=2,
    )

    allocation = _allocation(
        row=row,
        additional_cases=4,
    )

    diagnostics = build_optimisation_diagnostics(
        inputs=[row],
        summary=_summary(
            allocations=(
                allocation,
            ),
            total_capacity=10,
        ),
        scenario=_scenario(
            total_capacity=10,
            service_capacity=10,
        ),
    )

    service = diagnostics.service_diagnostics[
        0
    ]

    assert service.waiting_ceiling_binding is True
    assert diagnostics.waiting_ceiling_binding_count == 1


def test_identifies_service_ceiling_binding() -> None:
    """Allocation equal to service capacity should flag its ceiling."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=20,
        vol_long_waits=10,
    )

    allocation = _allocation(
        row=row,
        additional_cases=6,
    )

    diagnostics = build_optimisation_diagnostics(
        inputs=[row],
        summary=_summary(
            allocations=(
                allocation,
            ),
            total_capacity=10,
        ),
        scenario=_scenario(
            total_capacity=10,
            service_capacity=6,
        ),
    )

    service = diagnostics.service_diagnostics[
        0
    ]

    assert service.service_ceiling_binding is True
    assert diagnostics.service_ceiling_binding_count == 1


def test_disabled_ceiling_is_not_reported_as_binding() -> None:
    """Disabled constraints must not be labelled as binding."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=6,
        vol_long_waits=2,
    )

    allocation = _allocation(
        row=row,
        additional_cases=6,
    )

    diagnostics = build_optimisation_diagnostics(
        inputs=[row],
        summary=_summary(
            allocations=(
                allocation,
            ),
            total_capacity=10,
        ),
        scenario=_scenario(
            total_capacity=10,
            service_capacity=6,
            enforce_waiting_ceiling=False,
            enforce_service_ceiling=False,
        ),
    )

    service = diagnostics.service_diagnostics[
        0
    ]

    assert service.waiting_ceiling_binding is False
    assert service.service_ceiling_binding is False


def test_rejects_non_feasible_solver_status() -> None:
    """Allocation diagnostics require a solved allocation."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=10,
        vol_long_waits=2,
    )

    summary = OptimisationSummary(
        scenario_name="diagnostic_test",
        status="infeasible",
        objective_value=None,
        total_capacity_available=10,
        total_additional_cases=0,
        unused_capacity=10,
        allocations=(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "require an optimal or feasible solver result"
        ),
    ):
        build_optimisation_diagnostics(
            inputs=[row],
            summary=summary,
            scenario=_scenario(),
        )


def test_rejects_scenario_name_mismatch() -> None:
    """Summary and scenario identifiers must agree."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=10,
        vol_long_waits=2,
    )

    allocation = _allocation(
        row=row,
        additional_cases=5,
    )

    with pytest.raises(
        ValueError,
        match=(
            "scenario_name does not match"
        ),
    ):
        build_optimisation_diagnostics(
            inputs=[row],
            summary=_summary(
                allocations=(
                    allocation,
                ),
                total_capacity=10,
                scenario_name="wrong-scenario",
            ),
            scenario=_scenario(),
        )


def test_rejects_missing_allocation() -> None:
    """Every diagnostic input must have a matching allocation."""
    row_a = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=10,
        vol_long_waits=2,
    )

    row_b = _input_row(
        facility_key="facility-b",
        service_key="service-b",
        vol_waiting=10,
        vol_long_waits=2,
    )

    allocation = _allocation(
        row=row_a,
        additional_cases=5,
    )

    with pytest.raises(
        ValueError,
        match=(
            "exactly one allocation for every "
            "optimisation input row"
        ),
    ):
        build_optimisation_diagnostics(
            inputs=[
                row_a,
                row_b,
            ],
            summary=_summary(
                allocations=(
                    allocation,
                ),
                total_capacity=10,
            ),
            scenario=_scenario(),
        )


def test_rejects_duplicate_allocations() -> None:
    """Duplicate facility-service allocations must be rejected."""
    row = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=10,
        vol_long_waits=2,
    )

    allocation = _allocation(
        row=row,
        additional_cases=5,
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate facility-service allocations"
        ),
    ):
        build_optimisation_diagnostics(
            inputs=[row],
            summary=_summary(
                allocations=(
                    allocation,
                    allocation,
                ),
                total_capacity=10,
            ),
            scenario=_scenario(),
        )


def test_rejects_duplicate_input_keys() -> None:
    """Diagnostic inputs must retain the optimisation grain."""
    row_a = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=10,
        vol_long_waits=2,
    )

    row_b = _input_row(
        facility_key="facility-a",
        service_key="service-a",
        vol_waiting=20,
        vol_long_waits=3,
    )

    allocation = _allocation(
        row=row_a,
        additional_cases=5,
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate facility-service keys"
        ),
    ):
        build_optimisation_diagnostics(
            inputs=[
                row_a,
                row_b,
            ],
            summary=_summary(
                allocations=(
                    allocation,
                ),
                total_capacity=10,
            ),
            scenario=_scenario(),
        )