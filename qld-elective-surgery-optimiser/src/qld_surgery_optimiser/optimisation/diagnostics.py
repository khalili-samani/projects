"""Diagnostics and allocation metrics for optimisation results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qld_surgery_optimiser.optimisation.constraints import (
    AllocationKey,
    allocation_key,
)
from qld_surgery_optimiser.optimisation.models import (
    AllocationDecision,
    OptimisationInputRow,
    OptimisationSummary,
    ResourceKind,
)
from qld_surgery_optimiser.optimisation.scenarios import (
    OptimisationScenario,
)


@dataclass(
    frozen=True,
    slots=True,
)
class ServiceAllocationDiagnostic:
    """Diagnostic metrics for one facility-service allocation."""

    facility_key: str
    facility_name: str
    service_key: str
    service_name: str
    resource_kind: ResourceKind

    vol_waiting: int
    vol_long_waits: int
    long_wait_share: float

    additional_cases: int
    allocation_share_of_waiting: float

    receives_capacity: bool
    waiting_ceiling_binding: bool
    service_ceiling_binding: bool


@dataclass(
    frozen=True,
    slots=True,
)
class OptimisationDiagnostics:
    """Aggregate diagnostics for one solved optimisation scenario."""

    scenario_name: str
    status: str

    total_capacity_available: int
    total_additional_cases: int
    unused_capacity: int
    capacity_utilisation: float

    services_considered: int
    services_receiving_capacity: int

    waiting_ceiling_binding_count: int
    service_ceiling_binding_count: int

    service_diagnostics: tuple[
        ServiceAllocationDiagnostic,
        ...,
    ]


def build_optimisation_diagnostics(
    *,
    inputs: Sequence[OptimisationInputRow],
    summary: OptimisationSummary,
    scenario: OptimisationScenario,
) -> OptimisationDiagnostics:
    """Build descriptive diagnostics for a solved optimisation result.

    Diagnostics are derived from observed optimisation inputs,
    synthetic scenario assumptions and solver allocation decisions.

    They do not represent observed post-intervention healthcare
    outcomes.
    """
    if summary.status not in {
        "optimal",
        "feasible",
    }:
        raise ValueError(
            "Allocation diagnostics require an optimal or feasible "
            "solver result."
        )

    if summary.scenario_name != scenario.scenario.name:
        raise ValueError(
            "Optimisation summary scenario_name does not match "
            "the supplied scenario."
        )

    _validate_input_context(
        inputs=inputs,
        scenario=scenario,
    )

    input_by_key = {
        allocation_key(
            row
        ): row
        for row in inputs
    }

    allocation_by_key = _build_allocation_mapping(
        summary.allocations
    )

    input_keys = set(
        input_by_key
    )

    allocation_keys = set(
        allocation_by_key
    )

    if input_keys != allocation_keys:
        missing = sorted(
            input_keys
            - allocation_keys
        )

        unexpected = sorted(
            allocation_keys
            - input_keys
        )

        raise ValueError(
            "Allocation diagnostics require exactly one allocation "
            "for every optimisation input row. "
            f"Missing allocations: {_format_keys(missing)}. "
            f"Unexpected allocations: {_format_keys(unexpected)}."
        )

    service_diagnostics = tuple(
        _build_service_diagnostic(
            row=input_by_key[
                key
            ],
            allocation=allocation_by_key[
                key
            ],
            scenario=scenario,
        )
        for key in sorted(
            input_keys
        )
    )

    services_receiving_capacity = sum(
        diagnostic.receives_capacity
        for diagnostic in service_diagnostics
    )

    waiting_ceiling_binding_count = sum(
        diagnostic.waiting_ceiling_binding
        for diagnostic in service_diagnostics
    )

    service_ceiling_binding_count = sum(
        diagnostic.service_ceiling_binding
        for diagnostic in service_diagnostics
    )

    capacity_utilisation = _capacity_utilisation(
        total_additional_cases=(
            summary.total_additional_cases
        ),
        total_capacity_available=(
            summary.total_capacity_available
        ),
    )

    return OptimisationDiagnostics(
        scenario_name=summary.scenario_name,
        status=summary.status,
        total_capacity_available=(
            summary.total_capacity_available
        ),
        total_additional_cases=(
            summary.total_additional_cases
        ),
        unused_capacity=summary.unused_capacity,
        capacity_utilisation=capacity_utilisation,
        services_considered=len(
            inputs
        ),
        services_receiving_capacity=(
            services_receiving_capacity
        ),
        waiting_ceiling_binding_count=(
            waiting_ceiling_binding_count
        ),
        service_ceiling_binding_count=(
            service_ceiling_binding_count
        ),
        service_diagnostics=service_diagnostics,
    )


def _build_service_diagnostic(
    *,
    row: OptimisationInputRow,
    allocation: AllocationDecision,
    scenario: OptimisationScenario,
) -> ServiceAllocationDiagnostic:
    """Build diagnostics for one optimisation allocation."""
    if allocation.additional_cases > row.vol_waiting:
        if (
            scenario.constraints.enforce_waiting_volume_ceiling
        ):
            raise ValueError(
                "Allocation exceeds waiting volume while the "
                "waiting-volume ceiling is enabled."
            )

    long_wait_share = _safe_ratio(
        numerator=row.vol_long_waits,
        denominator=row.vol_waiting,
    )

    allocation_share_of_waiting = _safe_ratio(
        numerator=allocation.additional_cases,
        denominator=row.vol_waiting,
    )

    waiting_ceiling_binding = (
        scenario.constraints.enforce_waiting_volume_ceiling
        and allocation.additional_cases
        == row.vol_waiting
    )

    service_ceiling_binding = (
        scenario.constraints.enforce_service_capacity_ceiling
        and allocation.additional_cases
        == (
            scenario.capacity
            .default_max_additional_cases_per_service
        )
    )

    return ServiceAllocationDiagnostic(
        facility_key=row.facility_key,
        facility_name=row.facility_name,
        service_key=row.service_key,
        service_name=row.service_name,
        resource_kind=row.resource_kind,
        vol_waiting=row.vol_waiting,
        vol_long_waits=row.vol_long_waits,
        long_wait_share=long_wait_share,
        additional_cases=allocation.additional_cases,
        allocation_share_of_waiting=(
            allocation_share_of_waiting
        ),
        receives_capacity=(
            allocation.additional_cases > 0
        ),
        waiting_ceiling_binding=(
            waiting_ceiling_binding
        ),
        service_ceiling_binding=(
            service_ceiling_binding
        ),
    )


def _build_allocation_mapping(
    allocations: Sequence[AllocationDecision],
) -> dict[
    AllocationKey,
    AllocationDecision,
]:
    """Return allocations keyed by facility and service."""
    result: dict[
        AllocationKey,
        AllocationDecision,
    ] = {}

    for allocation in allocations:
        key = (
            allocation.facility_key,
            allocation.service_key,
        )

        if key in result:
            raise ValueError(
                "Optimisation summary contains duplicate "
                "facility-service allocations: "
                f"{key[0]} / {key[1]}."
            )

        result[
            key
        ] = allocation

    return result


def _validate_input_context(
    *,
    inputs: Sequence[OptimisationInputRow],
    scenario: OptimisationScenario,
) -> None:
    """Validate context required for allocation diagnostics."""
    input_keys = [
        allocation_key(
            row
        )
        for row in inputs
    ]

    if len(
        input_keys
    ) != len(
        set(
            input_keys
        )
    ):
        raise ValueError(
            "Optimisation diagnostic inputs contain duplicate "
            "facility-service keys."
        )

    resource_kinds = {
        row.resource_kind
        for row in inputs
    }

    if len(
        resource_kinds
    ) > 1:
        raise ValueError(
            "Optimisation diagnostic inputs must contain exactly "
            "one resource_kind."
        )

    if (
        resource_kinds
        and scenario.scenario.resource_kind
        not in resource_kinds
    ):
        raise ValueError(
            "Scenario resource_kind does not match diagnostic "
            "input rows."
        )

    reporting_periods = {
        row.reporting_period
        for row in inputs
    }

    if len(
        reporting_periods
    ) > 1:
        raise ValueError(
            "Optimisation diagnostic inputs must contain exactly "
            "one reporting period."
        )


def _safe_ratio(
    *,
    numerator: int,
    denominator: int,
) -> float:
    """Return a ratio while treating a zero denominator as zero."""
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def _capacity_utilisation(
    *,
    total_additional_cases: int,
    total_capacity_available: int,
) -> float:
    """Return the proportion of scenario capacity allocated."""
    if total_capacity_available == 0:
        return 0.0

    return (
        total_additional_cases
        / total_capacity_available
    )


def _format_keys(
    keys: Sequence[AllocationKey],
) -> str:
    """Format allocation keys for validation messages."""
    if not keys:
        return "none"

    return ", ".join(
        f"{facility_key} / {service_key}"
        for (
            facility_key,
            service_key,
        )
        in keys
    )