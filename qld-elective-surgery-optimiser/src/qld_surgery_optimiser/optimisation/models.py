"""Typed domain models for elective surgery capacity optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


ResourceKind = Literal[
    "specialty",
    "category",
]

SolverStatus = Literal[
    "optimal",
    "feasible",
    "infeasible",
    "model_invalid",
    "unknown",
]


@dataclass(
    frozen=True,
    slots=True,
)
class OptimisationInputRow:
    """One observed service-level row supplied to the optimiser.

    This model represents analytical data derived from the validated
    warehouse. It must not contain synthetic capacity assumptions.

    Capacity assumptions belong to the scenario configuration so that
    observed source data and model assumptions remain explicitly
    separated.
    """

    facility_key: str
    facility_name: str
    service_key: str
    service_name: str
    resource_kind: ResourceKind
    reporting_period: date
    vol_waiting: int
    vol_long_waits: int

    def __post_init__(self) -> None:
        """Validate invariants required by the optimisation layer."""
        if not self.facility_key.strip():
            raise ValueError(
                "facility_key must not be empty."
            )

        if not self.facility_name.strip():
            raise ValueError(
                "facility_name must not be empty."
            )

        if not self.service_key.strip():
            raise ValueError(
                "service_key must not be empty."
            )

        if not self.service_name.strip():
            raise ValueError(
                "service_name must not be empty."
            )

        if self.vol_waiting < 0:
            raise ValueError(
                "vol_waiting must not be negative."
            )

        if self.vol_long_waits < 0:
            raise ValueError(
                "vol_long_waits must not be negative."
            )

        if (
            self.vol_long_waits
            > self.vol_waiting
        ):
            raise ValueError(
                "vol_long_waits must not exceed "
                "vol_waiting."
            )


@dataclass(
    frozen=True,
    slots=True,
)
class AllocationDecision:
    """One capacity-allocation decision produced by the optimiser."""

    facility_key: str
    facility_name: str
    service_key: str
    service_name: str
    resource_kind: ResourceKind
    reporting_period: date
    additional_cases: int

    def __post_init__(self) -> None:
        """Validate an optimisation allocation."""
        if not self.facility_key.strip():
            raise ValueError(
                "facility_key must not be empty."
            )

        if not self.service_key.strip():
            raise ValueError(
                "service_key must not be empty."
            )

        if self.additional_cases < 0:
            raise ValueError(
                "additional_cases must not be negative."
            )


@dataclass(
    frozen=True,
    slots=True,
)
class OptimisationSummary:
    """Summary returned by one optimisation run."""

    scenario_name: str
    status: SolverStatus
    objective_value: float | None
    total_capacity_available: int
    total_additional_cases: int
    unused_capacity: int
    allocations: tuple[
        AllocationDecision,
        ...,
    ]

    def __post_init__(self) -> None:
        """Validate optimisation-result invariants."""
        if not self.scenario_name.strip():
            raise ValueError(
                "scenario_name must not be empty."
            )

        if self.total_capacity_available < 0:
            raise ValueError(
                "total_capacity_available must not "
                "be negative."
            )

        if self.total_additional_cases < 0:
            raise ValueError(
                "total_additional_cases must not "
                "be negative."
            )

        if self.unused_capacity < 0:
            raise ValueError(
                "unused_capacity must not be negative."
            )

        if (
            self.total_additional_cases
            > self.total_capacity_available
        ):
            raise ValueError(
                "total_additional_cases must not exceed "
                "total_capacity_available."
            )

        expected_unused_capacity = (
            self.total_capacity_available
            - self.total_additional_cases
        )

        if (
            self.unused_capacity
            != expected_unused_capacity
        ):
            raise ValueError(
                "unused_capacity must equal "
                "total_capacity_available minus "
                "total_additional_cases."
            )

        allocation_total = sum(
            allocation.additional_cases
            for allocation
            in self.allocations
        )

        if (
            allocation_total
            != self.total_additional_cases
        ):
            raise ValueError(
                "Allocation totals must equal "
                "total_additional_cases."
            )