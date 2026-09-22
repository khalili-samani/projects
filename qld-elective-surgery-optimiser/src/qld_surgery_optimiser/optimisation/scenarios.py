"""Typed loading and validation for optimisation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml


ResourceKind = Literal[
    "specialty",
    "category",
]


@dataclass(
    frozen=True,
    slots=True,
)
class ScenarioMetadata:
    """Human-readable metadata describing an optimisation scenario."""

    name: str
    description: str
    resource_kind: ResourceKind
    synthetic_assumptions: bool
    notes: str


@dataclass(
    frozen=True,
    slots=True,
)
class CapacityConfig:
    """Capacity assumptions supplied to the optimisation model."""

    total_additional_cases: int
    default_max_additional_cases_per_service: int


@dataclass(
    frozen=True,
    slots=True,
)
class ObjectiveConfig:
    """Objective weights used by the optimisation model."""

    waiting_weight: int
    long_wait_weight: int


@dataclass(
    frozen=True,
    slots=True,
)
class ConstraintConfig:
    """Constraint switches controlling baseline model behaviour."""

    enforce_waiting_volume_ceiling: bool
    enforce_service_capacity_ceiling: bool


@dataclass(
    frozen=True,
    slots=True,
)
class SolverConfig:
    """OR-Tools solver settings."""

    max_time_seconds: float
    num_search_workers: int
    random_seed: int


@dataclass(
    frozen=True,
    slots=True,
)
class OptimisationScenario:
    """Complete validated optimisation scenario."""

    scenario: ScenarioMetadata
    capacity: CapacityConfig
    objective: ObjectiveConfig
    constraints: ConstraintConfig
    solver: SolverConfig


def load_scenario(
    path: Path,
) -> OptimisationScenario:
    """Load and validate an optimisation scenario YAML file.

    Parameters
    ----------
    path:
        Path to a YAML scenario configuration.

    Returns
    -------
    OptimisationScenario
        Fully validated, immutable scenario configuration.

    Raises
    ------
    FileNotFoundError
        If the scenario file does not exist.
    ValueError
        If the YAML structure or any scenario value is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Scenario path is not a file: {path}"
        )

    try:
        raw = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Scenario YAML could not be parsed: {path}"
        ) from exc

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            "Scenario YAML must contain a mapping at the root."
        )

    data = cast(
        dict[str, Any],
        raw,
    )

    return parse_scenario(
        data
    )


def parse_scenario(
    data: dict[str, Any],
) -> OptimisationScenario:
    """Validate a parsed scenario mapping."""
    scenario_section = _required_mapping(
        data,
        "scenario",
    )

    capacity_section = _required_mapping(
        data,
        "capacity",
    )

    objective_section = _required_mapping(
        data,
        "objective",
    )

    constraints_section = _required_mapping(
        data,
        "constraints",
    )

    solver_section = _required_mapping(
        data,
        "solver",
    )

    scenario = ScenarioMetadata(
        name=_required_text(
            scenario_section,
            "name",
        ),
        description=_required_text(
            scenario_section,
            "description",
        ),
        resource_kind=_required_resource_kind(
            scenario_section,
            "resource_kind",
        ),
        synthetic_assumptions=_required_bool(
            scenario_section,
            "synthetic_assumptions",
        ),
        notes=_required_text(
            scenario_section,
            "notes",
        ),
    )

    if not scenario.synthetic_assumptions:
        raise ValueError(
            "Phase 5 scenario configurations must explicitly declare "
            "synthetic_assumptions: true unless a future scenario "
            "is backed by independently verified capacity data."
        )

    capacity = CapacityConfig(
        total_additional_cases=_required_non_negative_int(
            capacity_section,
            "total_additional_cases",
        ),
        default_max_additional_cases_per_service=(
            _required_non_negative_int(
                capacity_section,
                "default_max_additional_cases_per_service",
            )
        ),
    )

    objective = ObjectiveConfig(
        waiting_weight=_required_non_negative_int(
            objective_section,
            "waiting_weight",
        ),
        long_wait_weight=_required_non_negative_int(
            objective_section,
            "long_wait_weight",
        ),
    )

    if (
        objective.waiting_weight == 0
        and objective.long_wait_weight == 0
    ):
        raise ValueError(
            "At least one objective weight must be greater than zero."
        )

    constraints = ConstraintConfig(
        enforce_waiting_volume_ceiling=_required_bool(
            constraints_section,
            "enforce_waiting_volume_ceiling",
        ),
        enforce_service_capacity_ceiling=_required_bool(
            constraints_section,
            "enforce_service_capacity_ceiling",
        ),
    )

    solver = SolverConfig(
        max_time_seconds=_required_positive_float(
            solver_section,
            "max_time_seconds",
        ),
        num_search_workers=_required_positive_int(
            solver_section,
            "num_search_workers",
        ),
        random_seed=_required_non_negative_int(
            solver_section,
            "random_seed",
        ),
    )

    return OptimisationScenario(
        scenario=scenario,
        capacity=capacity,
        objective=objective,
        constraints=constraints,
        solver=solver,
    )


def _required_mapping(
    data: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return a required mapping section."""
    if key not in data:
        raise ValueError(
            f"Scenario configuration is missing section {key!r}."
        )

    value = data[
        key
    ]

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"Scenario section {key!r} must be a mapping."
        )

    return {
        str(
            nested_key
        ): nested_value
        for (
            nested_key,
            nested_value,
        )
        in value.items()
    }


def _required_text(
    data: dict[str, Any],
    key: str,
) -> str:
    """Return a required non-empty string."""
    if key not in data:
        raise ValueError(
            f"Missing required scenario field {key!r}."
        )

    value = data[
        key
    ]

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"Scenario field {key!r} must be a string."
        )

    text = value.strip()

    if not text:
        raise ValueError(
            f"Scenario field {key!r} must not be empty."
        )

    return text


def _required_resource_kind(
    data: dict[str, Any],
    key: str,
) -> ResourceKind:
    """Return a supported optimisation resource kind."""
    value = _required_text(
        data,
        key,
    ).casefold()

    if value == "specialty":
        return "specialty"

    if value == "category":
        return "category"

    raise ValueError(
        "Scenario resource_kind must be either "
        "'specialty' or 'category'."
    )


def _required_bool(
    data: dict[str, Any],
    key: str,
) -> bool:
    """Return a required boolean."""
    if key not in data:
        raise ValueError(
            f"Missing required scenario field {key!r}."
        )

    value = data[
        key
    ]

    if not isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"Scenario field {key!r} must be a boolean."
        )

    return value


def _required_non_negative_int(
    data: dict[str, Any],
    key: str,
) -> int:
    """Return a required integer greater than or equal to zero."""
    if key not in data:
        raise ValueError(
            f"Missing required scenario field {key!r}."
        )

    value = data[
        key
    ]

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise ValueError(
            f"Scenario field {key!r} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"Scenario field {key!r} must not be negative."
        )

    return value


def _required_positive_int(
    data: dict[str, Any],
    key: str,
) -> int:
    """Return a required strictly positive integer."""
    value = _required_non_negative_int(
        data,
        key,
    )

    if value == 0:
        raise ValueError(
            f"Scenario field {key!r} must be greater than zero."
        )

    return value


def _required_positive_float(
    data: dict[str, Any],
    key: str,
) -> float:
    """Return a required strictly positive numeric value."""
    if key not in data:
        raise ValueError(
            f"Missing required scenario field {key!r}."
        )

    value = data[
        key
    ]

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"Scenario field {key!r} must be numeric."
        )

    if not isinstance(
        value,
        int | float,
    ):
        raise ValueError(
            f"Scenario field {key!r} must be numeric."
        )

    numeric = float(
        value
    )

    if numeric <= 0:
        raise ValueError(
            f"Scenario field {key!r} must be greater than zero."
        )

    return numeric