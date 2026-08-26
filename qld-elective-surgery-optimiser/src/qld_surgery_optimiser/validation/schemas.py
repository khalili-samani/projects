"""Expected source schemas and schema-drift helpers."""

from __future__ import annotations

from collections.abc import Iterable

from qld_surgery_optimiser.config import ValidationConfig


def required_columns_for_kind(
    kind: str,
    config: ValidationConfig,
) -> set[str]:
    """Return required columns for a raw resource family."""
    common = set(config.required_common_columns)

    if kind == "category":
        return common | set(config.category_identity_columns)

    if kind == "specialty":
        return common | set(config.specialty_identity_columns)

    raise ValueError(f"Unsupported resource kind: {kind}")


def known_columns(
    config: ValidationConfig,
) -> set[str]:
    """Return the full set of recognised source columns."""
    return (
        set(config.required_common_columns)
        | set(config.category_identity_columns)
        | set(config.specialty_identity_columns)
        | set(config.numeric_volume_columns)
        | set(config.percentage_columns)
        | set(config.date_columns)
    )


def missing_required_columns(
    columns: Iterable[str],
    *,
    kind: str,
    config: ValidationConfig,
) -> list[str]:
    """Return required columns absent from the source."""
    present = set(columns)

    missing = required_columns_for_kind(
        kind,
        config,
    ) - present

    return sorted(missing)


def unexpected_columns(
    columns: Iterable[str],
    *,
    config: ValidationConfig,
) -> list[str]:
    """Return source columns not covered by the known schema."""
    present = set(columns)

    return sorted(
        present - known_columns(config)
    )