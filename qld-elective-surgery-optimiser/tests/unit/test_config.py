"""Tests for typed configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qld_surgery_optimiser.config import (
    AppSettings,
    create_required_directories,
    load_base_config,
    load_scenario_config,
)
from qld_surgery_optimiser.exceptions import ConfigurationError


def test_load_base_config_from_repository_file() -> None:
    """The committed base configuration should satisfy its schema."""
    config = load_base_config(Path("configs/base.yml"))

    assert config.project.name == "qld-elective-surgery-optimiser"
    assert config.project.patient_level_use_permitted is False
    assert config.storage.overwrite_raw_files is False
    assert config.validation.percentage_maximum == 100.0


def test_load_baseline_scenario_from_repository_file() -> None:
    """The baseline scenario should satisfy its schema."""
    scenario = load_scenario_config(Path("configs/scenarios/baseline.yml"))

    assert scenario.scenario.name == "baseline"
    assert scenario.scenario.incremental_sessions_available == 120
    assert scenario.capacity.default_patients_per_session == 3.0
    assert scenario.simulation.iterations == 1000


def test_scenario_rejects_invalid_cancellation_rate(
    temporary_project_root: Path,
) -> None:
    """Cancellation probabilities above one must be rejected."""
    scenario_path = temporary_project_root / "invalid_scenario.yml"

    payload = {
        "scenario": {
            "name": "invalid",
            "description": "Invalid test scenario.",
            "planning_periods": 1,
            "incremental_sessions_available": 10,
            "random_seed": 42,
        },
        "capacity": {
            "default_patients_per_session": 3.0,
            "default_cancellation_rate": 1.5,
            "emergency_displacement_rate": 0.05,
        },
        "demand": {
            "quarterly_growth_rate": 0.02,
            "uncertainty_standard_deviation": 0.05,
        },
        "policy": {
            "minimum_sessions_per_eligible_combination": 0,
            "maximum_share_per_facility": 0.20,
            "enforce_regional_coverage": True,
            "minimum_regions_served": 3,
        },
        "simulation": {
            "iterations": 100,
            "patients_per_session_standard_deviation": 0.4,
            "cancellation_rate_standard_deviation": 0.02,
            "demand_growth_standard_deviation": 0.03,
        },
    }

    scenario_path.write_text(
        yaml.safe_dump(payload),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_scenario_config(scenario_path)


def test_missing_configuration_file_raises_clear_error(
    temporary_project_root: Path,
) -> None:
    """A missing YAML file should raise the project configuration exception."""
    missing_path = temporary_project_root / "missing.yml"

    with pytest.raises(ConfigurationError, match="does not exist"):
        load_scenario_config(missing_path)


def test_create_required_directories(
    temporary_project_root: Path,
) -> None:
    """The bootstrap helper should create each configured local directory."""
    settings = AppSettings(
        data_dir=temporary_project_root / "data",
        raw_data_dir=temporary_project_root / "data/raw",
        interim_data_dir=temporary_project_root / "data/interim",
        processed_data_dir=temporary_project_root / "data/processed",
        quarantine_data_dir=temporary_project_root / "data/quarantine",
        reports_dir=temporary_project_root / "reports",
        duckdb_path=temporary_project_root / "data/processed/test.duckdb",
    )

    created = create_required_directories(settings)

    assert settings.raw_data_dir.exists()
    assert settings.processed_data_dir.exists()
    assert (settings.reports_dir / "figures").exists()
    assert (settings.reports_dir / "outputs").exists()
    assert settings.raw_data_dir in created


def test_create_required_directories_is_idempotent(
    temporary_project_root: Path,
) -> None:
    """Running directory creation twice should not report existing paths again."""
    settings = AppSettings(
        data_dir=temporary_project_root / "data",
        raw_data_dir=temporary_project_root / "data/raw",
        interim_data_dir=temporary_project_root / "data/interim",
        processed_data_dir=temporary_project_root / "data/processed",
        quarantine_data_dir=temporary_project_root / "data/quarantine",
        reports_dir=temporary_project_root / "reports",
        duckdb_path=temporary_project_root / "data/processed/test.duckdb",
    )

    first_created = create_required_directories(settings)
    second_created = create_required_directories(settings)

    assert first_created
    assert second_created == []