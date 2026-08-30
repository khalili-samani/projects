"""Typed configuration loading and validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from qld_surgery_optimiser.exceptions import ConfigurationError


class ProjectConfig(BaseModel):
    """Project identity and approved analytical scope."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    geography: str
    decision_scope: str
    patient_level_use_permitted: bool = False


class QueenslandOpenDataConfig(BaseModel):
    """Queensland Government Open Data CKAN configuration."""

    model_config = ConfigDict(extra="forbid")

    organisation: str
    api_base_url: str
    dataset_id: str

    allowed_formats: list[str] = Field(min_length=1)

    category_patterns: list[str] = Field(min_length=1)
    specialty_patterns: list[str] = Field(min_length=1)

    include_historical_resources: bool = True


class SourcesConfig(BaseModel):
    """External source configuration."""

    model_config = ConfigDict(extra="forbid")

    queensland_open_data: QueenslandOpenDataConfig


class StorageConfig(BaseModel):
    """Raw and processed storage behaviour."""

    model_config = ConfigDict(extra="forbid")

    raw_format: str
    processed_format: str
    preserve_raw_files: bool
    calculate_sha256: bool
    overwrite_raw_files: bool


class ValidationConfig(BaseModel):
    """Global data-quality policy."""

    model_config = ConfigDict(extra="forbid")

    fail_on_missing_required_columns: bool
    fail_on_duplicate_business_keys: bool
    quarantine_invalid_rows: bool
    allow_unexpected_columns: bool

    maximum_unresolved_entity_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    percentage_minimum: float
    percentage_maximum: float

    null_tokens: list[str] = Field(min_length=1)

    required_common_columns: list[str] = Field(min_length=1)
    category_identity_columns: list[str] = Field(min_length=1)
    specialty_identity_columns: list[str] = Field(min_length=1)

    numeric_volume_columns: list[str] = Field(min_length=1)
    percentage_columns: list[str] = Field(min_length=1)
    date_columns: list[str] = Field(min_length=1)


class WarehouseConfig(BaseModel):
    """Analytical warehouse configuration."""

    model_config = ConfigDict(extra="forbid")

    database_schema: str
    replace_derived_tables: bool
    preserve_run_metadata: bool


class ReportingConfig(BaseModel):
    """Required recommendation metadata and disclosures."""

    model_config = ConfigDict(extra="forbid")

    include_data_freshness: bool
    include_scenario_provenance: bool
    include_solver_status: bool
    include_limitations: bool
    include_responsible_use_notice: bool


class BaseConfig(BaseModel):
    """Complete base YAML configuration."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    sources: SourcesConfig
    storage: StorageConfig
    validation: ValidationConfig
    warehouse: WarehouseConfig
    reporting: ReportingConfig


class ScenarioMetadata(BaseModel):
    """Scenario identity and execution settings."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    planning_periods: PositiveInt
    incremental_sessions_available: int = Field(ge=0)
    random_seed: int = Field(ge=0)


class CapacityScenario(BaseModel):
    """Capacity assumptions used by optimisation and simulation."""

    model_config = ConfigDict(extra="forbid")

    default_patients_per_session: PositiveFloat
    default_cancellation_rate: float = Field(ge=0.0, le=1.0)
    emergency_displacement_rate: float = Field(ge=0.0, le=1.0)


class DemandScenario(BaseModel):
    """Incoming demand assumptions."""

    model_config = ConfigDict(extra="forbid")

    quarterly_growth_rate: float = Field(gt=-1.0)
    uncertainty_standard_deviation: float = Field(ge=0.0)


class PolicyScenario(BaseModel):
    """Allocation-policy constraints."""

    model_config = ConfigDict(extra="forbid")

    minimum_sessions_per_eligible_combination: int = Field(ge=0)
    maximum_share_per_facility: float = Field(gt=0.0, le=1.0)
    enforce_regional_coverage: bool
    minimum_regions_served: int = Field(ge=0)


class SimulationScenario(BaseModel):
    """Monte Carlo configuration."""

    model_config = ConfigDict(extra="forbid")

    iterations: PositiveInt
    patients_per_session_standard_deviation: float = Field(ge=0.0)
    cancellation_rate_standard_deviation: float = Field(ge=0.0)
    demand_growth_standard_deviation: float = Field(ge=0.0)


class ScenarioConfig(BaseModel):
    """Complete scenario configuration."""

    model_config = ConfigDict(extra="forbid")

    scenario: ScenarioMetadata
    capacity: CapacityScenario
    demand: DemandScenario
    policy: PolicyScenario
    simulation: SimulationScenario


class AppSettings(BaseSettings):
    """Environment-based application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"

    base_config_path: Path = Path("configs/base.yml")
    optimisation_config_path: Path = Path("configs/optimisation.yml")
    default_scenario_path: Path = Path("configs/scenarios/baseline.yml")
    facility_aliases_path: Path = Path(
    "data/reference/facility_aliases.csv"
    )

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    interim_data_dir: Path = Path("data/interim")
    processed_data_dir: Path = Path("data/processed")
    quarantine_data_dir: Path = Path("data/quarantine")
    reports_dir: Path = Path("reports")
    duckdb_path: Path = Path("data/processed/elective_surgery.duckdb")

    request_timeout_seconds: PositiveInt = 30
    request_max_retries: int = Field(default=3, ge=0)
    request_retry_backoff_seconds: float = Field(default=1.0, ge=0.0)
    user_agent: str = "qld-elective-surgery-optimiser/0.1.0"

    random_seed: int = Field(default=42, ge=0)

    solver_time_limit_seconds: PositiveInt = 60
    solver_num_workers: PositiveInt = 1

    @property
    def required_directories(self) -> tuple[Path, ...]:
        """Directories required for normal local execution."""
        return (
            self.data_dir,
            self.raw_data_dir,
            self.interim_data_dir,
            self.processed_data_dir,
            self.quarantine_data_dir,
            self.reports_dir,
            self.reports_dir / "figures",
            self.reports_dir / "outputs",
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its root mapping."""
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            content = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read configuration file: {path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {path}"
        ) from exc

    if not isinstance(content, dict):
        raise ConfigurationError(
            f"Configuration root must be a mapping: {path}"
        )

    return content


def load_base_config(path: Path) -> BaseConfig:
    """Load and validate the project base configuration."""
    try:
        return BaseConfig.model_validate(_read_yaml(path))
    except ValueError as exc:
        raise ConfigurationError(
            f"Base configuration failed validation: {path}"
        ) from exc


def load_scenario_config(path: Path) -> ScenarioConfig:
    """Load and validate a planning scenario."""
    try:
        return ScenarioConfig.model_validate(_read_yaml(path))
    except ValueError as exc:
        raise ConfigurationError(
            f"Scenario configuration failed validation: {path}"
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""
    return AppSettings()


def create_required_directories(
    settings: AppSettings,
) -> list[Path]:
    """Create missing local execution directories."""
    created: list[Path] = []

    for directory in settings.required_directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)

    return created