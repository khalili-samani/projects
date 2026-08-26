"""Command-line interface for project administration and pipeline execution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from qld_surgery_optimiser import __version__
from qld_surgery_optimiser.config import (
    create_required_directories,
    get_settings,
    load_base_config,
    load_scenario_config,
)
from qld_surgery_optimiser.exceptions import (
    ConfigurationError,
    DataValidationError,
    DownloadError,
    SourceDiscoveryError,
)
from qld_surgery_optimiser.ingestion.ckan_client import CkanClient
from qld_surgery_optimiser.ingestion.pipeline import run_ingestion
from qld_surgery_optimiser.logging_config import configure_logging
from qld_surgery_optimiser.validation.pipeline import run_validation


app = typer.Typer(
    name="qld-surgery",
    help="Queensland elective surgery capacity optimisation tools.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _serialise_path(value: object) -> object:
    """Convert paths to strings for JSON output."""
    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object is not JSON serialisable: "
        f"{type(value).__name__}"
    )


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Display the installed package version.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Run project administration and pipeline commands."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("doctor")
def doctor() -> None:
    """Validate local configuration and create required directories."""
    settings = get_settings()

    configure_logging(
        settings.log_level,
        json_output=True,
    )

    try:
        base_config = load_base_config(
            settings.base_config_path
        )

        scenario = load_scenario_config(
            settings.default_scenario_path
        )

        created_directories = create_required_directories(
            settings
        )

    except ConfigurationError as exc:
        logger.error(
            "Configuration health check failed",
            extra={
                "error": str(exc),
            },
        )

        typer.echo(
            f"Configuration error: {exc}",
            err=True,
        )

        raise typer.Exit(code=1) from exc

    logger.info(
        "Configuration health check passed",
        extra={
            "project": base_config.project.name,
            "scenario": scenario.scenario.name,
            "created_directories": [
                str(path)
                for path in created_directories
            ],
        },
    )

    typer.echo(
        "Configuration health check passed."
    )

    typer.echo(
        f"Project: {base_config.project.name}"
    )

    typer.echo(
        f"Default scenario: "
        f"{scenario.scenario.name}"
    )

    if created_directories:
        typer.echo(
            "Created directories:"
        )

        for path in created_directories:
            typer.echo(
                f"  - {path}"
            )
    else:
        typer.echo(
            "All required directories already exist."
        )


@app.command("show-config")
def show_config(
    scenario_path: Annotated[
        Path | None,
        typer.Option(
            "--scenario",
            help="Optional scenario YAML file to display.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """Display resolved non-secret application configuration."""
    settings = get_settings()

    configure_logging(
        settings.log_level,
        json_output=True,
    )

    selected_scenario = (
        scenario_path
        or settings.default_scenario_path
    )

    try:
        base_config = load_base_config(
            settings.base_config_path
        )

        scenario = load_scenario_config(
            selected_scenario
        )

    except ConfigurationError as exc:
        logger.error(
            "Configuration display failed",
            extra={
                "error": str(exc),
            },
        )

        typer.echo(
            f"Configuration error: {exc}",
            err=True,
        )

        raise typer.Exit(code=1) from exc

    payload = {
        "settings": settings.model_dump(
            mode="python"
        ),
        "base_config": base_config.model_dump(
            mode="python"
        ),
        "scenario": scenario.model_dump(
            mode="python"
        ),
    }

    typer.echo(
        json.dumps(
            payload,
            indent=2,
            default=_serialise_path,
            ensure_ascii=False,
        )
    )


@app.command("discover")
def discover() -> None:
    """Discover eligible upstream elective-surgery resources."""
    settings = get_settings()

    configure_logging(
        settings.log_level,
        json_output=True,
    )

    try:
        base_config = load_base_config(
            settings.base_config_path
        )

        source_config = (
            base_config.sources.queensland_open_data
        )

        with CkanClient(
            source_config,
            timeout_seconds=(
                settings.request_timeout_seconds
            ),
            max_retries=(
                settings.request_max_retries
            ),
            retry_backoff_seconds=(
                settings.request_retry_backoff_seconds
            ),
            user_agent=settings.user_agent,
        ) as client:
            dataset, resources = client.get_dataset()

    except (
        ConfigurationError,
        SourceDiscoveryError,
    ) as exc:
        logger.exception(
            "Source discovery failed",
            extra={
                "error": str(exc),
            },
        )

        typer.echo(
            f"Discovery failed: {exc}",
            err=True,
        )

        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Dataset: {dataset.title}"
    )

    typer.echo(
        f"Eligible resources: "
        f"{len(resources)}"
    )

    if dataset.organisation:
        typer.echo(
            f"Organisation: "
            f"{dataset.organisation}"
        )

    if dataset.licence_title:
        typer.echo(
            f"Licence: "
            f"{dataset.licence_title}"
        )

    typer.echo("")

    for resource in resources:
        typer.echo(
            f"[{resource.resource_kind}] "
            f"{resource.name} "
            f"({resource.resource_id})"
        )


@app.command("ingest")
def ingest(
    latest_only: Annotated[
        bool,
        typer.Option(
            "--latest-only",
            help=(
                "Retrieve only the most recent "
                "Category and Speciality resources."
            ),
        ),
    ] = False,
) -> None:
    """Download raw source resources and update the lineage manifest."""
    settings = get_settings()

    configure_logging(
        settings.log_level,
        json_output=True,
    )

    try:
        create_required_directories(
            settings
        )

        base_config = load_base_config(
            settings.base_config_path
        )

        summary = run_ingestion(
            settings=settings,
            base_config=base_config,
            latest_only=latest_only,
        )

    except (
        ConfigurationError,
        SourceDiscoveryError,
        DownloadError,
    ) as exc:
        logger.exception(
            "Ingestion failed",
            extra={
                "error": str(exc),
            },
        )

        typer.echo(
            f"Ingestion failed: {exc}",
            err=True,
        )

        raise typer.Exit(code=1) from exc

    typer.echo(
        "Ingestion completed successfully."
    )

    typer.echo(
        f"Dataset: "
        f"{summary.dataset_id}"
    )

    typer.echo(
        f"Resources discovered: "
        f"{summary.resources_discovered}"
    )

    typer.echo(
        f"Resources selected: "
        f"{summary.resources_selected}"
    )

    typer.echo(
        f"New raw files: "
        f"{summary.resources_downloaded}"
    )

    typer.echo(
        f"Already present: "
        f"{summary.resources_already_present}"
    )

    typer.echo(
        f"Manifest records added: "
        f"{summary.manifest_records_added}"
    )

    typer.echo(
        f"Manifest: "
        f"{summary.manifest_path}"
    )


@app.command("validate")
def validate() -> None:
    """Validate all retrieved raw elective-surgery resources."""
    settings = get_settings()

    configure_logging(
        settings.log_level,
        json_output=True,
    )

    try:
        create_required_directories(
            settings
        )

        base_config = load_base_config(
            settings.base_config_path
        )

        summary = run_validation(
            settings=settings,
            base_config=base_config,
        )

    except (
        ConfigurationError,
        DataValidationError,
    ) as exc:
        logger.exception(
            "Validation failed",
            extra={
                "error": str(exc),
            },
        )

        typer.echo(
            f"Validation failed: {exc}",
            err=True,
        )

        raise typer.Exit(code=1) from exc

    typer.echo(
        "Validation completed."
    )

    typer.echo(
        f"Files processed: "
        f"{summary.files_processed}"
    )

    typer.echo(
        f"Files passed: "
        f"{summary.files_passed}"
    )

    typer.echo(
        f"Files failed: "
        f"{summary.files_failed}"
    )

    typer.echo(
        f"Rows read: "
        f"{summary.rows_read}"
    )

    typer.echo(
        f"Rows valid: "
        f"{summary.rows_valid}"
    )

    typer.echo(
        f"Rows quarantined: "
        f"{summary.rows_quarantined}"
    )

    typer.echo(
        f"Validated data directory: "
        f"{summary.interim_directory}"
    )

    typer.echo(
        f"Quarantine directory: "
        f"{summary.quarantine_directory}"
    )

    typer.echo(
        f"Quality report: "
        f"{summary.report_path}"
    )


if __name__ == "__main__":
    app()