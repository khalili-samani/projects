"""Canonical processing and DuckDB warehouse construction."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from qld_surgery_optimiser.config import AppSettings
from qld_surgery_optimiser.exceptions import (
    DataValidationError,
)
from qld_surgery_optimiser.processing.entities import (
    count_unresolved_facilities,
    load_facility_aliases,
    resolve_facilities,
)
from qld_surgery_optimiser.processing.longitudinal import (
    build_longitudinal_frame,
)
from qld_surgery_optimiser.processing.models import (
    WarehouseBuildSummary,
)
from qld_surgery_optimiser.processing.normalise import (
    infer_resource_kind,
    normalise_validated_frame,
)

logger = logging.getLogger(__name__)


def _validated_files(
    interim_directory: Path,
) -> list[Path]:
    """Return validated Phase 3 Parquet files."""
    return sorted(
        interim_directory.rglob(
            "*_validated.parquet"
        )
    )


def _read_manifest(
    raw_data_directory: Path,
) -> pd.DataFrame:
    """Read Phase 2 raw-resource manifest."""
    manifest_path = (
        raw_data_directory
        / "manifest.csv"
    )

    if not manifest_path.exists():
        raise DataValidationError(
            "Raw source manifest does not exist. "
            "Run ingestion before building the warehouse."
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "resource_id",
        "source_url",
        "retrieved_at",
        "local_path",
        "sha256",
    }

    missing = required - set(
        manifest.columns
    )

    if missing:
        raise DataValidationError(
            "Raw manifest is missing required fields: "
            + ", ".join(sorted(missing))
        )

    manifest["_raw_stem"] = manifest[
        "local_path"
    ].map(
        lambda value: Path(value).stem
    )

    return manifest


def _metadata_for_validated_file(
    path: Path,
    *,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Resolve source metadata for a validated Phase 3 file."""
    validated_suffix = "_validated"

    if not path.stem.endswith(
        validated_suffix
    ):
        raise DataValidationError(
            f"Unexpected validated filename: {path}"
        )

    raw_stem = path.stem[
        : -len(validated_suffix)
    ]

    matches = manifest.loc[
        manifest["_raw_stem"].eq(
            raw_stem
        )
    ]

    if matches.empty:
        raise DataValidationError(
            "Could not map validated file to raw "
            f"manifest entry: {path}"
        )

    if len(matches) > 1:
        unique_hashes = matches[
            "sha256"
        ].nunique()

        if unique_hashes > 1:
            raise DataValidationError(
                "Validated file matched multiple "
                "different source hashes: "
                f"{path}"
            )

    row = matches.iloc[-1]

    return {
        "resource_id": row["resource_id"],
        "sha256": row["sha256"],
        "source_url": row["source_url"],
        "retrieved_at": row["retrieved_at"],
        "source_file": raw_stem,
    }


def _load_canonical_data(
    *,
    settings: AppSettings,
) -> pd.DataFrame:
    """Load and canonicalise all validated source files."""
    files = _validated_files(
        settings.interim_data_dir
    )

    if not files:
        raise DataValidationError(
            "No validated Parquet files found. "
            "Run validation before building the warehouse."
        )

    manifest = _read_manifest(
        settings.raw_data_dir
    )

    frames: list[pd.DataFrame] = []

    for path in files:
        resource_kind = infer_resource_kind(
            path
        )

        source_metadata = (
            _metadata_for_validated_file(
                path,
                manifest=manifest,
            )
        )

        frame = pd.read_parquet(
            path
        )

        canonical = normalise_validated_frame(
            frame,
            resource_kind=resource_kind,
            source_metadata=source_metadata,
        )

        frames.append(canonical)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _hash_key(
    *values: object,
) -> str:
    """Create stable dimension identifiers."""
    payload = "|".join(
        ""
        if value is None or pd.isna(value)
        else str(value)
        for value in values
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _build_dimensions(
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build dimensional warehouse tables."""
    facility = (
        frame[
            [
                "canonical_facility_code",
                "canonical_facility_name",
                "hhs",
                "region",
                "facility_resolution_status",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    facility["facility_key"] = facility.apply(
        lambda row: _hash_key(
            row["canonical_facility_code"],
            row["canonical_facility_name"],
        ),
        axis=1,
    )

    facility = facility.rename(
        columns={
            "canonical_facility_code": (
                "facility_code"
            ),
            "canonical_facility_name": (
                "facility_name"
            ),
            "facility_resolution_status": (
                "resolution_status"
            ),
        }
    )

    specialty = (
        frame.loc[
            frame["resource_kind"].eq(
                "specialty"
            ),
            [
                "service_code",
                "service_name",
            ],
        ]
        .drop_duplicates()
        .copy()
    )

    specialty["specialty_key"] = (
        specialty.apply(
            lambda row: _hash_key(
                row["service_code"],
                row["service_name"],
            ),
            axis=1,
        )
    )

    specialty = specialty.rename(
        columns={
            "service_code": "specialty_code",
            "service_name": "specialty_name",
        }
    )

    urgency = (
        frame.loc[
            frame["resource_kind"].eq(
                "category"
            ),
            ["service_name"],
        ]
        .drop_duplicates()
        .copy()
    )

    urgency[
        "urgency_category_key"
    ] = urgency["service_name"].map(
        _hash_key
    )

    urgency = urgency.rename(
        columns={
            "service_name": (
                "urgency_category_name"
            )
        }
    )

    reporting = (
        frame[
            ["report_month"]
        ]
        .drop_duplicates()
        .dropna()
        .copy()
    )

    reporting[
        "reporting_period_key"
    ] = pd.to_datetime(
        reporting["report_month"]
    ).dt.date

    report_timestamp = pd.to_datetime(
        reporting["report_month"]
    )

    reporting["calendar_year"] = (
        report_timestamp.dt.year
    )

    reporting["calendar_quarter"] = (
        report_timestamp.dt.quarter
    )

    reporting["month"] = (
        report_timestamp.dt.month
    )

    reporting["quarter_label"] = (
        reporting["calendar_year"].astype(
            str
        )
        + "-Q"
        + reporting[
            "calendar_quarter"
        ].astype(str)
    )

    reporting = reporting.drop(
        columns=["report_month"]
    )

    source = (
        frame[
            [
                "source_resource_id",
                "source_sha256",
                "source_url",
                "source_file",
                "source_retrieved_at",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    source[
        "source_resource_key"
    ] = source.apply(
        lambda row: _hash_key(
            row["source_resource_id"],
            row["source_sha256"],
        ),
        axis=1,
    )

    source = source.rename(
        columns={
            "source_resource_id": (
                "resource_id"
            ),
            "source_retrieved_at": (
                "retrieved_at"
            ),
        }
    )

    return {
        "dim_facility": facility,
        "dim_specialty": specialty,
        "dim_urgency_category": urgency,
        "dim_reporting_period": reporting,
        "dim_source_resource": source,
    }


def _build_fact(
    frame: pd.DataFrame,
    dimensions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create the canonical elective-surgery performance fact."""
    fact = frame.copy()

    facilities = dimensions[
        "dim_facility"
    ]

    fact = fact.merge(
        facilities[
            [
                "facility_key",
                "facility_code",
                "facility_name",
            ]
        ],
        left_on=[
            "canonical_facility_code",
            "canonical_facility_name",
        ],
        right_on=[
            "facility_code",
            "facility_name",
        ],
        how="left",
        validate="many_to_one",
    )

    sources = dimensions[
        "dim_source_resource"
    ]

    fact = fact.merge(
        sources[
            [
                "source_resource_key",
                "resource_id",
                "source_sha256",
            ]
        ],
        left_on=[
            "source_resource_id",
            "source_sha256",
        ],
        right_on=[
            "resource_id",
            "source_sha256",
        ],
        how="left",
        validate="many_to_one",
    )

    specialty_lookup = dimensions[
        "dim_specialty"
    ].rename(
        columns={
            "specialty_code": (
                "service_code"
            ),
            "specialty_name": (
                "service_name"
            ),
        }
    )

    fact = fact.merge(
        specialty_lookup[
            [
                "specialty_key",
                "service_code",
                "service_name",
            ]
        ],
        on=[
            "service_code",
            "service_name",
        ],
        how="left",
    )

    urgency_lookup = dimensions[
        "dim_urgency_category"
    ].rename(
        columns={
            "urgency_category_name": (
                "service_name"
            ),
        }
    )

    urgency_lookup = urgency_lookup[
        [
            "urgency_category_key",
            "service_name",
        ]
    ]

    category_mask = fact[
        "resource_kind"
    ].eq("category")

    category_keys = (
        fact.loc[
            category_mask,
            ["service_name"],
        ]
        .merge(
            urgency_lookup,
            on="service_name",
            how="left",
        )[
            "urgency_category_key"
        ]
        .to_numpy()
    )

    fact[
        "urgency_category_key"
    ] = pd.NA

    fact.loc[
        category_mask,
        "urgency_category_key",
    ] = category_keys

    fact.loc[
        category_mask,
        "specialty_key",
    ] = pd.NA

    fact[
        "reporting_period_key"
    ] = pd.to_datetime(
        fact["report_month"]
    ).dt.date

    output_columns = [
        "record_id",
        "facility_key",
        "reporting_period_key",
        "resource_kind",
        "specialty_key",
        "urgency_category_key",
        "vol_treated",
        "pct_treated_in_time",
        "pct_variation_treated_prior_year",
        "vol_waiting",
        "vol_long_waits",
        "pct_waiting_in_time_total",
        "vol_long_waits_rfs",
        "vol_long_waits_nrfs",
        "pct_waiting_in_time_rfs",
        "previous_vol_waiting",
        "backlog_change",
        "previous_vol_long_waits",
        "long_wait_change",
        "long_wait_share",
        "treatment_to_waiting_ratio",
        "data_last_update",
        "source_resource_key",
    ]

    return fact[
        output_columns
    ].copy()


def _load_quality_events(
    report_path: Path,
) -> pd.DataFrame:
    """Convert Phase 3 quality report issues into a warehouse fact."""
    columns = [
        "event_id",
        "source_path",
        "resource_kind",
        "rule_id",
        "severity",
        "row_index",
        "column_name",
        "observed_value",
        "message",
    ]

    if not report_path.exists():
        return pd.DataFrame(
            columns=columns
        )

    payload = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    records: list[
        dict[str, object]
    ] = []

    for file_result in payload.get(
        "files",
        [],
    ):
        for issue in file_result.get(
            "issues",
            [],
        ):
            event_payload = "|".join(
                [
                    str(
                        file_result.get(
                            "source_path",
                            "",
                        )
                    ),
                    str(
                        issue.get(
                            "rule_id",
                            "",
                        )
                    ),
                    str(
                        issue.get(
                            "row_index",
                            "",
                        )
                    ),
                    str(
                        issue.get(
                            "column",
                            "",
                        )
                    ),
                    str(
                        issue.get(
                            "message",
                            "",
                        )
                    ),
                ]
            )

            event_id = hashlib.sha256(
                event_payload.encode(
                    "utf-8"
                )
            ).hexdigest()

            records.append(
                {
                    "event_id": event_id,
                    "source_path": (
                        file_result.get(
                            "source_path"
                        )
                    ),
                    "resource_kind": (
                        file_result.get(
                            "resource_kind"
                        )
                    ),
                    "rule_id": issue.get(
                        "rule_id"
                    ),
                    "severity": issue.get(
                        "severity"
                    ),
                    "row_index": issue.get(
                        "row_index"
                    ),
                    "column_name": issue.get(
                        "column"
                    ),
                    "observed_value": issue.get(
                        "observed_value"
                    ),
                    "message": issue.get(
                        "message"
                    ),
                }
            )

    return pd.DataFrame(
        records,
        columns=columns,
    )


def _replace_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Replace warehouse table contents with one dataframe."""
    view_name = (
        f"_load_{table_name}"
    )

    connection.register(
        view_name,
        dataframe,
    )

    try:
        connection.execute(
            f"DELETE FROM {table_name}"
        )

        if not dataframe.empty:
            connection.execute(
                f"""
                INSERT INTO {table_name}
                SELECT * FROM {view_name}
                """
            )
    finally:
        connection.unregister(
            view_name
        )


def _write_reconciliation_report(
    *,
    canonical_rows: int,
    longitudinal_rows: int,
    fact_rows: int,
    duplicate_rows_removed: int,
    path: Path,
) -> None:
    """Persist warehouse reconciliation evidence."""
    report = {
        "canonical_rows": canonical_rows,
        "longitudinal_rows": longitudinal_rows,
        "fact_rows": fact_rows,
        "duplicate_canonical_keys_removed": (
            duplicate_rows_removed
        ),
        "fact_matches_longitudinal": (
            fact_rows
            == longitudinal_rows
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_warehouse(
    *,
    settings: AppSettings,
) -> WarehouseBuildSummary:
    """Build canonical Parquet outputs and the DuckDB warehouse."""
    canonical = _load_canonical_data(
        settings=settings
    )

    aliases = load_facility_aliases(
        settings.facility_aliases_path
    )

    resolved = resolve_facilities(
        canonical,
        aliases=aliases,
    )

    unresolved_count = (
        count_unresolved_facilities(
            resolved
        )
    )

    longitudinal, duplicates_removed = (
        build_longitudinal_frame(
            resolved
        )
    )

    settings.processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_path = (
        settings.processed_data_dir
        / "canonical_performance.parquet"
    )

    longitudinal_path = (
        settings.processed_data_dir
        / "longitudinal_performance.parquet"
    )

    resolved.to_parquet(
        canonical_path,
        index=False,
    )

    longitudinal.to_parquet(
        longitudinal_path,
        index=False,
    )

    dimensions = _build_dimensions(
        longitudinal
    )

    fact = _build_fact(
        longitudinal,
        dimensions,
    )

    quality_events = (
        _load_quality_events(
            settings.reports_dir
            / "outputs"
            / "data_quality_summary.json"
        )
    )

    ddl_path = Path(
        "sql/create_warehouse.sql"
    )

    if not ddl_path.exists():
        raise DataValidationError(
            f"Warehouse DDL not found: {ddl_path}"
        )

    settings.duckdb_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with duckdb.connect(
        str(settings.duckdb_path)
    ) as connection:
        connection.execute(
            ddl_path.read_text(
                encoding="utf-8"
            )
        )

        for table_name, dataframe in (
            dimensions.items()
        ):
            _replace_table(
                connection,
                table_name=table_name,
                dataframe=dataframe,
            )

        _replace_table(
            connection,
            table_name=(
                "fact_elective_surgery_performance"
            ),
            dataframe=fact,
        )

        _replace_table(
            connection,
            table_name=(
                "fact_data_quality_event"
            ),
            dataframe=quality_events,
        )

    reconciliation_path = (
        settings.reports_dir
        / "outputs"
        / "warehouse_reconciliation.json"
    )

    _write_reconciliation_report(
        canonical_rows=len(resolved),
        longitudinal_rows=len(
            longitudinal
        ),
        fact_rows=len(fact),
        duplicate_rows_removed=(
            duplicates_removed
        ),
        path=reconciliation_path,
    )

    summary = WarehouseBuildSummary(
        validated_files=len(
            _validated_files(
                settings.interim_data_dir
            )
        ),
        canonical_rows=len(resolved),
        longitudinal_rows=len(
            longitudinal
        ),
        facility_count=len(
            dimensions["dim_facility"]
        ),
        specialty_count=len(
            dimensions["dim_specialty"]
        ),
        urgency_category_count=len(
            dimensions[
                "dim_urgency_category"
            ]
        ),
        reporting_period_count=len(
            dimensions[
                "dim_reporting_period"
            ]
        ),
        unresolved_facilities=(
            unresolved_count
        ),
        duplicate_canonical_keys_removed=(
            duplicates_removed
        ),
        database_path=(
            settings.duckdb_path
        ),
        canonical_parquet_path=(
            canonical_path
        ),
        longitudinal_parquet_path=(
            longitudinal_path
        ),
        reconciliation_report_path=(
            reconciliation_path
        ),
    )

    logger.info(
        "Warehouse build completed",
        extra={
            "canonical_rows": (
                summary.canonical_rows
            ),
            "longitudinal_rows": (
                summary.longitudinal_rows
            ),
            "facility_count": (
                summary.facility_count
            ),
            "specialty_count": (
                summary.specialty_count
            ),
            "urgency_category_count": (
                summary.urgency_category_count
            ),
            "unresolved_facilities": (
                summary.unresolved_facilities
            ),
            "database_path": str(
                summary.database_path
            ),
        },
    )

    return summary