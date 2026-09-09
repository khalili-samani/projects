"""Build the Phase 4 analytical DuckDB warehouse.

This module consumes validated Parquet outputs from the validation layer,
normalises source records into a canonical analytical representation,
constructs warehouse dimensions and facts, persists processed Parquet
artifacts, loads DuckDB tables, and writes reconciliation metadata.

The warehouse operates only on aggregate elective-surgery information.
It does not perform patient-level decision making.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from qld_surgery_optimiser.config import AppSettings


WAREHOUSE_TABLES = frozenset(
    {
        "dim_facility",
        "dim_specialty",
        "dim_urgency_category",
        "dim_reporting_period",
        "dim_source_resource",
        "fact_elective_surgery_performance",
        "fact_data_quality_event",
    }
)


COLUMN_ALIASES: dict[str, str] = {
    "Facility_Code": "facility_code",
    "Facility_Name": "facility_name",
    "Report_Month": "report_month",
    "Specialty_Code": "service_code",
    "Specialty_Desc": "service_name",
    "Specialty": "service_name",
    "Category": "service_name",
    "Vol_Treated": "vol_treated",
    "Percent_Treated_InTime": "pct_treated_in_time",
    "Percent_Variation_Treated_Prior_Year": (
        "pct_variation_treated_prior_year"
    ),
    "Vol_Waiting": "vol_waiting",
    "Vol_LongWaits": "vol_long_waits",
    "Percent_Waiting_InTime_Total": (
        "pct_waiting_in_time_total"
    ),
    "data_last_update": "data_last_update",
    "Vol_LongWaits_RFS": "vol_long_waits_rfs",
    "Vol_LongWaits_NRFS": "vol_long_waits_nrfs",
    "Percent_Waiting_InTime_RFS": (
        "pct_waiting_in_time_rfs"
    ),
}


CANONICAL_COLUMNS = [
    "record_id",
    "resource_kind",
    "source_resource_id",
    "source_sha256",
    "source_url",
    "source_retrieved_at",
    "source_file",
    "facility_code",
    "facility_name",
    "report_month",
    "service_code",
    "service_name",
    "vol_treated",
    "pct_treated_in_time",
    "pct_variation_treated_prior_year",
    "vol_waiting",
    "vol_long_waits",
    "pct_waiting_in_time_total",
    "data_last_update",
    "vol_long_waits_rfs",
    "vol_long_waits_nrfs",
    "pct_waiting_in_time_rfs",
]


@dataclass(frozen=True)
class WarehouseBuildSummary:
    """Summary returned after a warehouse build."""

    validated_files: int
    canonical_rows: int
    longitudinal_rows: int
    facility_count: int
    specialty_count: int
    urgency_category_count: int
    reporting_period_count: int
    source_resource_count: int
    quality_event_count: int
    duplicate_canonical_keys_removed: int
    duckdb_path: Path
    canonical_path: Path
    longitudinal_path: Path
    reconciliation_report_path: Path


def _stable_key(
    *values: object,
) -> str:
    """Return a deterministic SHA-256 key for business values."""

    serialised = "|".join(
        ""
        if value is None or pd.isna(value)
        else str(value).strip()
        for value in values
    )

    return hashlib.sha256(
        serialised.encode("utf-8")
    ).hexdigest()


def _normalise_text(
    value: object,
) -> str | None:
    """Normalise a nullable text value."""

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    return text or None


def _normalise_code(
    value: object,
) -> str | None:
    """Normalise a nullable code value."""

    text = _normalise_text(value)

    if text is None:
        return None

    if text.endswith(".0"):
        numeric = text[:-2]

        if numeric.isdigit():
            return numeric

    return text


def _infer_resource_kind(
    path: Path,
) -> str:
    """Infer the resource family from its validated path."""

    parts = {
        part.casefold()
        for part in path.parts
    }

    if "specialty" in parts:
        return "specialty"

    if "category" in parts:
        return "category"

    filename = path.name.casefold()

    if "special" in filename:
        return "specialty"

    if "categor" in filename:
        return "category"

    return "unknown"


def _raw_stem_from_validated_path(
    path: Path,
) -> str:
    """Return the raw file stem represented by a validated file."""

    suffix = "_validated"
    stem = path.stem

    if stem.endswith(suffix):
        return stem[:-len(suffix)]

    return stem


def _find_validated_files(
    interim_data_dir: Path,
) -> list[Path]:
    """Return validated Parquet files in deterministic order."""

    if not interim_data_dir.exists():
        return []

    return sorted(
        path
        for path in interim_data_dir.rglob(
            "*_validated.parquet"
        )
        if path.is_file()
    )


def _read_manifest(
    raw_data_dir: Path,
) -> pd.DataFrame:
    """Read the ingestion manifest."""

    manifest_path = (
        raw_data_dir
        / "manifest.csv"
    )

    columns = [
        "resource_id",
        "source_url",
        "retrieved_at",
        "local_path",
        "sha256",
    ]

    if not manifest_path.exists():
        return pd.DataFrame(
            columns=columns
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
    )

    for column in columns:
        if column not in manifest.columns:
            manifest[column] = pd.NA

    manifest = manifest[
        columns
    ].copy()

    manifest["retrieved_at"] = pd.to_datetime(
        manifest["retrieved_at"],
        errors="coerce",
        utc=True,
    )

    return manifest


def _manifest_row_for_validated_file(
    validated_path: Path,
    manifest: pd.DataFrame,
) -> dict[str, object]:
    """Resolve manifest lineage for a validated Parquet file."""

    if manifest.empty:
        return {}

    raw_stem = (
        _raw_stem_from_validated_path(
            validated_path
        )
    )

    matches: list[int] = []

    for index, value in manifest[
        "local_path"
    ].items():
        if pd.isna(value):
            continue

        if Path(str(value)).stem == raw_stem:
            matches.append(index)

    if not matches:
        return {}

    return manifest.loc[
        matches[-1]
    ].to_dict()


def _canonicalise_frame(
    *,
    source: pd.DataFrame,
    validated_path: Path,
    resource_kind: str,
    lineage: dict[str, object],
) -> pd.DataFrame:
    """Convert one validated source frame to canonical columns."""

    frame = source.rename(
        columns={
            source_name: canonical_name
            for source_name, canonical_name
            in COLUMN_ALIASES.items()
            if source_name in source.columns
        }
    ).copy()

    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["resource_kind"] = resource_kind
    frame["source_resource_id"] = lineage.get(
        "resource_id"
    )
    frame["source_sha256"] = lineage.get(
        "sha256"
    )
    frame["source_url"] = lineage.get(
        "source_url"
    )
    frame["source_retrieved_at"] = lineage.get(
        "retrieved_at"
    )
    frame["source_file"] = str(
        validated_path
    )

    frame["facility_code"] = (
        frame["facility_code"]
        .map(_normalise_code)
    )

    frame["facility_name"] = (
        frame["facility_name"]
        .map(_normalise_text)
    )

    frame["service_code"] = (
        frame["service_code"]
        .map(_normalise_code)
    )

    frame["service_name"] = (
        frame["service_name"]
        .map(_normalise_text)
    )

    frame["report_month"] = pd.to_datetime(
        frame["report_month"],
        errors="coerce",
    )

    frame["data_last_update"] = pd.to_datetime(
        frame["data_last_update"],
        errors="coerce",
    )

    numeric_columns = [
        "vol_treated",
        "pct_treated_in_time",
        "pct_variation_treated_prior_year",
        "vol_waiting",
        "vol_long_waits",
        "pct_waiting_in_time_total",
        "vol_long_waits_rfs",
        "vol_long_waits_nrfs",
        "pct_waiting_in_time_rfs",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["record_id"] = [
        _stable_key(
            resource_kind,
            facility_code,
            facility_name,
            report_month,
            service_code,
            service_name,
            lineage.get("resource_id"),
            row_index,
        )
        for row_index, (
            facility_code,
            facility_name,
            report_month,
            service_code,
            service_name,
        ) in enumerate(
            zip(
                frame["facility_code"],
                frame["facility_name"],
                frame["report_month"],
                frame["service_code"],
                frame["service_name"],
                strict=False,
            )
        )
    ]

    return frame[
        CANONICAL_COLUMNS
    ].copy()


def _drop_duplicate_canonical_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate canonical business keys."""

    if frame.empty:
        return frame.copy(), 0

    key_columns = [
        "resource_kind",
        "facility_code",
        "facility_name",
        "report_month",
        "service_code",
        "service_name",
    ]

    duplicate_mask = frame.duplicated(
        subset=key_columns,
        keep="last",
    )

    removed = int(
        duplicate_mask.sum()
    )

    return (
        frame.loc[
            ~duplicate_mask
        ].reset_index(
            drop=True
        ),
        removed,
    )


def _read_facility_aliases(
    path: Path,
) -> pd.DataFrame:
    """Read exact facility aliases without fuzzy matching."""

    columns = [
        "alias_name",
        "canonical_name",
        "canonical_code",
        "hhs",
        "region",
        "active",
    ]

    if not path.exists():
        return pd.DataFrame(
            columns=columns
        )

    aliases = pd.read_csv(
        path,
        dtype=str,
    )

    for column in columns:
        if column not in aliases.columns:
            aliases[column] = pd.NA

    aliases = aliases[
        columns
    ].copy()

    if aliases.empty:
        return aliases

    aliases["alias_name"] = (
        aliases["alias_name"]
        .map(_normalise_text)
    )

    aliases["canonical_name"] = (
        aliases["canonical_name"]
        .map(_normalise_text)
    )

    aliases["canonical_code"] = (
        aliases["canonical_code"]
        .map(_normalise_code)
    )

    active = (
        aliases["active"]
        .fillna("true")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(
            {
                "1",
                "true",
                "yes",
                "y",
            }
        )
    )

    aliases = aliases.loc[
        active
    ].reset_index(
        drop=True
    )

    return aliases.drop_duplicates(
        subset=[
            "alias_name",
        ],
        keep="last",
    )


def _resolve_facilities(
    canonical: pd.DataFrame,
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve facility identity using exact aliases."""

    frame = canonical.copy()

    alias_lookup = (
        {}
        if aliases.empty
        else aliases.set_index(
            "alias_name"
        ).to_dict(
            orient="index"
        )
    )

    resolved_codes: list[
        str | None
    ] = []

    resolved_names: list[
        str | None
    ] = []

    hhs_values: list[
        str | None
    ] = []

    region_values: list[
        str | None
    ] = []

    statuses: list[str] = []

    for source_code, source_name in zip(
        frame["facility_code"],
        frame["facility_name"],
        strict=False,
    ):
        alias = alias_lookup.get(
            source_name
        )

        if alias is None:
            resolved_codes.append(
                source_code
            )
            resolved_names.append(
                source_name
            )
            hhs_values.append(None)
            region_values.append(None)
            statuses.append("source")
            continue

        resolved_codes.append(
            _normalise_code(
                alias.get(
                    "canonical_code"
                )
            )
            or source_code
        )

        resolved_names.append(
            _normalise_text(
                alias.get(
                    "canonical_name"
                )
            )
            or source_name
        )

        hhs_values.append(
            _normalise_text(
                alias.get("hhs")
            )
        )

        region_values.append(
            _normalise_text(
                alias.get("region")
            )
        )

        statuses.append("alias")

    frame[
        "resolved_facility_code"
    ] = resolved_codes

    frame[
        "resolved_facility_name"
    ] = resolved_names

    frame["hhs"] = hhs_values
    frame["region"] = region_values

    frame[
        "facility_resolution_status"
    ] = statuses

    frame["facility_key"] = [
        _stable_key(
            code,
            name,
        )
        for code, name in zip(
            resolved_codes,
            resolved_names,
            strict=False,
        )
    ]

    return frame


def _add_service_keys(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add specialty and urgency-category keys."""

    result = frame.copy()

    specialty_keys: list[
        str | None
    ] = []

    urgency_keys: list[
        str | None
    ] = []

    for (
        resource_kind,
        service_code,
        service_name,
    ) in zip(
        result["resource_kind"],
        result["service_code"],
        result["service_name"],
        strict=False,
    ):
        if resource_kind == "specialty":
            specialty_keys.append(
                _stable_key(
                    service_code,
                    service_name,
                )
            )
            urgency_keys.append(None)

        elif resource_kind == "category":
            specialty_keys.append(None)
            urgency_keys.append(
                _stable_key(
                    service_name
                )
            )

        else:
            specialty_keys.append(None)
            urgency_keys.append(None)

    result[
        "specialty_key"
    ] = specialty_keys

    result[
        "urgency_category_key"
    ] = urgency_keys

    return result


def _add_source_resource_key(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add deterministic source lineage keys."""

    result = frame.copy()

    result[
        "source_resource_key"
    ] = [
        _stable_key(
            resource_id,
            sha256,
            source_url,
            source_file,
        )
        for (
            resource_id,
            sha256,
            source_url,
            source_file,
        ) in zip(
            result["source_resource_id"],
            result["source_sha256"],
            result["source_url"],
            result["source_file"],
            strict=False,
        )
    ]

    return result


def _build_longitudinal(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Add descriptive longitudinal measures."""

    if canonical.empty:
        return canonical.copy()

    frame = canonical.copy()

    group_columns = [
        "facility_key",
        "resource_kind",
        "service_code",
        "service_name",
    ]

    frame = frame.sort_values(
        [
            *group_columns,
            "report_month",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    grouped = frame.groupby(
        group_columns,
        dropna=False,
        sort=False,
    )

    frame[
        "previous_vol_waiting"
    ] = grouped[
        "vol_waiting"
    ].shift(1)

    frame[
        "backlog_change"
    ] = (
        frame["vol_waiting"]
        - frame[
            "previous_vol_waiting"
        ]
    )

    frame[
        "previous_vol_long_waits"
    ] = grouped[
        "vol_long_waits"
    ].shift(1)

    frame[
        "long_wait_change"
    ] = (
        frame["vol_long_waits"]
        - frame[
            "previous_vol_long_waits"
        ]
    )

    waiting_denominator = (
        frame["vol_waiting"]
        .replace(
            0,
            pd.NA,
        )
    )

    frame[
        "long_wait_share"
    ] = (
        frame["vol_long_waits"]
        / waiting_denominator
    )

    frame[
        "treatment_to_waiting_ratio"
    ] = (
        frame["vol_treated"]
        / waiting_denominator
    )

    return frame


def _build_dim_facility(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the facility dimension."""

    columns = [
        "facility_key",
        "facility_code",
        "facility_name",
        "hhs",
        "region",
        "resolution_status",
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=columns
        )

    dimension = pd.DataFrame(
        {
            "facility_key": frame[
                "facility_key"
            ],
            "facility_code": frame[
                "resolved_facility_code"
            ],
            "facility_name": frame[
                "resolved_facility_name"
            ],
            "hhs": frame["hhs"],
            "region": frame["region"],
            "resolution_status": frame[
                "facility_resolution_status"
            ],
        }
    )

    dimension[
        "_completeness"
    ] = (
        dimension[
            [
                "facility_code",
                "facility_name",
                "hhs",
                "region",
            ]
        ]
        .notna()
        .sum(axis=1)
    )

    return (
        dimension
        .sort_values(
            [
                "facility_key",
                "_completeness",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "facility_key",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_completeness",
            ]
        )
        .sort_values(
            "facility_key",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def _build_dim_specialty(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the specialty dimension."""

    columns = [
        "specialty_key",
        "specialty_code",
        "specialty_name",
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=columns
        )

    dimension = frame.loc[
        (
            frame["resource_kind"]
            == "specialty"
        )
        & frame[
            "specialty_key"
        ].notna(),
        [
            "specialty_key",
            "service_code",
            "service_name",
        ],
    ].copy()

    dimension = dimension.rename(
        columns={
            "service_code": (
                "specialty_code"
            ),
            "service_name": (
                "specialty_name"
            ),
        }
    )

    return (
        dimension
        .drop_duplicates(
            subset=[
                "specialty_key",
            ],
            keep="last",
        )
        .sort_values(
            "specialty_key",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def _build_dim_urgency_category(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the urgency category dimension."""

    columns = [
        "urgency_category_key",
        "urgency_category_name",
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=columns
        )

    dimension = frame.loc[
        (
            frame["resource_kind"]
            == "category"
        )
        & frame[
            "urgency_category_key"
        ].notna(),
        [
            "urgency_category_key",
            "service_name",
        ],
    ].copy()

    dimension = dimension.rename(
        columns={
            "service_name": (
                "urgency_category_name"
            )
        }
    )

    return (
        dimension
        .drop_duplicates(
            subset=[
                "urgency_category_key",
            ],
            keep="last",
        )
        .sort_values(
            "urgency_category_key",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def _build_dim_reporting_period(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the reporting-period dimension."""

    columns = [
        "reporting_period_key",
        "calendar_year",
        "calendar_quarter",
        "month",
        "quarter_label",
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=columns
        )

    periods = (
        frame["report_month"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    dimension = pd.DataFrame(
        {
            "reporting_period_key": (
                periods.dt.date
            ),
            "calendar_year": (
                periods.dt.year
            ),
            "calendar_quarter": (
                periods.dt.quarter
            ),
            "month": (
                periods.dt.month
            ),
        }
    )

    dimension[
        "quarter_label"
    ] = (
        dimension[
            "calendar_year"
        ].astype(str)
        + "-Q"
        + dimension[
            "calendar_quarter"
        ].astype(str)
    )

    return dimension[
        columns
    ].reset_index(
        drop=True
    )


def _build_dim_source_resource(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the source-resource dimension."""

    columns = [
        "source_resource_key",
        "resource_id",
        "source_sha256",
        "source_url",
        "source_file",
        "retrieved_at",
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=columns
        )

    dimension = pd.DataFrame(
        {
            "source_resource_key": frame[
                "source_resource_key"
            ],
            "resource_id": frame[
                "source_resource_id"
            ],
            "source_sha256": frame[
                "source_sha256"
            ],
            "source_url": frame[
                "source_url"
            ],
            "source_file": frame[
                "source_file"
            ],
            "retrieved_at": frame[
                "source_retrieved_at"
            ],
        }
    )

    dimension[
        "retrieved_at"
    ] = pd.to_datetime(
        dimension["retrieved_at"],
        errors="coerce",
        utc=True,
    )

    return (
        dimension
        .drop_duplicates(
            subset=[
                "source_resource_key",
            ],
            keep="last",
        )
        .sort_values(
            "source_resource_key",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def _build_fact_performance(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the elective-surgery performance fact."""

    columns = [
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

    if frame.empty:
        return pd.DataFrame(
            columns=columns
        )

    fact = frame.copy()

    fact[
        "reporting_period_key"
    ] = pd.to_datetime(
        fact["report_month"],
        errors="coerce",
    ).dt.date

    return fact[
        columns
    ].reset_index(
        drop=True
    )


def _load_quality_events(
    reports_dir: Path,
) -> pd.DataFrame:
    """Load Phase 3 quality events when available."""

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

    path = (
        reports_dir
        / "outputs"
        / "data_quality_summary.json"
    )

    if not path.exists():
        return pd.DataFrame(
            columns=columns
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return pd.DataFrame(
            columns=columns
        )

    events: list[
        dict[str, object]
    ] = []

    files = payload.get(
        "files",
        []
    )

    if not isinstance(files, list):
        return pd.DataFrame(
            columns=columns
        )

    for file_entry in files:
        if not isinstance(
            file_entry,
            dict,
        ):
            continue

        source_path = (
            file_entry.get(
                "source_path"
            )
            or file_entry.get("path")
        )

        resource_kind = (
            file_entry.get(
                "resource_kind"
            )
        )

        file_events = (
            file_entry.get("events")
            or file_entry.get(
                "quality_events"
            )
            or []
        )

        if not isinstance(
            file_events,
            list,
        ):
            continue

        for event in file_events:
            if not isinstance(
                event,
                dict,
            ):
                continue

            event_id = (
                event.get("event_id")
                or _stable_key(
                    source_path,
                    resource_kind,
                    event.get("rule_id"),
                    event.get("severity"),
                    event.get("row_index"),
                    event.get(
                        "column_name"
                    ),
                    event.get(
                        "observed_value"
                    ),
                    event.get("message"),
                )
            )

            events.append(
                {
                    "event_id": (
                        event_id
                    ),
                    "source_path": (
                        source_path
                    ),
                    "resource_kind": (
                        resource_kind
                    ),
                    "rule_id": (
                        event.get(
                            "rule_id"
                        )
                    ),
                    "severity": (
                        event.get(
                            "severity"
                        )
                    ),
                    "row_index": (
                        event.get(
                            "row_index"
                        )
                    ),
                    "column_name": (
                        event.get(
                            "column_name"
                        )
                    ),
                    "observed_value": (
                        None
                        if event.get(
                            "observed_value"
                        )
                        is None
                        else str(
                            event.get(
                                "observed_value"
                            )
                        )
                    ),
                    "message": (
                        event.get(
                            "message"
                        )
                    ),
                }
            )

    quality = pd.DataFrame(
        events,
        columns=columns,
    )

    if quality.empty:
        return quality

    quality[
        "row_index"
    ] = pd.to_numeric(
        quality["row_index"],
        errors="coerce",
    ).astype("Int64")

    return (
        quality
        .drop_duplicates(
            subset=[
                "event_id",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


def _replace_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Replace warehouse contents using explicit named columns."""

    if table_name not in WAREHOUSE_TABLES:
        raise ValueError(
            "Unsupported warehouse table: "
            f"{table_name}"
        )

    view_name = (
        f"_load_{table_name}"
    )

    connection.register(
        view_name,
        dataframe,
    )

    try:
        connection.execute(
            f'DELETE FROM "{table_name}"'
        )

        if dataframe.empty:
            return

        columns = list(
            dataframe.columns
        )

        quoted_columns = ", ".join(
            f'"{column}"'
            for column in columns
        )

        connection.execute(
            f"""
            INSERT INTO "{table_name}" (
                {quoted_columns}
            )
            SELECT
                {quoted_columns}
            FROM "{view_name}"
            """
        )

    finally:
        connection.unregister(
            view_name
        )


def _resolve_warehouse_ddl_path() -> Path:
    """Locate sql/create_warehouse.sql portably."""

    candidates = [
        (
            Path.cwd()
            / "sql"
            / "create_warehouse.sql"
        ),
        (
            Path(__file__)
            .resolve()
            .parents[3]
            / "sql"
            / "create_warehouse.sql"
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate "
        "sql/create_warehouse.sql."
    )


def _execute_ddl(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create warehouse tables."""

    ddl_path = (
        _resolve_warehouse_ddl_path()
    )

    connection.execute(
        ddl_path.read_text(
            encoding="utf-8"
        )
    )


def _write_reconciliation(
    *,
    path: Path,
    canonical: pd.DataFrame,
    longitudinal: pd.DataFrame,
    fact_performance: pd.DataFrame,
    dim_facility: pd.DataFrame,
    dim_specialty: pd.DataFrame,
    dim_urgency_category: pd.DataFrame,
    dim_reporting_period: pd.DataFrame,
    dim_source_resource: pd.DataFrame,
    fact_quality: pd.DataFrame,
    duplicate_canonical_keys_removed: int,
) -> None:
    """Write warehouse reconciliation metadata."""

    payload = {
        "generated_at": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "canonical_rows": int(
            len(canonical)
        ),
        "longitudinal_rows": int(
            len(longitudinal)
        ),
        "fact_rows": int(
            len(fact_performance)
        ),
        "facility_rows": int(
            len(dim_facility)
        ),
        "specialty_rows": int(
            len(dim_specialty)
        ),
        "urgency_category_rows": int(
            len(dim_urgency_category)
        ),
        "reporting_period_rows": int(
            len(dim_reporting_period)
        ),
        "source_resource_rows": int(
            len(dim_source_resource)
        ),
        "quality_event_rows": int(
            len(fact_quality)
        ),
        "duplicate_canonical_keys_removed": (
            duplicate_canonical_keys_removed
        ),
        "fact_matches_longitudinal": (
            len(fact_performance)
            == len(longitudinal)
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_summary(
    *,
    summary: WarehouseBuildSummary,
    reports_dir: Path,
) -> Path:
    """Write warehouse build summary JSON."""

    output_path = (
        reports_dir
        / "outputs"
        / "warehouse_build_summary.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(
        summary
    )

    payload = {
        key: (
            str(value)
            if isinstance(
                value,
                Path,
            )
            else value
        )
        for key, value
        in payload.items()
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return output_path


def build_warehouse(
    *,
    settings: AppSettings,
) -> WarehouseBuildSummary:
    """Build canonical, longitudinal and DuckDB warehouse outputs."""

    processed_directory = Path(
        settings.processed_data_dir
    )

    reports_directory = Path(
        settings.reports_dir
    )

    raw_directory = Path(
        settings.raw_data_dir
    )

    interim_directory = Path(
        settings.interim_data_dir
    )

    facility_aliases_path = Path(
        settings.facility_aliases_path
    )

    duckdb_path = Path(
        settings.duckdb_path
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        reports_directory
        / "outputs"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    duckdb_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validated_files = (
        _find_validated_files(
            interim_directory
        )
    )

    manifest = _read_manifest(
        raw_directory
    )

    canonical_frames: list[
        pd.DataFrame
    ] = []

    for validated_path in (
        validated_files
    ):
        source = pd.read_parquet(
            validated_path
        )

        resource_kind = (
            _infer_resource_kind(
                validated_path
            )
        )

        lineage = (
            _manifest_row_for_validated_file(
                validated_path,
                manifest,
            )
        )

        canonical_frames.append(
            _canonicalise_frame(
                source=source,
                validated_path=(
                    validated_path
                ),
                resource_kind=(
                    resource_kind
                ),
                lineage=lineage,
            )
        )

    if canonical_frames:
        canonical = pd.concat(
            canonical_frames,
            ignore_index=True,
            sort=False,
        )
    else:
        canonical = pd.DataFrame(
            columns=CANONICAL_COLUMNS
        )

    (
        canonical,
        duplicate_canonical_keys_removed,
    ) = _drop_duplicate_canonical_rows(
        canonical
    )

    aliases = _read_facility_aliases(
        facility_aliases_path
    )

    enriched = _resolve_facilities(
        canonical,
        aliases,
    )

    enriched = _add_service_keys(
        enriched
    )

    enriched = _add_source_resource_key(
        enriched
    )

    longitudinal = _build_longitudinal(
        enriched
    )

    canonical_output = (
        processed_directory
        / "canonical_performance.parquet"
    )

    longitudinal_output = (
        processed_directory
        / "longitudinal_performance.parquet"
    )

    canonical.to_parquet(
        canonical_output,
        index=False,
    )

    longitudinal.to_parquet(
        longitudinal_output,
        index=False,
    )

    dim_facility = (
        _build_dim_facility(
            longitudinal
        )
    )

    dim_specialty = (
        _build_dim_specialty(
            longitudinal
        )
    )

    dim_urgency_category = (
        _build_dim_urgency_category(
            longitudinal
        )
    )

    dim_reporting_period = (
        _build_dim_reporting_period(
            longitudinal
        )
    )

    dim_source_resource = (
        _build_dim_source_resource(
            longitudinal
        )
    )

    fact_performance = (
        _build_fact_performance(
            longitudinal
        )
    )

    fact_quality = (
        _load_quality_events(
            reports_directory
        )
    )

    with duckdb.connect(
        str(duckdb_path)
    ) as connection:
        _execute_ddl(
            connection
        )

        connection.execute(
            "BEGIN TRANSACTION"
        )

        try:
            _replace_table(
                connection,
                table_name=(
                    "dim_facility"
                ),
                dataframe=(
                    dim_facility
                ),
            )

            _replace_table(
                connection,
                table_name=(
                    "dim_specialty"
                ),
                dataframe=(
                    dim_specialty
                ),
            )

            _replace_table(
                connection,
                table_name=(
                    "dim_urgency_category"
                ),
                dataframe=(
                    dim_urgency_category
                ),
            )

            _replace_table(
                connection,
                table_name=(
                    "dim_reporting_period"
                ),
                dataframe=(
                    dim_reporting_period
                ),
            )

            _replace_table(
                connection,
                table_name=(
                    "dim_source_resource"
                ),
                dataframe=(
                    dim_source_resource
                ),
            )

            _replace_table(
                connection,
                table_name=(
                    "fact_elective_surgery_performance"
                ),
                dataframe=(
                    fact_performance
                ),
            )

            _replace_table(
                connection,
                table_name=(
                    "fact_data_quality_event"
                ),
                dataframe=(
                    fact_quality
                ),
            )

            connection.execute(
                "COMMIT"
            )

        except Exception:
            connection.execute(
                "ROLLBACK"
            )
            raise

    reconciliation_report_path = (
        reports_directory
        / "outputs"
        / "warehouse_reconciliation.json"
    )

    _write_reconciliation(
        path=(
            reconciliation_report_path
        ),
        canonical=canonical,
        longitudinal=longitudinal,
        fact_performance=(
            fact_performance
        ),
        dim_facility=(
            dim_facility
        ),
        dim_specialty=(
            dim_specialty
        ),
        dim_urgency_category=(
            dim_urgency_category
        ),
        dim_reporting_period=(
            dim_reporting_period
        ),
        dim_source_resource=(
            dim_source_resource
        ),
        fact_quality=(
            fact_quality
        ),
        duplicate_canonical_keys_removed=(
            duplicate_canonical_keys_removed
        ),
    )

    summary = WarehouseBuildSummary(
        validated_files=len(
            validated_files
        ),
        canonical_rows=len(
            canonical
        ),
        longitudinal_rows=len(
            longitudinal
        ),
        facility_count=len(
            dim_facility
        ),
        specialty_count=len(
            dim_specialty
        ),
        urgency_category_count=len(
            dim_urgency_category
        ),
        reporting_period_count=len(
            dim_reporting_period
        ),
        source_resource_count=len(
            dim_source_resource
        ),
        quality_event_count=len(
            fact_quality
        ),
        duplicate_canonical_keys_removed=(
            duplicate_canonical_keys_removed
        ),
        duckdb_path=duckdb_path,
        canonical_path=(
            canonical_output
        ),
        longitudinal_path=(
            longitudinal_output
        ),
        reconciliation_report_path=(
            reconciliation_report_path
        ),
    )

    _write_summary(
        summary=summary,
        reports_dir=(
            reports_directory
        ),
    )

    return summary