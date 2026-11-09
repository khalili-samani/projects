
# Queensland Elective Surgery Capacity and Waitlist Recovery Optimiser

**A reproducible data engineering, analytics and mathematical optimisation project using publicly available Queensland elective surgery data.**

The project explores how hypothetical additional elective surgery capacity could be allocated across facilities and services to support waitlist recovery, subject to transparent capacity constraints and configurable optimisation objectives.

It combines real public-sector data with an auditable analytical pipeline and a synthetic capacity-allocation model.

**Core technologies:** Python 3.12 · Pandas · DuckDB · Parquet · Google OR-Tools CP-SAT · Pydantic · Typer · pytest

## Project at a glance

| Area | Description |
| --- | --- |
| Domain | Healthcare operations and elective surgery |
| Geographic scope | Queensland, Australia |
| Primary source | Queensland Government Open Data |
| Data granularity | Aggregate facility, service and reporting-period statistics |
| Data engineering | CKAN discovery, immutable ingestion, lineage, validation and canonical modelling |
| Analytics | Longitudinal waitlist measures and DuckDB warehousing |
| Optimisation | Integer capacity allocation using Google OR-Tools CP-SAT |
| Scenario assumptions | Synthetic and explicitly identified |
| Interface | Python command-line application |
| Current milestone | Live ingestion and validation verified; live warehouse and optimisation verification pending |

### Latest verified live-data results

The most recent documented live execution, completed on **10 November 2026 (AEDT)**, successfully ingested and validated the currently discovered Queensland Government elective surgery resources.

| Measure | Result |
| --- | ---: |
| Resources discovered | 31 |
| Category-level resources | 16 |
| Specialty-level resources | 15 |
| Resources ingested | 31 |
| Source records read | 11,021 |
| Records passing validation | 11,021 |
| Quarantined records | 0 |
| Files failing validation | 0 |
| Row acceptance rate | 100% |

These results demonstrate successful **source ingestion and validation**, not completed verification of every downstream warehouse or optimisation stage.

The catalogue is externally maintained, so the number of resources and their contents may change in future runs.

---

## 1. The problem

Elective surgery planning involves competing demands across facilities, services and reporting periods.

A facility may report substantial waiting demand, a high proportion of long waits or changes in treatment activity. Planning additional activity requires deciding how limited resources might be distributed while respecting operational constraints.

The central question explored here is:

> Given a hypothetical quantity of additional elective surgery capacity, how might it be allocated across Queensland facilities and services to prioritise waiting demand and long-wait burden?

This is not simply an optimisation problem.

Before allocation decisions can be modelled, the underlying data need to be discovered, downloaded, validated, standardised and reconciled.

The project therefore treats **data quality and analytical traceability as prerequisites for optimisation**.

### Project objectives

1. Build a reproducible pipeline for Queensland elective surgery open data.
2. Preserve original source files and resource-level provenance.
3. Support historical variations in the published CSV formats.
4. Validate source records using explicit business and data-quality rules.
5. Produce a consistent analytical model with longitudinal waitlist measures.
6. Make aggregate facility and service information queryable through DuckDB.
7. Implement a transparent capacity-allocation model using synthetic assumptions.
8. Produce interpretable allocation decisions and diagnostics.
9. Distinguish verified technical results from hypothetical operational outcomes.

---

## 2. Data source

### Queensland Government Open Data

The primary source is the Queensland Government's **Elective surgery** dataset.

- Dataset: [Queensland Government Elective surgery](https://www.data.qld.gov.au/dataset/elective-surgery)
- CKAN API: `https://www.data.qld.gov.au/api/3/action`
- Dataset identifier: `d660d925-400f-42ba-8245-a08bfc18abf4`
- Source organisation: Queensland Health
- Licence reported in the ingestion manifest: Creative Commons Attribution 4.0

The ingestion layer discovers eligible resources through the CKAN API rather than relying on a fixed list of manually maintained CSV links.

### Two distinct source families

The published data provide separate aggregate reporting views:

| Source family | Main analytical dimension |
| --- | --- |
| Category | Elective surgery urgency category |
| Specialty | Surgical specialty |

Both include information about reporting periods, facilities, treatment activity, waiting volumes and long waits.

**These are separate aggregate views.**

The project does not construct an unsupported facility-by-specialty-by-urgency-category dataset, because the published sources do not provide that joint distribution.

### Data governance

The project operates on aggregate public data. It does not use patient identifiers, individual medical histories or patient-level waiting-list records.

No optimisation output represents an individual treatment recommendation.

---

## 3. System architecture

The pipeline is organised into ingestion, validation, processing, warehousing and optimisation components.

```text
Queensland Government CKAN API
                |
                v
        Resource discovery
                |
                v
          CSV ingestion
                |
                v
     Immutable raw storage
                |
                v
     SHA-256 + source manifest
                |
                v
      Source-layout detection
                |
                v
      Coercion and validation
            /       \
           v         v
    Valid records   Quarantine
           |         |
           |     Quality report
           v
    Canonical modelling
           |
           v
    Facility resolution
           |
           v
    Longitudinal measures
           |
           v
     Parquet + DuckDB
           |
           v
      Reconciliation
           |
           v
   Optimisation input contract
           |
           v
   Synthetic scenario settings
           |
           v
     Google OR-Tools CP-SAT
           |
           v
    Allocation + diagnostics
           |
           v
       JSON / CSV outputs
```

The principal components are implemented in the Python package. Live execution of the ingestion and validation stages has been verified; the downstream live-data integration checks are the next milestone.

---

## 4. Data ingestion and source lineage

The ingestion pipeline:

- Queries the Queensland Government CKAN catalogue.
- Identifies eligible elective surgery CSV resources.
- Classifies resources as category or specialty.
- Downloads original publisher-supplied bytes.
- Uses SHA-256 checksums for content identification.
- Stores files without casually overwriting earlier raw versions.
- Maintains metadata and source lineage in a manifest.
- Reuses already-present content when its checksum matches.

The manifest is written to:

```text
data/raw/manifest.csv
```

It records fields including:

```text
dataset_id
resource_id
resource_name
resource_kind
source_url
retrieved_at
local_path
sha256
byte_count
content_type
```

This supports reproducibility and allows source resources to be traced from downloaded files into subsequent processing.

### Historical CSV compatibility

A substantial live-data engineering challenge was that the publisher's historical resources did not use one consistent CSV layout.

The source-layout adapter supports:

- Modern headered category CSVs.
- Legacy category headers requiring canonical name mapping.
- Headerless category CSVs.
- Historical category files containing three verified empty trailing fields.
- Modern headered specialty CSVs.
- Headerless specialty CSVs.

Unsupported layouts, inconsistent record widths and populated trailing fields are rejected rather than silently reinterpreted.

The original raw files remain unchanged.

### Verified ingestion results

The latest live run reported:

```text
Resources discovered: 31
Resources selected: 31
New raw files: 27
Already present: 4
Manifest records: 31
Ingestion status: successful
```

The 31 resources consisted of 16 category files and 15 specialty files.

---

## 5. Validation and data quality

The validation layer reads raw source records as text before applying controlled transformations.

The workflow checks:

- Required source columns.
- Required business-key values.
- Numeric volumes.
- Percentage values.
- Reporting-period dates.
- Non-negative patient volumes.
- Bounded percentage measures.
- Duplicate business keys.
- Long waits relative to total waiting volume.
- Long-wait component consistency.
- Unsupported or malformed source layouts.

Invalid observations are identified through quality-rule identifiers and can be quarantined with their original values and source details.

### Explicit reporting-period parsing

Historical source files use several date representations.

Supported examples include:

| Source value | Normalised value |
| --- | --- |
| `Sep-24` | `2024-09-01` |
| `Jun-26` | `2026-06-01` |
| `JUN2023` | `2023-06-01` |
| `16-Jan-2024` | `2024-01-16` |
| `07/11/2026` | `2026-11-07` |

Month-year reporting labels are represented by the first day of the corresponding month. This is a modelling convention, not a claim about the specific day on which the underlying activity occurred.

Numeric day/month/year dates follow Australian conventions.

### Percentage semantics

The validation logic distinguishes two kinds of percentage measure.

**Bounded percentages**, such as the proportion of patients treated within time, are expected to lie between 0% and 100%.

**Percentage changes**, such as year-on-year treatment-volume variation, may legitimately be negative or exceed 100%.

For example, an increase from 10 to 30 cases represents a 200% increase. Applying an ordinary 0–100 percentage bound would incorrectly reject it.

Percentage changes remain subject to numeric parsing and other applicable checks.

### Verified validation results

The latest live run produced:

```text
Files processed: 31
Files passed: 31
Files failed: 0

Rows read: 11021
Rows valid: 11021
Rows quarantined: 0

Rule counts: {}
```

The quality report is written to:

```text
reports/outputs/data_quality_summary.json
```

A 100% acceptance rate means all records passed the currently implemented rules for this run. It does not establish that the source data are free of every possible error or that downstream analytical outputs have been fully reconciled.

---

## 6. Canonical analytical modelling

The processing layer transforms the publisher's source-specific fields into a consistent analytical schema.

The intended canonical structure includes:

```text
facility_key
facility_name
report_month
resource_kind
service_code
service_name
vol_treated
vol_waiting
vol_long_waits
source_resource_key
```

### Facility identity resolution

Facility identities are resolved using deterministic rules and reviewed aliases rather than unrestricted fuzzy matching.

This reduces the risk of combining different facilities because of superficially similar names.

### Analytical grain

The canonical observations retain the distinction between:

```text
facility × reporting period × category
```

and:

```text
facility × reporting period × specialty
```

The two source families are not merged into a fabricated joint classification.

### Longitudinal measures

The processing layer derives measures such as:

| Measure | Interpretation |
| --- | --- |
| `previous_vol_waiting` | Waiting volume in the preceding available period |
| `backlog_change` | Current waiting volume minus previous waiting volume |
| `previous_vol_long_waits` | Previous long-wait volume |
| `long_wait_change` | Change in long-wait volume |
| `long_wait_share` | Long waits divided by total waiting volume |
| `treatment_to_waiting_ratio` | Treated volume divided by waiting volume |

These are descriptive measures.

They do not establish causal relationships between additional treatment capacity and waiting-list outcomes.

---

## 7. DuckDB analytical warehouse

The implemented warehouse layer uses Parquet and DuckDB to provide structured analytical outputs.

Its dimensional model includes:

```text
dim_facility
dim_specialty
dim_urgency_category
dim_reporting_period
dim_source_resource
```

The fact model includes:

```text
fact_elective_surgery_performance
fact_data_quality_event
```

The warehouse uses explicit named-column insertion and includes reconciliation reporting.

Typical expected outputs are:

```text
data/processed/canonical_performance.parquet
data/processed/longitudinal_performance.parquet
data/processed/<warehouse>.duckdb

reports/outputs/warehouse_reconciliation.json
reports/outputs/warehouse_build_summary.json
```

**Verification status:** The warehouse implementation is covered by controlled automated tests, but a complete live-data warehouse build and reconciliation against the newly ingested 31 resources has not yet been confirmed.

No live warehouse row counts or reconciliation results are claimed here.

---

## 8. Mathematical optimisation

The optimisation component uses **Google OR-Tools CP-SAT**, a constraint programming solver.

It allocates a synthetic quantity of additional elective surgery cases across eligible facility/service combinations.

### Optimisation grain

The baseline decision grain is:

```text
facility_key × service_key
```

A scenario selects either the specialty or category source family.

### Decision variable

For each facility/service combination:

```text
additional_cases[facility, service] >= 0
```

Each decision variable is a non-negative integer representing hypothetical additional activity.

### Capacity constraints

The baseline model supports three principal constraints.

**Total hypothetical capacity**

```text
sum(additional_cases) <= total_additional_cases
```

The model cannot allocate more activity than the scenario-wide capacity assumption.

**Service-level capacity ceiling**

```text
additional_cases[facility, service]
    <= default_max_additional_cases_per_service
```

This limits the modelled additional allocation to each service.

**Waiting-volume ceiling**

```text
additional_cases[facility, service]
    <= observed waiting volume
```

Where enabled, the model cannot allocate more additional cases than the reported waiting volume for the relevant facility/service combination.

### Objective function

The baseline objective rewards allocations towards waiting demand, with additional emphasis on long-wait burden.

It uses configurable weights:

```text
waiting_weight
long_wait_weight
```

and information derived from:

```text
long_wait_share = vol_long_waits / vol_waiting
```

The coefficients are scaled for integer CP-SAT optimisation.

The objective is designed to be explainable and reproducible. It is not a clinically validated or Queensland Health-endorsed prioritisation formula.

### What the optimisation does not model

The baseline model does not establish actual available theatre sessions, staffing rosters, surgical case mix, clinical urgency decisions, patient-level scheduling feasibility or realised reductions in waiting times.

The capacity constraints are synthetic scenario inputs rather than observed Queensland hospital operating capacity.

---

## 9. Optimisation scenarios and outputs

The project keeps planning configuration and optimisation scenario configuration separate.

The optimisation baseline is defined in:

```text
configs/scenarios/optimisation_baseline.yml
```

An illustrative scenario is:

```yaml
scenario:
  name: baseline_capacity_recovery
  description: >
    Synthetic baseline scenario for testing elective surgery
    capacity allocation across facility-service combinations.
  resource_kind: specialty
  synthetic_assumptions: true
  notes: >
    Capacity values are modelling assumptions, not observed
    Queensland Health operating capacity.

capacity:
  total_additional_cases: 100
  default_max_additional_cases_per_service: 25

objective:
  waiting_weight: 1
  long_wait_weight: 3

constraints:
  enforce_waiting_volume_ceiling: true
  enforce_service_capacity_ceiling: true

solver:
  max_time_seconds: 30
  num_search_workers: 1
  random_seed: 0
```

The optimisation scenario contract explicitly requires:

```text
synthetic_assumptions: true
```

### Optimisation input contract

The model validates required analytical fields before constructing the solver.

These include:

```text
facility_key
facility_name
resource_kind
report_month
service_code
service_name
vol_waiting
vol_long_waits
```

It checks the selected source family, reporting period, facility/service uniqueness, non-negative whole-number volumes and long-wait consistency.

### Expected outputs

A successful scenario run writes outputs similar to:

```text
reports/outputs/optimisation/
└── baseline_capacity_recovery/
    ├── optimisation_summary.json
    ├── optimisation_allocations.csv
    └── optimisation_diagnostics.json
```

The allocation results describe hypothetical model decisions.

Diagnostics may include total allocated cases, unused capacity, utilisation, services receiving allocation and capacity-ceiling behaviour.

These are **scenario outputs**, not observed changes in healthcare performance.

---

## 10. Repository structure

A simplified view of the project:

```text
qld-elective-surgery-optimiser/
│
├── configs/
│   ├── base.yml
│   ├── facilities.yml
│   └── scenarios/
│       ├── baseline.yml
│       └── optimisation_baseline.yml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── quarantine/
│   └── reference/
│
├── reports/
│   └── outputs/
│
├── sql/
│   ├── create_warehouse.sql
│   └── reconciliation.sql
│
├── src/
│   └── qld_surgery_optimiser/
│       ├── ingestion/
│       │   ├── source_layout.py
│       │   └── ...
│       ├── validation/
│       │   ├── coercion.py
│       │   ├── quality_rules.py
│       │   └── ...
│       ├── processing/
│       ├── optimisation/
│       ├── cli.py
│       ├── config.py
│       ├── exceptions.py
│       └── logging_config.py
│
├── tests/
│   ├── integration/
│   └── unit/
│
├── pyproject.toml
└── README.md
```

The repository uses a `src/` package layout and separates ingestion, validation, analytical processing and optimisation concerns.

---

## 11. Getting started

### Requirements

- Python 3.12
- Git
- An internet connection for live CKAN ingestion

The project's configured Python target is:

```text
>=3.12,<3.13
```

### Clone the repository

```bash
git clone https://github.com/khalili-samani/projects.git
cd projects/qld-elective-surgery-optimiser
```

### Create a virtual environment

**Windows CMD**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**macOS or Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

An editable installation is recommended for development so that Python uses the source code in the selected working copy.

### Check the environment

```bash
python --version
python -m qld_surgery_optimiser.cli doctor
```

Then inspect the available commands:

```bash
python -m qld_surgery_optimiser.cli --help
```

---

## 12. Run the pipeline

The command-line interface provides the following workflows:

| Command | Purpose |
| --- | --- |
| `doctor` | Check configuration and prepare local directories |
| `show-config` | Display resolved non-secret configuration |
| `discover` | Discover eligible resources through CKAN |
| `ingest` | Download and version raw source files |
| `validate` | Validate raw observations and produce quality reporting |
| `warehouse` | Build canonical Parquet datasets and the DuckDB warehouse |
| `optimise` | Run a synthetic capacity-allocation scenario |

### Step 1: Discover resources

```bash
python -m qld_surgery_optimiser.cli discover
```

### Step 2: Ingest source files

```bash
python -m qld_surgery_optimiser.cli ingest
```

To use the project's latest-resource selection option:

```bash
python -m qld_surgery_optimiser.cli ingest --latest-only
```

The meaning of "latest" should be checked against reporting periods rather than assumed from CKAN metadata timestamps alone.

### Step 3: Validate

```bash
python -m qld_surgery_optimiser.cli validate
```

Inspect:

```text
reports/outputs/data_quality_summary.json
```

The 10 November 2026 live validation completed with 11,021 valid records and zero quarantined observations.

### Step 4: Build the warehouse

```bash
python -m qld_surgery_optimiser.cli warehouse
```

Inspect generated Parquet, DuckDB and reconciliation outputs.

This is the next outstanding live-data integration milestone.

### Step 5: Run an optimisation scenario

**Windows CMD**

```bat
python -m qld_surgery_optimiser.cli optimise ^
  --analytical-data data\processed\longitudinal_performance.parquet ^
  --scenario configs\scenarios\optimisation_baseline.yml ^
  --output-dir reports\outputs\optimisation
```

**macOS or Linux**

```bash
python -m qld_surgery_optimiser.cli optimise \
  --analytical-data data/processed/longitudinal_performance.parquet \
  --scenario configs/scenarios/optimisation_baseline.yml \
  --output-dir reports/outputs/optimisation
```

An explicit reporting period can be provided using:

```text
--reporting-period YYYY-MM-DD
```

If omitted, the optimisation contract selects the latest eligible period for the scenario's chosen source family.

An optimisation run should only be interpreted after its analytical inputs, solver status and diagnostics have been examined.

---

## 13. Automated testing and code quality

The project includes unit and integration tests covering the main engineering components.

Test areas include:

- Configuration and source discovery.
- CKAN client behaviour.
- Immutable downloading and manifest handling.
- Historical CSV source-layout detection.
- Data coercion and quality rules.
- Quarantine and validation reporting.
- Facility resolution and canonical normalisation.
- Longitudinal processing.
- Warehouse construction and reconciliation.
- Optimisation input contracts.
- Scenario validation.
- CP-SAT constraints and objective construction.
- Solver behaviour and diagnostics.
- Optimisation pipeline integration and CLI execution.

Run the complete automated suite:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Run static type checking:

```bash
mypy src tests
```

### Verification boundaries

Automated tests use controlled inputs, fixtures and mocks where appropriate.

They establish expected behaviour in the tested conditions but do not replace live-source integration testing.

The repository has previously recorded a passing automated suite of 143 tests, with subsequent test additions and source changes. An updated current test count should be taken from a fresh complete `pytest` run rather than inferred from that older checkpoint.

The live ingestion and validation results in this README are independently evidenced by actual command-line execution.

---

## 14. Development status

| Component | Implementation status | Live-data verification |
| --- | --- | --- |
| Architecture and configuration | Implemented | Configuration tested |
| CKAN resource discovery | Implemented | Verified |
| Raw ingestion and lineage | Implemented | Verified for 31 resources |
| Historical CSV layout adaptation | Implemented | Verified during live ingestion |
| Validation and quarantine | Implemented | Verified for 11,021 records |
| Canonical analytical modelling | Implemented | Pending full live-data reconciliation |
| Longitudinal processing | Implemented | Pending full live-data reconciliation |
| DuckDB warehouse | Implemented | Pending live warehouse build verification |
| OR-Tools CP-SAT optimisation | Implemented | Pending live-data-derived scenario verification |
| Interactive decision-support interface | Not implemented | Not applicable |

### Completed live-data milestone

The public-source ingestion and validation pipeline has successfully processed all 31 resources available in the documented run.

This included resolving historical source-format inconsistencies and reporting-period variations.

### Next milestone

Build and reconcile the canonical analytical datasets and DuckDB warehouse against the validated 31-resource snapshot.

Following that, execute a synthetic optimisation scenario using the verified analytical data and inspect the resulting allocations, constraints and diagnostics.

---

## 15. Known limitations

### Aggregate rather than patient-level data

The project cannot estimate patient-specific waiting times, clinical risk or individual treatment priorities.

### Separate source families

Category and specialty statistics cannot be treated as a jointly observed specialty-by-category distribution.

### Synthetic capacity assumptions

The baseline model does not use measured additional theatre, staffing or surgical capacity. Capacity values are hypothetical scenario parameters.

### Simplified optimisation objective

Waiting and long-wait burden provide a transparent modelling objective, but do not capture all clinical, operational, equity or resource-allocation considerations.

### Source reporting changes

The public CKAN catalogue may introduce new resources, revised schemas, new date formats or changes to metadata.

Future runs should continue to validate source compatibility.

### Latest-resource selection

The most recently modified CKAN resource is not necessarily the one with the latest clinical reporting period.

Reporting-period-aware selection remains an important source-management consideration.

### Downstream live verification

Although the warehouse and optimisation components are implemented and covered by controlled tests, their current live-data integration results are not yet established.

### Operational deployment

The project is a portfolio demonstration. It is not an operational Queensland Health system or a clinically endorsed resource-allocation tool.

---

## 16. Engineering principles

**Reproducibility:** Raw inputs are versioned, checksummed and linked to their source metadata.

**Data quality:** Invalid or unsupported inputs are surfaced explicitly rather than silently discarded or reinterpreted.

**Traceability:** Analytical observations retain source identifiers and provenance.

**Determinism:** Facility resolution and scenario settings favour reproducible behaviour.

**Separation of concerns:** Observed data, derived measures, synthetic assumptions and model decisions are distinct.

**Responsible interpretation:** Model allocations are not presented as realised health outcomes.

**Verification before claims:** Documentation distinguishes implemented functionality, passing automated tests and confirmed live-data execution.

---

## 17. Roadmap

Near-term engineering priorities are:

1. Complete the live DuckDB warehouse build.
2. Reconcile canonical, longitudinal and warehouse record counts.
3. Verify source lineage and reporting-period coverage in downstream outputs.
4. Execute a synthetic optimisation scenario using live-data-derived analytical inputs.
5. Inspect solver status, capacity utilisation and allocation diagnostics.
6. Add regression coverage for historical reporting-period formats and percentage-change semantics.
7. Improve logging hygiene by avoiding exposure of temporary presigned URL parameters.

Potential later extensions include scenario comparison, sensitivity analysis, interactive dashboards, FastAPI or Streamlit interfaces, downloadable reports and deployment tooling.

---

## 18. Project interpretation

This project does not claim that a simple mathematical model can solve elective surgery planning in practice.

Instead, it demonstrates a disciplined approach to building a technical decision-support workflow:

```text
Public source data
    → Reproducible ingestion
    → Explicit validation
    → Canonical analytical model
    → Traceable warehouse
    → Synthetic scenario assumptions
    → Constrained optimisation
    → Interpretable diagnostics
```

The central principle is that **optimisation should be built on trustworthy, auditable data and transparent assumptions**, rather than treated as a standalone algorithm.

The verified milestone is successful live ingestion and validation. Subsequent integration milestones will be documented as they are completed.