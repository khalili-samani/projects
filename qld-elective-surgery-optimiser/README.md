
# Queensland Elective Surgery Capacity and Waitlist Recovery Optimiser

**An end-to-end Python data engineering, analytics and mathematical optimisation project using Queensland Government open data.**

This project investigates how hypothetical additional elective surgery capacity could be allocated across Queensland hospitals and surgical services to support waitlist recovery.

It combines public healthcare statistics, reproducible data ingestion, data-quality controls, analytical modelling, a DuckDB warehouse and a constrained optimisation model built with Google OR-Tools CP-SAT.

The project has been successfully executed from **live public data ingestion through to an optimal synthetic capacity-allocation solution**.

**Technologies:** Python 3.12 · Pandas · DuckDB · Parquet · SQL · Google OR-Tools CP-SAT · Pydantic · Typer · pytest · Ruff · mypy

---

## 1. Project overview

Queensland publishes aggregate elective surgery statistics describing waiting lists, treatment activity, long waits, surgical specialties and urgency categories.

These statistics provide a foundation for analysing service demand, but translating them into a capacity-allocation model requires several engineering and analytical steps.

The main question explored is:

> If a limited amount of additional elective surgery capacity were available, how could it be distributed across facilities and services while prioritising waiting demand and long-wait burden?

The project addresses this question through five connected capabilities:

1. **Data ingestion:** Discover and download Queensland Government elective surgery resources while preserving original files and source lineage.
2. **Data validation:** Handle historical CSV variations, validate records and quarantine invalid observations.
3. **Analytics engineering:** Standardise source records, resolve facility identities and calculate longitudinal waitlist measures.
4. **Data warehousing:** Build canonical Parquet datasets and a reconciled DuckDB analytical warehouse.
5. **Mathematical optimisation:** Allocate hypothetical additional capacity using explicit constraints and a configurable objective function.

The result is a reproducible analytical workflow rather than an isolated optimisation algorithm.

### Verified execution results

The latest documented execution completed the full pipeline using live Queensland Government resources.

| Metric | Verified result |
| --- | ---: |
| Source resources discovered and ingested | 31 |
| Category-level resources | 16 |
| Specialty-level resources | 15 |
| Source records processed | 11,021 |
| Records passing validation | 11,021 |
| Quarantined records | 0 |
| Canonical analytical records | 11,021 |
| Longitudinal analytical records | 11,021 |
| DuckDB fact records | 11,021 |
| Canonical duplicates removed | 0 |
| Facilities represented | 108 |
| Surgical specialties | 12 |
| Urgency categories | 4 |
| Reporting periods | 16 |
| Optimisation solver status | Optimal |
| Hypothetical additional capacity | 100 cases |
| Hypothetical additional cases allocated | 100 |
| Unused hypothetical capacity | 0 |
| Capacity utilisation | 100% |

The live ingestion, validation, warehouse build and optimisation commands all completed successfully.

**Interpretation:** The optimisation result represents an optimal allocation for the implemented mathematical model and synthetic scenario assumptions. It is not evidence that 100 additional operations occurred or that any hospital's waiting list was reduced in practice.

---

## 2. Why this project matters

Elective surgery capacity planning involves several competing considerations.

Facilities and services can differ in:

- Waiting-list volume.
- Number of patients experiencing long waits.
- Treatment activity.
- Changes in waiting demand between reporting periods.
- Potential benefit from hypothetical additional activity.

A useful analytical system needs to bring these measures together without losing their original meaning or inventing information that is not present in the published data.

This project demonstrates how to:

- Engineer a reproducible pipeline from a public government API.
- Manage changes in historical source formats.
- Validate data explicitly rather than silently accepting inconsistent records.
- Retain provenance from raw files into analytical outputs.
- Reconcile records across processing and warehousing stages.
- Formulate an operational allocation problem mathematically.
- Distinguish observed healthcare data from synthetic planning assumptions.
- Explain what an optimisation result does and does not establish.

The emphasis is on **data quality, analytical traceability and transparent optimisation**.

---

## 3. Data source and scope

### Primary data source

The project uses the Queensland Government Open Data **Elective surgery** dataset.

**Dataset:** [Queensland Government Elective surgery](https://www.data.qld.gov.au/dataset/elective-surgery)

**CKAN API base:**

```text
https://www.data.qld.gov.au/api/3/action
```

**Dataset identifier:**

```text
d660d925-400f-42ba-8245-a08bfc18abf4
```

The data are retrieved through the CKAN API rather than a fixed collection of manually downloaded spreadsheets.

### Two separate reporting families

The publisher provides aggregate elective surgery information through two distinct views.

| Source family | Reporting dimension |
| --- | --- |
| Category | Elective surgery urgency category |
| Specialty | Surgical specialty |

These source families contain overlapping healthcare concepts but different reporting dimensions.

The project processes them as separate analytical observations.

**It does not create an unsupported specialty-by-urgency-category breakdown.**

For example, a category-level observation cannot be assumed to reveal how many patients within that urgency category belong to a particular surgical specialty.

### Reporting coverage

The verified live snapshot contains:

- 31 source resources.
- 16 category resources.
- 15 specialty resources.
- 16 distinct reporting periods.
- Reporting periods extending through June 2026.

### Data governance

The project uses aggregate public statistics.

It does not process:

- Patient names or identifiers.
- Individual medical histories.
- Patient-level waiting-list records.
- Individual clinical risk assessments.
- Patient-specific surgical scheduling decisions.

The optimisation operates at an aggregate facility/service level.

---

## 4. Architecture

The system is organised into distinct ingestion, validation, processing, warehouse and optimisation layers.

```text
Queensland Government Open Data
              |
              v
       CKAN discovery
              |
              v
        CSV ingestion
              |
              v
    Immutable raw storage
              |
              v
     SHA-256 + manifest
              |
              v
    Source-layout detection
              |
              v
    Coercion + validation
           /      \
          v        v
    Valid data   Quarantine
          |        |
          |    Quality report
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
    Warehouse reconciliation
          |
          v
    Optimisation data contract
          |
          v
    Synthetic scenario settings
          |
          v
    OR-Tools CP-SAT model
          |
          v
    Allocation decisions
          |
          v
    Diagnostics + reports
```

Each stage has a distinct responsibility, making it easier to inspect, test and troubleshoot the pipeline.

---

## 5. Data ingestion and provenance

The ingestion layer discovers and retrieves eligible elective surgery CSV resources through the Queensland Government CKAN API.

It provides:

- Resource discovery and classification.
- CSV downloading.
- Version-aware raw storage.
- SHA-256 content hashing.
- Source metadata capture.
- An ingestion manifest.
- Reuse of already-downloaded content where applicable.

The original downloaded bytes are preserved instead of being overwritten by cleaned or transformed data.

### Raw data manifest

The ingestion manifest is stored at:

```text
data/raw/manifest.csv
```

It includes lineage information such as:

```text
dataset_id
dataset_title
source_organisation
source_licence
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

This information supports reproducibility and identifies the source associated with an analytical observation.

### Verified ingestion result

The live ingestion run reported:

```text
Eligible resources: 31
Resources selected: 31
New files downloaded: 27
Already present: 4
Total manifest records: 31

Status: completed successfully
```

The result demonstrates that the pipeline can operate against the current collection of published resources, not only controlled test fixtures.

---

## 6. Historical CSV compatibility

One of the important engineering challenges was variation in the publisher's historical CSV structures.

The source-layout detector supports:

- Current headered category files.
- Legacy uppercase category headers.
- Headerless category files.
- Category files containing three trailing empty fields.
- Current headered specialty files.
- Headerless specialty files.

The legacy category padding is accepted only when the additional fields are confirmed to be empty.

Unexpected source structures are rejected rather than silently interpreted using an uncertain schema.

### Why this matters

Without source-layout detection, a pipeline designed for the latest CSV format may fail on older files or interpret values as the wrong columns.

The adapter provides a controlled way to transform supported historical layouts into consistent column names while preserving the original source files.

---

## 7. Data validation and quarantine

The validation layer reads source values as text before applying explicit type conversion and quality rules.

Its checks include:

- Required business-key values.
- Numeric parsing.
- Percentage parsing.
- Reporting-period parsing.
- Non-negative volume constraints.
- Percentage-range validation.
- Duplicate business keys.
- Long waits not exceeding total waiting volume.
- Consistency between long-wait components.
- Unsupported source layouts.

Invalid observations can be quarantined with their original values and quality-rule information.

### Reporting-period normalisation

Historical source resources contain different reporting-period formats.

Examples supported by the date parser include:

| Original value | Parsed value |
| --- | --- |
| `JUN2023` | `2023-06-01` |
| `Sep-24` | `2024-09-01` |
| `Jun-26` | `2026-06-01` |
| `16-Jan-2024` | `2024-01-16` |
| `07/11/2026` | `2026-11-07` |

Month-year values are represented using the first day of the month as a consistent analytical convention.

This representation does not imply that the published healthcare activity occurred on that specific day.

Australian day/month/year ordering is used for applicable numeric date formats.

### Correct percentage semantics

The validation rules distinguish between bounded percentages and percentage changes.

A percentage representing a proportion, such as the proportion treated within time, is expected to fall between 0% and 100%.

A year-on-year percentage variation can legitimately be negative or exceed 100%.

For example:

```text
Previous-year treatment volume: 10
Current treatment volume:       30

Percentage variation:          +200%
```

Rejecting this variation because it exceeds 100% would be a data-quality error.

The revised validation layer applies bounds only to measures for which those bounds are meaningful.

### Verified validation result

```text
Files processed:       31
Files passed:          31
Files failed:           0

Rows read:          11021
Rows valid:         11021
Rows quarantined:       0

Rule counts: {}
```

**All 11,021 records passed the implemented validation rules.**

This means no violations were detected under those rules. It does not mean the source data have been independently certified as error-free.

The validation report is written to:

```text
reports/outputs/data_quality_summary.json
```

---

## 8. Canonical analytical model

The processing layer converts publisher-specific column names and source structures into a consistent analytical schema.

Representative canonical fields include:

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

### Analytical grain

The data are structured around:

```text
facility × reporting period × service
```

where the service belongs to one of the two separate source families:

```text
resource_kind = specialty
```

or:

```text
resource_kind = category
```

The source family remains explicit so that observations from distinct reporting dimensions are not incorrectly combined.

### Facility identity resolution

Facility names and identifiers are resolved using deterministic mapping rules and reviewed reference data.

The approach avoids unrestricted fuzzy matching that could incorrectly merge different facilities.

### Verified canonical results

The live warehouse build reported:

```text
Canonical rows:           11021
Canonical duplicates removed: 0
```

No canonical records were lost through duplicate removal in the verified run.

---

## 9. Longitudinal waitlist analytics

The processing layer derives descriptive measures for comparing observations across reporting periods.

| Measure | Definition or purpose |
| --- | --- |
| `previous_vol_waiting` | Waiting volume from the preceding available observation |
| `backlog_change` | Current waiting volume minus previous waiting volume |
| `previous_vol_long_waits` | Long-wait volume from the preceding observation |
| `long_wait_change` | Change in long-wait volume |
| `long_wait_share` | Long-wait volume divided by waiting volume |
| `treatment_to_waiting_ratio` | Treated volume divided by waiting volume |

Conceptually:

```text
backlog_change =
    current_vol_waiting - previous_vol_waiting

long_wait_change =
    current_vol_long_waits - previous_vol_long_waits

long_wait_share =
    vol_long_waits / vol_waiting

treatment_to_waiting_ratio =
    vol_treated / vol_waiting
```

These measures support descriptive analysis of reported demand and activity.

They should not be interpreted as evidence that changes in treatment activity caused particular waiting-list outcomes.

### Verified longitudinal result

```text
Longitudinal rows: 11021
```

The longitudinal dataset retained the same number of observations as the canonical dataset in the verified run.

---

## 10. DuckDB analytical warehouse

The project builds a DuckDB warehouse from the canonical and longitudinal datasets.

It includes dimensional tables such as:

```text
dim_facility
dim_specialty
dim_urgency_category
dim_reporting_period
dim_source_resource
```

and fact tables such as:

```text
fact_elective_surgery_performance
fact_data_quality_event
```

### Generated files

The warehouse build produces outputs including:

```text
data/processed/
    canonical_performance.parquet
    longitudinal_performance.parquet
    elective_surgery.duckdb

reports/outputs/
    warehouse_reconciliation.json
    warehouse_build_summary.json
```

### Verified warehouse build

The command:

```bash
python -m qld_surgery_optimiser.cli warehouse
```

completed successfully and reported:

| Measure | Result |
| --- | ---: |
| Validated source files | 31 |
| Canonical rows | 11,021 |
| Longitudinal rows | 11,021 |
| Warehouse fact rows | 11,021 |
| Facilities | 108 |
| Specialties | 12 |
| Urgency categories | 4 |
| Reporting periods | 16 |
| Source-resource dimension records | 31 |
| Data-quality events | 0 |

### Reconciliation

The generated reconciliation report contained:

```json
{
  "canonical_rows": 11021,
  "duplicate_canonical_keys_removed": 0,
  "facility_rows": 108,
  "fact_matches_longitudinal": true,
  "fact_rows": 11021,
  "longitudinal_rows": 11021,
  "quality_event_rows": 0,
  "reporting_period_rows": 16,
  "source_resource_rows": 31,
  "specialty_rows": 12,
  "urgency_category_rows": 4
}
```

This verifies the reported row-count agreement between the longitudinal dataset and the warehouse fact table.

The reconciliation does not, by itself, prove that every analytical field or relationship has been independently checked against the publisher's original data.

---

## 11. Capacity optimisation model

The optimisation layer uses Google OR-Tools CP-SAT to allocate hypothetical additional elective surgery activity.

The model is formulated as an integer constrained allocation problem.

### Optimisation grain

The baseline allocation decision is made for each:

```text
facility_key × service_key
```

A scenario selects one source family at a time, either specialty or urgency category.

The model does not create unsupported combinations between these reporting dimensions.

### Decision variables

For each eligible facility/service combination, the solver selects an integer number of additional cases:

```text
additional_cases[facility, service] >= 0
```

The selected values represent **hypothetical extra activity**.

They are not measurements of current hospital operating capacity.

### Capacity constraints

The baseline model supports three principal constraints.

**1. Total scenario capacity**

```text
sum(additional_cases) <= total_additional_cases
```

Total allocated activity cannot exceed the scenario's capacity assumption.

**2. Service-level capacity ceiling**

```text
additional_cases[facility, service]
    <= default_max_additional_cases_per_service
```

An optional service-level limit prevents excessive allocation to a single facility/service combination.

**3. Waiting-volume ceiling**

```text
additional_cases[facility, service]
    <= observed_waiting_volume
```

Where enabled, the model cannot allocate more hypothetical cases than the reported waiting volume associated with that allocation row.

### Objective function

The objective rewards allocating additional capacity towards waiting demand, with additional emphasis on long-wait burden.

It uses configurable weights:

```text
waiting_weight
long_wait_weight
```

and a long-wait share derived from:

```text
vol_long_waits / vol_waiting
```

The model uses integer-scaled coefficients compatible with CP-SAT.

Its objective is a transparent mathematical representation of the selected planning priorities, not a clinically validated formula.

---

## 12. Synthetic optimisation scenario

The baseline scenario is configured in:

```text
configs/scenarios/optimisation_baseline.yml
```

Its structure includes:

```yaml
scenario:
  name: baseline_capacity_recovery
  description: >
    Synthetic baseline scenario for testing elective surgery
    capacity allocation across facility-service combinations.
  resource_kind: specialty
  synthetic_assumptions: true

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

The scenario contract explicitly requires:

```text
synthetic_assumptions: true
```

This helps prevent hypothetical operating-capacity parameters from being mistaken for observed public healthcare measurements.

### Optimisation input checks

Before constructing the solver model, the analytical data are checked for required fields such as:

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

The contract checks source-family selection, reporting periods, uniqueness, data types and volume consistency.

---

## 13. Verified optimisation results

The baseline optimisation was executed against the longitudinal Parquet dataset generated from the live Queensland Government resources.

The command completed successfully:

```text
Optimisation completed successfully.

Scenario: baseline_capacity_recovery
Solver status: optimal

Capacity available: 100
Additional cases allocated: 100
Unused capacity: 0
Capacity utilisation: 100.0%
```

### Result interpretation

| Metric | Meaning |
| --- | --- |
| Solver status: optimal | CP-SAT found an optimal solution for the mathematical model |
| Capacity available: 100 | The scenario assumes 100 additional cases can be allocated |
| Cases allocated: 100 | The model assigned the full hypothetical capacity |
| Unused capacity: 0 | No hypothetical capacity was left unallocated |
| Utilisation: 100% | All capacity allowed by the scenario was used |

The solution demonstrates that the mathematical model was able to construct and solve a feasible allocation problem using live-data-derived analytical inputs.

**It does not establish that 100 additional procedures can actually be delivered.**

Real-world delivery would depend on factors such as operating theatres, surgical teams, anaesthesia, beds, equipment, case complexity, clinical priority and funding.

The baseline scenario does not validate all of those operational constraints.

### Persisted outputs

The run generated:

```text
reports/outputs/optimisation/
    baseline_capacity_recovery/
        optimisation_summary.json
        optimisation_allocations.csv
        optimisation_diagnostics.json
```

These artefacts contain the model summary, facility/service allocation decisions and diagnostic measures.

The optimisation CLI reported successful output creation. A separate detailed review of the persisted allocation distribution and diagnostics has not yet been documented.

---

## 14. Installation and local setup

### Requirements

- Python 3.12.
- Git.
- Internet access for live CKAN discovery and ingestion.

The configured Python compatibility range is:

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

### Install the package

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

An editable installation ensures changes to the project's source files are reflected in the active Python environment.

If working with multiple copies of the repository, verify which package installation is active.

### Run the environment check

```bash
python --version
python -m qld_surgery_optimiser.cli doctor
```

### List CLI commands

```bash
python -m qld_surgery_optimiser.cli --help
```

Available commands include:

| Command | Purpose |
| --- | --- |
| `doctor` | Check configuration and prepare directories |
| `show-config` | Display resolved non-secret configuration |
| `discover` | Query the CKAN source catalogue |
| `ingest` | Download and archive raw resources |
| `validate` | Apply data-quality checks |
| `warehouse` | Build canonical, longitudinal and DuckDB datasets |
| `optimise` | Run the CP-SAT allocation workflow |

---

## 15. Reproduce the pipeline

The following workflow runs the principal stages in order.

### Step 1: Check configuration

```bash
python -m qld_surgery_optimiser.cli doctor
```

### Step 2: Discover resources

```bash
python -m qld_surgery_optimiser.cli discover
```

### Step 3: Ingest raw source data

```bash
python -m qld_surgery_optimiser.cli ingest
```

The pipeline maintains source lineage and preserves raw source versions.

### Step 4: Validate source observations

```bash
python -m qld_surgery_optimiser.cli validate
```

Review:

```text
reports/outputs/data_quality_summary.json
```

### Step 5: Build the warehouse

```bash
python -m qld_surgery_optimiser.cli warehouse
```

Review:

```text
reports/outputs/warehouse_reconciliation.json
```

### Step 6: Run optimisation

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

A reporting period can also be selected explicitly using:

```text
--reporting-period YYYY-MM-DD
```

Without this option, the optimisation input contract selects the latest eligible period for the scenario's source family.

### Reproducibility note

The published CKAN catalogue may change as additional reporting periods become available or existing resources are revised.

Consequently, future live runs may discover different resource counts or produce different optimisation inputs.

Reproducing the exact documented snapshot requires retaining the associated raw files, manifest and scenario configuration.

---

## 16. Repository structure

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
│   │   └── manifest.csv
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
│       ├── validation/
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

The project uses a `src/` package structure and separates functionality into independently testable modules.

---

## 17. Automated testing and code quality

The project includes unit and integration tests across the ingestion, validation, analytical and optimisation workflows.

Test areas include:

- Configuration handling.
- CKAN resource discovery.
- Raw file downloading.
- Manifest and lineage management.
- Historical source-layout detection.
- Controlled type coercion.
- Data-quality rules and quarantine.
- Facility identity resolution.
- Canonical normalisation.
- Longitudinal measures.
- Warehouse construction and reconciliation.
- Optimisation input contracts.
- Scenario validation.
- CP-SAT model construction.
- Constraints and objective functions.
- Solver results.
- Allocation diagnostics.
- CLI integration.

### Run automated tests

```bash
pytest
```

### Run Ruff

```bash
ruff check .
```

### Run mypy

```bash
mypy src tests
```

The project has previously documented a passing suite of 143 tests. Subsequent code and test changes mean the latest total should be taken from a fresh test run rather than assumed to remain at 143.

The successful live pipeline runs are separate evidence from the automated test suite.

---

## 18. Engineering verification summary

The project has now demonstrated successful execution across its main pipeline stages.

| Stage | Status | Evidence |
| --- | --- | --- |
| CKAN resource discovery | Verified | 31 eligible resources |
| Raw data ingestion | Verified | 31 resources archived and recorded |
| Historical source-format handling | Verified | All discovered resources ingested and validated |
| Data validation | Verified | 11,021 accepted; zero quarantined |
| Canonical modelling | Verified | 11,021 records |
| Longitudinal processing | Verified | 11,021 records |
| DuckDB warehouse | Verified | 11,021 fact records |
| Warehouse reconciliation | Verified | Fact count matches longitudinal count |
| CP-SAT optimisation | Verified | Optimal solution |
| Synthetic capacity allocation | Verified | 100 of 100 hypothetical cases allocated |
| Detailed persisted optimisation diagnostics review | Pending | Output files generated but not yet individually reviewed |
| Interactive decision-support interface | Not implemented | Future development |

The verified milestone is an end-to-end execution of the implemented data and optimisation workflow using live public-source data and synthetic planning parameters.

This does not amount to clinical validation, operational deployment or proof of realised waitlist improvement.

---

## 19. Limitations and responsible interpretation

### Public aggregate data

The source statistics describe aggregated activity and waiting demand.

They cannot provide individual patient-level prioritisation or clinical risk information.

### Distinct category and specialty views

The published category and specialty resources represent different groupings of aggregate data.

They must not be treated as a jointly observed specialty-by-category dataset.

### Synthetic additional capacity

The optimisation model uses a hypothetical capacity budget.

It does not determine actual spare operating theatre time, staffing or funded additional capacity.

### Simplified objective

The allocation objective prioritises waiting demand and long-wait burden.

Real healthcare planning requires additional clinical, equity, financial, operational and governance considerations.

### Optimality is model-specific

An optimal mathematical solution is optimal only relative to the implemented objective, constraints and supplied data.

It should not be interpreted as the best possible real-world clinical or operational allocation.

### Reporting-period semantics

Month-year labels are normalised to first-of-month dates for analytical consistency.

They are reporting-period identifiers, not individual event dates.

### Source catalogue changes

Resources and source schemas may change over time.

The live ingestion and validation controls should be rerun when new resources become available.

### Independent portfolio project

This system is an independent technical demonstration.

It is not a Queensland Health production application and is not claimed to have influenced actual surgical scheduling or health-service planning.

---

## 20. Future development

With the principal ingestion-to-optimisation workflow now executed successfully, further work can focus on strengthening analytical verification and making the results easier to explore.

### Data engineering and verification

- Independently inspect warehouse relationships and sample source-to-fact lineage.
- Add focused regression tests for historical reporting-period formats.
- Extend source-schema drift detection.
- Improve reporting-period-aware latest-resource selection.
- Avoid recording sensitive temporary URL query parameters in ingestion logs.

### Optimisation analysis

- Review the persisted allocation distribution and diagnostics.
- Compare multiple hypothetical capacity budgets.
- Conduct objective-weight sensitivity analysis.
- Compare alternative service-level limits.
- Add richer binding-constraint and infeasibility explanations.

### Reporting and visualisation

- Facility-level allocation charts.
- Scenario comparison tables.
- Longitudinal waiting-list visualisations.
- Downloadable analytical reports.

### Application layer

Potential later development includes:

- FastAPI.
- Streamlit.
- Interactive scenario configuration.
- A dashboard for exploring allocations and model diagnostics.

### Deployment

Additional future work may include containerisation, CI/CD improvements and packaged reproducible workflows.

---

## 21. What this project demonstrates

The repository brings together multiple technical disciplines in one reproducible workflow.

| Discipline | Demonstrated capabilities |
| --- | --- |
| Data engineering | Public API integration, ingestion, checksums, versioned raw data and lineage |
| Data quality | Controlled parsing, validation rules, quarantine and reporting |
| Analytics engineering | Canonical schemas, deterministic keys, longitudinal measures and reconciliation |
| SQL and warehousing | DuckDB, Parquet, dimensional modelling and analytical fact tables |
| Mathematical optimisation | Integer decisions, constraints, objective functions and CP-SAT solving |
| Software engineering | Modular Python design, CLI workflows, type checking and automated testing |
| Technical governance | Explicit synthetic assumptions, reproducibility and responsible interpretation |

### Final perspective

The project demonstrates a complete technical workflow:

```text
Public healthcare data
          |
          v
Reproducible ingestion
          |
          v
Data-quality validation
          |
          v
Canonical analytical modelling
          |
          v
Longitudinal waitlist analytics
          |
          v
DuckDB analytical warehouse
          |
          v
Reconciliation checks
          |
          v
Synthetic capacity assumptions
          |
          v
Constrained mathematical optimisation
          |
          v
Allocation results and diagnostics
```

The central design principle is that **mathematical optimisation should be supported by transparent assumptions, reproducible data engineering and traceable analytical inputs**.

The completed execution provides evidence that this architecture works across the tested live public-data snapshot and baseline hypothetical scenario, while preserving the distinction between an optimal model solution and actual healthcare outcomes.

## Licence

This project is distributed under the MIT Licence. See [`LICENSE`](LICENSE).