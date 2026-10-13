# Queensland Elective Surgery Capacity and Waitlist Recovery Optimiser

A reproducible data engineering and optimisation project for exploring how hypothetical additional elective surgery capacity could be allocated across Queensland facilities and services to support waitlist recovery.

The project combines:

- public Queensland elective surgery data;
- reproducible source ingestion and lineage;
- validation and quarantine;
- canonical analytical modelling;
- longitudinal waitlist measures;
- a DuckDB analytical warehouse;
- scenario-driven optimisation with Google OR-Tools CP-SAT;
- allocation diagnostics;
- persisted optimisation outputs;
- a Typer command-line interface; and
- automated unit, integration, linting and static-type checks.

> **Current status:** Phases 1–5 are implemented. The current local automated test suite passes with **143 tests**. Ruff and mypy quality gates also pass locally. Live end-to-end processing of the current Queensland Government CKAN resources remains a separate verification milestone and is **not yet claimed as verified**.

---

## 1. The problem

Queensland publishes aggregate elective surgery performance data across hospitals, specialties and urgency categories.

A practical capacity-planning question is:

> Given a limited amount of additional elective surgery capacity, where should that capacity be allocated to support the strongest waitlist-recovery outcome while respecting operational constraints?

Answering that credibly requires more than an optimisation algorithm.

The underlying data must first be:

- discovered reproducibly;
- downloaded with lineage;
- validated;
- quarantined when invalid;
- normalised into a consistent analytical model;
- reconciled across pipeline stages; and
- separated from any synthetic assumptions introduced for modelling.

This repository therefore treats data engineering, analytical modelling and optimisation as one end-to-end decision-support system.

---

## 2. What the project currently does

The implemented system can:

- discover eligible elective surgery resources from Queensland Government Open Data;
- classify supported resources by source family;
- download source CSV files into immutable versioned storage;
- calculate and retain SHA-256 checksums;
- maintain source lineage in an ingestion manifest;
- validate schemas and controlled data types;
- quarantine invalid observations;
- produce machine-readable data-quality reporting;
- normalise source data into a canonical analytical model;
- resolve facility identities deterministically;
- calculate longitudinal waitlist measures;
- persist processed Parquet datasets;
- construct a DuckDB analytical warehouse;
- generate reconciliation metadata;
- build optimisation-ready facility/service inputs;
- load typed synthetic optimisation scenarios;
- solve additional-capacity allocation problems using OR-Tools CP-SAT;
- produce service-level allocation decisions;
- calculate optimisation diagnostics;
- persist optimisation summaries, allocations and diagnostics;
- execute the optimisation pipeline through the command line; and
- validate behaviour through unit and integration tests.

The system operates on aggregate public data and does **not** make patient-level clinical decisions.

---

## 3. Key design boundary

The project deliberately separates four different information classes:

### Observed public data

Examples:

- waiting-list volume;
- long-wait volume;
- treatment activity;
- reporting period;
- facility;
- specialty or urgency category.

### Derived analytical measures

Examples:

- backlog change;
- previous-period waiting volume;
- long-wait share;
- treatment-to-waiting ratio.

### Synthetic scenario assumptions

Examples:

- total hypothetical additional cases available;
- maximum hypothetical additional cases per service;
- objective weights.

These assumptions are explicitly labelled as synthetic.

### Model decisions and diagnostics

Examples:

- recommended additional cases by facility/service;
- unused hypothetical capacity;
- capacity utilisation;
- binding scenario ceilings.

Optimisation outputs are **modelled scenario results**, not observed healthcare outcomes and not guarantees of future performance.

---

## 4. Data source

### Primary source

Queensland Government Open Data — Elective Surgery dataset.

CKAN API base:

```text
https://www.data.qld.gov.au/api/3/action
```

Dataset:

```text
elective-surgery
```

Known dataset identifier:

```text
d660d925-400f-42ba-8245-a08bfc18abf4
```

The published resources include aggregate elective surgery information broadly corresponding to:

- elective surgery by specialty; and
- elective surgery by urgency category.

The ingestion layer discovers resources through CKAN rather than depending on manually copied download URLs.

### Secondary contextual source

Australian Institute of Health and Welfare material may be used for contextual interpretation and benchmarking.

AIHW data is not currently treated as the primary transactional input to the optimisation warehouse.

---

## 5. System architecture

The implemented architecture is:

```text
Queensland Government CKAN
           |
           v
   Resource discovery
           |
           v
    Raw CSV ingestion
           |
           v
Immutable versioned storage
           |
           v
      Manifest + SHA-256
           |
           v
 Validation and coercion
       /           \
      /             \
 Validated data   Quarantine
      |               |
      |         Quality reporting
      v
 Canonical normalisation
      |
      v
 Facility resolution
      |
      v
 Longitudinal processing
      |
      v
 Analytical dimensions/facts
      |
      v
 Parquet + DuckDB warehouse
      |
      v
 Reconciliation reporting
      |
      v
 Optimisation input contract
      |
      v
 Synthetic scenario config
      |
      v
 OR-Tools CP-SAT model
      |
      v
 Allocation decisions
      |
      v
 Optimisation diagnostics
      |
      v
 JSON + CSV outputs
      |
      v
 CLI decision-support workflow
```

---

## 6. Optimisation model

The Phase 5 optimisation layer allocates hypothetical additional elective surgery activity across facility/service combinations.

### Optimisation grain

The baseline optimisation grain is:

```text
facility_key x service_key
```

The model works on **one published source family at a time**.

For example:

```text
resource_kind: specialty
```

or:

```text
resource_kind: category
```

The project does not create a synthetic facility × specialty × urgency-category cube because specialty and urgency category are published as separate aggregate views.

---

## 7. Decision variable

For each facility/service combination, the model creates an integer decision variable representing:

```text
additional_cases
```

Conceptually:

```text
additional_cases[facility, service] >= 0
```

The decision variable represents modelled hypothetical extra activity under the selected scenario.

It is not an observed Queensland Health capacity value.

---

## 8. Capacity constraints

The baseline model supports:

### Total scenario capacity

The sum of allocated additional cases cannot exceed the scenario-wide capacity assumption.

Conceptually:

```text
sum(additional_cases) <= total_additional_cases
```

### Service-level capacity ceiling

Where enabled:

```text
additional_cases[facility, service]
    <= default_max_additional_cases_per_service
```

### Waiting-volume ceiling

Where enabled:

```text
additional_cases[facility, service]
    <= observed waiting volume
```

This prevents the baseline model from assigning more hypothetical additional activity than the reported waiting volume for that optimisation row.

---

## 9. Objective function

The baseline objective rewards allocation towards services with waiting demand while placing additional emphasis on long-wait burden.

The objective uses:

```text
waiting_weight
long_wait_weight
```

and a scaled long-wait share derived from:

```text
vol_long_waits / vol_waiting
```

The resulting coefficients are integer-scaled for CP-SAT.

A higher long-wait burden therefore increases the relative attractiveness of allocating hypothetical additional capacity to that facility/service combination, subject to the configured constraints.

The objective is intentionally transparent and configurable rather than presented as a clinically validated prioritisation formula.

---

## 10. Optimisation scenarios

Phase 5 optimisation scenarios are stored separately from the earlier planning/simulation configuration.

Current structure:

```text
configs/scenarios/
|-- baseline.yml
`-- optimisation_baseline.yml
```

These files have different purposes.

### `baseline.yml`

Used by the existing planning/simulation configuration model.

### `optimisation_baseline.yml`

Used by the Phase 5 CP-SAT allocation model.

A baseline optimisation scenario resembles:

```yaml
scenario:
  name: baseline_capacity_recovery
  description: >
    Synthetic baseline scenario for testing elective surgery
    capacity allocation across facility-service combinations.
  resource_kind: specialty
  synthetic_assumptions: true
  notes: >
    Capacity values in this scenario are modelling assumptions for
    portfolio demonstration and testing. They are not observed
    Queensland Health capacity values.

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

The optimisation scenario loader validates these inputs before solving.

It also requires:

```text
synthetic_assumptions: true
```

for the current Phase 5 configuration boundary.

---

## 11. Optimisation diagnostics

After a feasible or optimal solve, the project derives descriptive diagnostics including:

- total hypothetical capacity available;
- total additional cases allocated;
- unused capacity;
- capacity utilisation;
- number of services considered;
- number of services receiving capacity;
- long-wait share;
- allocation share of waiting volume;
- waiting-volume ceiling utilisation; and
- service-capacity ceiling utilisation.

These diagnostics describe the model solution.

They do **not** represent measured post-intervention healthcare outcomes.

---

## 12. Optimisation outputs

A successful optimisation run creates scenario-specific outputs such as:

```text
reports/outputs/optimisation/
`-- baseline_capacity_recovery/
    |-- optimisation_summary.json
    |-- optimisation_allocations.csv
    `-- optimisation_diagnostics.json
```

### `optimisation_summary.json`

Contains model-level information such as:

```text
scenario_name
status
objective_value
total_capacity_available
total_additional_cases
unused_capacity
allocations
```

### `optimisation_allocations.csv`

Contains flat service-level decisions such as:

```text
facility_key
facility_name
service_key
service_name
resource_kind
reporting_period
additional_cases
```

### `optimisation_diagnostics.json`

Contains descriptive metrics explaining utilisation and constraint behaviour.

---

## 13. Repository structure

A simplified repository view is:

```text
qld-elective-surgery-optimiser/
|
|-- configs/
|   |-- base.yml
|   |-- facilities.yml
|   `-- scenarios/
|       |-- baseline.yml
|       `-- optimisation_baseline.yml
|
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   |-- quarantine/
|   `-- reference/
|
|-- reports/
|   `-- outputs/
|
|-- sql/
|   |-- create_warehouse.sql
|   `-- reconciliation.sql
|
|-- src/
|   `-- qld_surgery_optimiser/
|       |-- ingestion/
|       |-- optimisation/
|       |   |-- __init__.py
|       |   |-- constraints.py
|       |   |-- contract.py
|       |   |-- diagnostics.py
|       |   |-- models.py
|       |   |-- objective.py
|       |   |-- pipeline.py
|       |   |-- scenarios.py
|       |   `-- solver.py
|       |-- processing/
|       |-- validation/
|       |-- cli.py
|       |-- config.py
|       |-- exceptions.py
|       `-- logging_config.py
|
|-- tests/
|   |-- integration/
|   `-- unit/
|
|-- pyproject.toml
`-- README.md
```

The repository follows a `src/` package layout to keep package imports explicit and reduce accidental local-import behaviour.

---

## 14. Technology stack

### Core

- Python 3.12
- pandas
- DuckDB
- Parquet

### Configuration

- Pydantic
- Pydantic Settings
- YAML

### Optimisation

- Google OR-Tools CP-SAT

### Command line

- Typer

### Engineering quality

- pytest
- pytest-cov
- Ruff
- mypy
- pandas-stubs
- GitHub Actions
- structured logging

### Future application layer

Potential later work includes:

- FastAPI;
- Streamlit; and
- richer scenario-comparison interfaces.

---

## 15. Python version

The project targets:

```text
Python >=3.12,<3.13
```

Check the active interpreter with:

```bash
python --version
```

---

## 16. Local installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd qld-elective-surgery-optimiser
```

Create a Python 3.12 virtual environment.

On Windows:

```bat
py -3.12 -m venv virtual12
virtual12\Scripts\activate
```

On macOS or Linux:

```bash
python3.12 -m venv virtual12
source virtual12/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## 17. Command-line interface

The project exposes a Typer CLI.

Show available commands:

```bash
python -m qld_surgery_optimiser.cli --help
```

Current workflows include configuration checks, source discovery, ingestion, validation, warehouse construction and optimisation.

---

## 18. Configuration health check

Run:

```bash
python -m qld_surgery_optimiser.cli doctor
```

This validates local configuration and creates required directories where appropriate.

A successful `doctor` command confirms local configuration health.

It does **not** prove that the current live Queensland Government source pipeline has executed successfully.

---

## 19. Show resolved configuration

Run:

```bash
python -m qld_surgery_optimiser.cli show-config
```

An alternative planning scenario can be supplied using:

```bash
python -m qld_surgery_optimiser.cli show-config \
  --scenario configs/scenarios/baseline.yml
```

---

## 20. Discover CKAN resources

Run:

```bash
python -m qld_surgery_optimiser.cli discover
```

This queries the configured Queensland Government CKAN dataset and reports eligible resources.

Live source execution should still be treated as a separate verification step from mocked automated tests.

---

## 21. Ingest raw resources

Run:

```bash
python -m qld_surgery_optimiser.cli ingest
```

To request only resources selected by the current latest-resource logic:

```bash
python -m qld_surgery_optimiser.cli ingest --latest-only
```

The ingestion layer records source lineage and avoids casually overwriting raw source files.

---

## 22. Validate source data

Run:

```bash
python -m qld_surgery_optimiser.cli validate
```

Validation produces accepted intermediate data, quarantine data for invalid observations, and quality reporting.

---

## 23. Build the analytical warehouse

Run:

```bash
python -m qld_surgery_optimiser.cli warehouse
```

This constructs the canonical analytical outputs, longitudinal measures and DuckDB warehouse.

Typical Phase 4 outputs include:

```text
data/processed/canonical_performance.parquet
data/processed/longitudinal_performance.parquet
data/processed/<warehouse>.duckdb
reports/outputs/warehouse_reconciliation.json
reports/outputs/warehouse_build_summary.json
```

---

## 24. Run an optimisation scenario

A typical Phase 5 run is:

```bat
python -m qld_surgery_optimiser.cli optimise ^
  --analytical-data data\processed\longitudinal_performance.parquet ^
  --scenario configs\scenarios\optimisation_baseline.yml ^
  --output-dir reports\outputs\optimisation
```

On shells that use backslash continuation:

```bash
python -m qld_surgery_optimiser.cli optimise \
  --analytical-data data/processed/longitudinal_performance.parquet \
  --scenario configs/scenarios/optimisation_baseline.yml \
  --output-dir reports/outputs/optimisation
```

If no reporting period is supplied, the optimisation contract selects the latest available valid reporting period for the scenario's resource family.

An explicit period can be supplied:

```bat
python -m qld_surgery_optimiser.cli optimise ^
  --analytical-data data\processed\longitudinal_performance.parquet ^
  --scenario configs\scenarios\optimisation_baseline.yml ^
  --output-dir reports\outputs\optimisation ^
  --reporting-period 2025-06-01
```

The reporting period uses:

```text
YYYY-MM-DD
```

---

## 25. Optimisation input contract

Before any CP-SAT model is built, the analytical dataset is validated against a dedicated optimisation contract.

Required fields include:

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

The contract validates:

- required columns;
- source-family selection;
- reporting-period parsing;
- latest-period selection;
- facility/service uniqueness;
- text identifiers;
- non-negative whole-number volumes; and
- the invariant that long waits cannot exceed total waiting volume.

This provides a clear boundary between analytical data and optimisation logic.

---

## 26. Data validation philosophy

The project uses explicit validation rather than silently repairing every unexpected source value.

Examples of checks include:

- required columns;
- controlled null handling;
- numeric parsing;
- percentage parsing;
- reporting-period parsing;
- negative-volume checks;
- percentage-range checks;
- duplicate business keys;
- long waits exceeding total waiting volume; and
- reconciliation of long-wait component fields.

Invalid observations can be quarantined instead of silently dropped.

---

## 27. Quarantine design

Quarantined rows retain:

- original source observations;
- source path;
- source-family information;
- quality rule identifiers;
- severity;
- affected column;
- observed invalid value; and
- validation message.

This makes data-quality failure visible and auditable.

---

## 28. Facility identity resolution

Facility names are resolved deterministically using a reviewed reference file.

Expected alias fields include:

```text
alias_name
canonical_name
canonical_code
hhs
region
active
```

The current approach intentionally avoids unreviewed fuzzy matching.

When a reviewed alias does not exist, the observed identity can be retained rather than forcing an uncertain match.

---

## 29. Canonical analytical model

Validated source observations are transformed into a consistent analytical schema.

The canonical model standardises concepts such as:

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

Downstream analytical and optimisation code therefore does not need to depend directly on publisher-specific column names.

---

## 30. Longitudinal measures

The processing layer derives descriptive period-over-period measures.

### Previous waiting volume

```text
previous_vol_waiting
```

### Backlog change

```text
backlog_change =
    vol_waiting - previous_vol_waiting
```

### Previous long waits

```text
previous_vol_long_waits
```

### Long-wait change

```text
long_wait_change =
    vol_long_waits - previous_vol_long_waits
```

### Long-wait share

```text
long_wait_share =
    vol_long_waits / vol_waiting
```

### Treatment-to-waiting ratio

```text
treatment_to_waiting_ratio =
    vol_treated / vol_waiting
```

These measures are descriptive and should not be interpreted as causal effects.

---

## 31. DuckDB analytical warehouse

The analytical warehouse includes dimensions such as:

```text
dim_facility
dim_specialty
dim_urgency_category
dim_reporting_period
dim_source_resource
```

and facts such as:

```text
fact_elective_surgery_performance
fact_data_quality_event
```

Warehouse loading uses explicit named columns instead of positional `SELECT *` insertion.

This reduces the risk of silent schema-order mismatches.

---

## 32. Source lineage

The pipeline retains lineage attributes including:

```text
resource_id
source_url
retrieved_at
local_path
sha256
```

A deterministic source-resource key is also generated for warehouse use.

The goal is that an analytical observation can be traced back to the material from which it was derived.

---

## 33. Reconciliation

Warehouse construction creates reconciliation metadata to make pipeline completeness visible.

Measures include concepts such as:

```text
canonical_rows
longitudinal_rows
fact_rows
facility_rows
specialty_rows
urgency_category_rows
reporting_period_rows
source_resource_rows
quality_event_rows
duplicate_canonical_keys_removed
fact_matches_longitudinal
```

Successful file creation is therefore not treated as sufficient evidence of analytical correctness.

---

## 34. Automated testing

Run the complete suite with:

```bash
pytest
```

At the current local checkpoint:

```text
143 passed
```

The suite covers:

- configuration;
- CKAN client behaviour;
- downloading;
- manifest handling;
- type coercion;
- validation rules;
- quarantine construction;
- entity resolution;
- normalisation;
- longitudinal processing;
- ingestion integration;
- validation integration;
- warehouse integration;
- optimisation input contracts;
- scenario validation;
- optimisation constraints;
- objective construction;
- CP-SAT solver behaviour;
- optimisation diagnostics;
- optimisation pipeline integration; and
- optimisation CLI behaviour.

The current pytest run also reports three non-failing pandas warnings in optimisation-contract tests. They do not currently cause test failures but remain useful cleanup work.

---

## 35. Static-quality checks

Run Ruff:

```bash
ruff check .
```

Run mypy:

```bash
mypy src tests
```

The current local checkpoint has both quality gates passing.

A useful full development gate is:

```bash
pytest
ruff check .
mypy src tests
```

---

## 36. Testing philosophy

The project distinguishes automated engineering verification from live-source verification.

Passing tests demonstrate that:

- controlled transformations behave as expected;
- configured contracts are enforced;
- solver behaviour is reproducible for tested scenarios;
- outputs reconcile in controlled integration tests; and
- CLI orchestration behaves correctly in the test environment.

Passing tests do **not** by themselves demonstrate that the latest public CKAN resource structure has been processed successfully end to end.

---

## 37. Current development status

### Phase 1 — Architecture and project design

**Complete**

Established:

- project objective;
- analytical scope;
- system architecture;
- repository structure;
- technology choices;
- reproducibility principles;
- governance boundary; and
- phased delivery plan.

### Phase 2 — Source discovery and ingestion

**Implemented and covered by automated tests**

Includes:

- CKAN discovery;
- source-family classification;
- CSV downloading;
- retry behaviour;
- versioned raw storage;
- SHA-256 checksums;
- source lineage; and
- ingestion manifests.

Live end-to-end execution against the current public CKAN resources remains to be verified separately.

### Phase 3 — Validation and quarantine

**Implemented and covered by automated tests**

Includes:

- schema validation;
- controlled coercion;
- quality rules;
- quarantine handling; and
- quality reporting.

### Phase 4 — Canonical model and analytical warehouse

**Implemented and covered by automated tests**

Includes:

- canonical normalisation;
- deterministic facility resolution;
- longitudinal measures;
- analytical dimensions;
- analytical facts;
- processed Parquet outputs;
- DuckDB loading; and
- reconciliation.

### Phase 5 — Capacity and waitlist optimisation

**Implemented and covered by automated tests**

Includes:

- typed optimisation domain models;
- optimisation input contracts;
- dedicated scenario configuration;
- CP-SAT decision variables;
- capacity constraints;
- waitlist-recovery objective;
- deterministic solver configuration;
- allocation summaries;
- allocation diagnostics;
- persisted JSON and CSV results;
- pipeline orchestration;
- integration testing; and
- CLI execution.

### Phase 6 — Decision-support interface

**Not yet implemented**

Potential later work includes:

- FastAPI;
- Streamlit;
- interactive scenario configuration;
- allocation visualisation;
- scenario comparison;
- downloadable reports; and
- richer explainability views.

---

## 38. Verification boundary

The following claims are currently supported by local automated execution:

- the supported Python package imports successfully;
- configuration tests pass;
- mocked CKAN behaviour is tested;
- ingestion integration tests pass;
- validation integration tests pass;
- warehouse integration tests pass;
- optimisation contract tests pass;
- scenario validation tests pass;
- constraint tests pass;
- objective tests pass;
- CP-SAT solver tests pass;
- diagnostics tests pass;
- optimisation pipeline integration tests pass;
- CLI optimisation tests pass;
- the full pytest suite passes with 143 tests;
- Ruff passes; and
- mypy passes.

The following claim is deliberately **not** made:

> The current codebase has successfully processed the latest live Queensland elective surgery resources end to end.

That requires explicit live-source execution and inspection.

---

## 39. Known limitations

### Live CKAN verification

The current automated suite tests CKAN behaviour, but the complete current pipeline still needs to be executed against the live source catalogue.

### Latest-resource semantics

A resource with the most recent CKAN metadata timestamp is not necessarily the resource containing the latest healthcare reporting period.

Reporting-period-aware source selection should therefore be verified against the live catalogue.

### Aggregate-data limitation

The project operates on published aggregate data.

It cannot reconstruct patient-level distributions or make patient-specific treatment recommendations.

### Separate published source families

Specialty and urgency-category views are separate aggregates.

The project intentionally does not fabricate a joint specialty × urgency-category distribution.

### Synthetic capacity assumptions

The current additional-capacity assumptions are modelling inputs.

They are not observed Queensland Health operating capacity measurements.

### Optimisation objective

The baseline objective is designed for a transparent portfolio demonstration and technical decision-support workflow.

It should not be interpreted as an endorsed clinical or health-service prioritisation policy.

### Current pandas warnings

Three non-failing warnings remain in optimisation-contract test paths and should be cleaned up before considering the test output completely warning-free.

---

## 40. Data governance

The project uses aggregate public data.

It does not use:

- patient names;
- patient identifiers;
- individual medical histories;
- patient-level waiting-list records; or
- automated patient treatment recommendations.

The optimisation boundary is operational capacity planning at aggregate facility/service level.

---

## 41. Design principles

### Reproducibility over convenience

Raw inputs are versioned and hashed rather than casually overwritten.

### Explicit data-quality failures

Invalid observations are surfaced and quarantined.

### Deterministic identity handling

Facility resolution favours reviewed mappings over opaque fuzzy matching.

### Source lineage

Analytical data retains enough provenance to trace records back to source material.

### Configuration over hidden assumptions

Operational assumptions belong in scenario configuration rather than being buried in solver code.

### Synthetic assumptions remain explicit

Hypothetical capacity values are clearly distinguished from observed data.

### Separation of concerns

Observed analytics, synthetic assumptions, optimisation decisions and scenario diagnostics remain separate.

### Tests before claims

Automated test success is not presented as proof of successful live-source execution.

---

## 42. Example end-to-end local workflow

Once appropriate source data are available, the intended workflow is:

```text
1. validate local configuration
2. discover source metadata
3. ingest source resources
4. validate source observations
5. inspect quarantine and quality outputs
6. build canonical and longitudinal datasets
7. build the DuckDB warehouse
8. inspect reconciliation
9. run the optimisation scenario
10. inspect allocation and diagnostic outputs
11. run automated quality gates
```

Representative commands:

```bash
python -m qld_surgery_optimiser.cli doctor
python -m qld_surgery_optimiser.cli discover
python -m qld_surgery_optimiser.cli ingest
python -m qld_surgery_optimiser.cli validate
python -m qld_surgery_optimiser.cli warehouse
```

Then:

```bash
python -m qld_surgery_optimiser.cli optimise \
  --analytical-data data/processed/longitudinal_performance.parquet \
  --scenario configs/scenarios/optimisation_baseline.yml \
  --output-dir reports/outputs/optimisation
```

Finally:

```bash
pytest
ruff check .
mypy src tests
```

---

## 43. Development workflow

Before committing significant changes:

```bash
pytest
ruff check .
mypy src tests
```

Inspect repository changes:

```bash
git status
git diff --stat
git diff
```

Stage only the intended project files before committing.

---

## 44. Suggested next engineering milestone

The most important next milestone is **live-source verification**.

That should include:

1. run CKAN discovery against the current Queensland dataset;
2. inspect the available live resources;
3. verify source-family classification;
4. verify reporting-period semantics;
5. ingest a reproducible live snapshot;
6. inspect manifest lineage and hashes;
7. run validation;
8. inspect quarantine and quality reporting;
9. build canonical and longitudinal datasets;
10. build and reconcile the warehouse;
11. run a clearly labelled synthetic optimisation scenario against the verified snapshot;
12. inspect the generated allocation and diagnostic outputs; and
13. document any live-source adaptations required.

Only after those steps should the README claim successful current live-data execution.

---

## 45. Future roadmap

Possible later development includes:

### Live-source hardening

- semantic latest-period selection;
- live-source schema adaptation;
- reproducible source snapshots;
- stronger source drift detection.

### Scenario analysis

- multiple capacity scenarios;
- sensitivity analysis;
- alternative objective weights;
- alternative service ceilings;
- scenario comparison.

### Optimisation diagnostics

- richer binding-constraint reporting;
- sensitivity summaries;
- additional infeasibility explanations.

### Decision-support interface

- FastAPI service layer;
- Streamlit interface;
- allocation charts;
- scenario comparison;
- downloadable scenario results.

### Deployment

- Docker;
- CI/CD hardening;
- packaged execution workflows.

---

## 46. Portfolio signals

This repository is intended to demonstrate capability across several disciplines.

### Data engineering

- public API integration;
- immutable ingestion;
- checksums;
- source lineage;
- Parquet;
- DuckDB;
- analytical warehouse construction.

### Data quality

- typed coercion;
- explicit validation;
- quarantine;
- quality events;
- reconciliation.

### Analytics engineering

- canonical modelling;
- deterministic keys;
- dimensions and facts;
- longitudinal measures.

### Optimisation

- mathematical decision variables;
- explicit constraints;
- configurable objective functions;
- OR-Tools CP-SAT;
- deterministic solver configuration;
- allocation diagnostics.

### Software engineering

- typed Python;
- modular package design;
- configuration management;
- CLI design;
- exception handling;
- unit testing;
- integration testing;
- linting;
- static type checking.

### Technical governance

- distinction between observed data and synthetic assumptions;
- explicit verification boundaries;
- aggregate rather than patient-level decision support;
- reproducibility and auditability.

---

## 47. Project intent

This is not intended to be a claim that a simple optimisation model can solve elective surgery planning in practice.

It is a technical demonstration of how a public-sector capacity-allocation problem can be approached as a reproducible data product:

```text
source data
    -> validation
    -> governed analytical model
    -> optimisation contract
    -> explicit assumptions
    -> constrained optimisation
    -> diagnostics
    -> reproducible outputs
```

The central design principle is that optimisation should sit on top of a trustworthy and auditable data pipeline, not bypass one.

---

## 48. Current checkpoint

At the current development checkpoint:

```text
Python: 3.12
Automated tests: 143 passed
Ruff: passed
mypy: passed
Phase 1: complete
Phase 2: implemented
Phase 3: implemented
Phase 4: implemented
Phase 5: implemented
Phase 6: planned
Live CKAN end-to-end verification: pending
```

The repository therefore demonstrates a complete tested path from controlled analytical inputs through to a scenario-driven CP-SAT allocation workflow, while keeping the remaining live-data verification boundary explicit.