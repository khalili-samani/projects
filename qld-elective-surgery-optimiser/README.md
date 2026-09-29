# Queensland Elective Surgery Capacity and Waitlist Recovery Optimiser

A reproducible data engineering, analytics and operations-research project for analysing Queensland elective surgery performance and modelling how hypothetical additional surgical capacity could be allocated across facilities and services to support waitlist recovery.

The repository is designed as an end-to-end decision-support system rather than a standalone notebook. It combines source discovery, immutable ingestion, validation and quarantine, canonical data modelling, longitudinal analytics, a DuckDB analytical warehouse, scenario configuration and an optimisation layer built around Google OR-Tools CP-SAT.

> **Development status:** Phases 1–4 are implemented. Phase 5 optimisation development is in progress, with the optimisation data contract, typed scenario framework, baseline capacity constraints and waitlist-recovery objective implemented or under active verification. The last confirmed project-wide checkpoint had pytest, Ruff and mypy passing. Live end-to-end execution against the current Queensland Government CKAN resources remains pending.

---

## 1. Project objective

Queensland publishes aggregate elective surgery performance information across facilities, specialties and urgency categories.

The operational question behind this project is:

> Given limited additional elective surgery capacity, where should that capacity be allocated to support waitlist recovery while respecting explicit operational constraints and modelling assumptions?

The project approaches that question in two stages.

First, it builds a trustworthy analytical foundation:

```text
public source data
        ↓
controlled ingestion
        ↓
validation and quarantine
        ↓
canonical modelling
        ↓
longitudinal analytics
        ↓
analytical warehouse
```

Second, it uses that analytical layer as input to a scenario-based optimisation model:

```text
validated analytical data
        +
synthetic capacity assumptions
        ↓
optimisation contract
        ↓
constraints + objective
        ↓
CP-SAT model
        ↓
allocation decisions
        ↓
diagnostics and scenario outputs
```

The project does **not** make patient-level clinical decisions.

---

## 2. What the project currently does

The implemented analytical system can:

- discover Queensland elective surgery resources through the Queensland Government Open Data CKAN API;
- classify resources into specialty and urgency-category source families;
- download supported CSV resources;
- preserve raw inputs in immutable, checksum-versioned storage;
- retain source lineage and SHA-256 hashes in an ingestion manifest;
- validate source structure and controlled data types;
- quarantine invalid observations instead of silently discarding them;
- generate machine-readable data-quality summaries;
- normalise validated data into a canonical analytical model;
- resolve facility identities deterministically;
- construct longitudinal waiting-list and treatment measures;
- build analytical dimensions and fact tables;
- persist processed datasets to Parquet;
- load a DuckDB analytical warehouse;
- generate warehouse reconciliation information;
- enforce automated testing, linting and static typing.

Phase 5 currently adds the foundations for:

- typed optimisation-domain models;
- an explicit optimisation input contract;
- scenario-driven synthetic capacity assumptions;
- deterministic scenario configuration loading;
- global incremental-capacity constraints;
- optional per-service allocation ceilings;
- optional waiting-volume ceilings;
- integer-scaled waitlist-recovery objective coefficients;
- additional prioritisation of services with higher long-wait shares.

The CP-SAT solver orchestration, diagnostics, optimisation pipeline and decision-support interface remain subsequent implementation steps.

---

## 3. Why this project exists

Public healthcare datasets can be straightforward to download but considerably harder to use reliably for decision support.

An optimisation result is only as credible as the analytical system beneath it.

A defensible model requires confidence in:

- source provenance;
- resource selection;
- schema consistency;
- type conversion;
- facility identity;
- reporting-period interpretation;
- duplicate handling;
- validation failures;
- lineage;
- longitudinal comparability;
- analytical reconciliation;
- optimisation input grain;
- assumptions;
- constraints; and
- objective interpretation.

This repository therefore treats data engineering, data quality and optimisation design as parts of the same system.

The intention is to demonstrate how a public-sector analytical use case can be developed as a reproducible software product rather than an opaque collection of notebook transformations.

---

## 4. Data sources

### Primary source

Queensland Government Open Data — Elective Surgery dataset.

CKAN API base:

```text
https://www.data.qld.gov.au/api/3/action
```

Dataset slug:

```text
elective-surgery
```

Known package identifier:

```text
d660d925-400f-42ba-8245-a08bfc18abf4
```

The source publishes aggregate elective surgery information including resource families broadly corresponding to:

- elective surgery by specialty; and
- elective surgery by urgency category.

The ingestion layer discovers resources through CKAN rather than depending on manually copied download URLs.

### Secondary contextual source

Australian Institute of Health and Welfare material may later be used for contextual interpretation and benchmarking.

AIHW data is not currently treated as the primary transactional input to the analytical warehouse or optimisation model.

---

## 5. Data governance and modelling boundary

This project operates on aggregate public data.

It does not use:

- patient identifiers;
- patient-level records;
- individual clinical histories;
- patient-level scheduling;
- individual treatment prioritisation; or
- automated clinical recommendations.

The optimisation layer operates at an aggregate facility/service level.

Inputs not directly observed in the source data — for example hypothetical incremental surgical capacity — are represented as explicit **scenario assumptions**.

The project deliberately distinguishes:

```text
observed source data
derived analytical metrics
synthetic scenario assumptions
optimisation decisions
modelled scenario outputs
```

Synthetic assumptions must not be presented as observed Queensland Health capacity values.

---

# 6. Development status

## Phase 1 — Architecture and project design

**Complete**

Established:

- project objective;
- analytical scope;
- system architecture;
- repository structure;
- technology choices;
- reproducibility principles;
- data-governance boundaries; and
- phased delivery plan.

---

## Phase 2 — Source discovery and ingestion

**Implemented and covered by automated tests**

Includes:

- CKAN source discovery;
- resource classification;
- CSV download handling;
- retry behaviour;
- raw-file versioning;
- SHA-256 checksums;
- immutable raw-data paths;
- ingestion manifest creation; and
- source metadata retention.

The CKAN behaviour is covered through automated tests.

Live end-to-end execution against the current public CKAN resources remains to be verified.

---

## Phase 3 — Validation and quarantine

**Implemented and covered by automated tests**

Includes:

- required-column validation;
- controlled null handling;
- numeric parsing;
- percentage parsing;
- date parsing;
- negative-volume validation;
- percentage-range validation;
- long-wait consistency checks;
- long-wait component reconciliation;
- duplicate business-key detection;
- valid-row output;
- quarantine output; and
- JSON data-quality summaries.

Rows failing error-level quality rules are quarantined rather than silently removed.

---

## Phase 4 — Canonical model and analytical warehouse

**Implemented and covered by automated tests**

Includes:

- canonical source normalisation;
- deterministic facility identity handling;
- surrogate keys;
- longitudinal analytical measures;
- dimensions;
- performance facts;
- quality-event facts;
- processed Parquet outputs;
- DuckDB warehouse loading; and
- reconciliation reporting.

---

## Phase 5 — Capacity and waitlist optimisation

**In progress**

The current Phase 5 implementation establishes the model boundary before introducing solver orchestration.

Implemented or currently being verified:

- optimisation package structure;
- typed optimisation input models;
- typed allocation-result models;
- optimisation input contract;
- facility × service optimisation grain;
- reporting-period selection;
- specialty/category source-family separation;
- baseline synthetic scenario configuration;
- typed YAML scenario loading;
- scenario validation;
- global capacity constraints;
- optional service-level capacity ceilings;
- optional waiting-volume ceilings;
- deterministic allocation keys;
- CP-SAT-compatible integer objective coefficients;
- waitlist-recovery objective;
- long-wait-share prioritisation; and
- unit tests for the optimisation contract, scenarios, constraints and objective.

Remaining Phase 5 work includes:

- CP-SAT solver orchestration;
- solver-result translation;
- diagnostics;
- infeasibility handling;
- optimisation pipeline orchestration;
- persisted scenario outputs;
- integration tests;
- CLI integration;
- sensitivity scenarios; and
- documentation of verified model behaviour.

---

## Phase 6 — Decision-support interface

**Planned**

Potential components include:

- FastAPI service layer;
- Streamlit analytical interface;
- scenario configuration;
- allocation visualisation;
- before/after scenario comparisons;
- explainability outputs;
- model diagnostics; and
- downloadable scenario outputs.

---

# 7. System architecture

The current target architecture is:

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
      +-----------------------------+
      |                             |
      v                             v
Observed analytical data     Synthetic scenario
                                    assumptions
      |                             |
      +--------------+--------------+
                     |
                     v
             Capacity constraints
                     +
            Recovery objective
                     |
                     v
              OR-Tools CP-SAT
                     |
                     v
          Allocation + diagnostics
                     |
                     v
       Decision-support application
```

---

# 8. Repository structure

A simplified view of the repository is:

```text
qld-elective-surgery-optimiser/
|
|-- configs/
|   |-- base.yml
|   |-- facilities.yml
|   `-- scenarios/
|       `-- baseline.yml
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
|   `-- ...
|
|-- src/
|   `-- qld_surgery_optimiser/
|       |-- ingestion/
|       |-- processing/
|       |-- validation/
|       |-- optimisation/
|       |   |-- __init__.py
|       |   |-- models.py
|       |   |-- contract.py
|       |   |-- scenarios.py
|       |   |-- constraints.py
|       |   `-- objective.py
|       |-- cli.py
|       |-- config.py
|       |-- exceptions.py
|       `-- logging_config.py
|
|-- tests/
|   |-- integration/
|   `-- unit/
|       |-- test_optimisation_contract.py
|       |-- test_optimisation_scenarios.py
|       |-- test_optimisation_constraints.py
|       `-- test_optimisation_objective.py
|
|-- pyproject.toml
`-- README.md
```

The repository follows a `src/` package layout to keep Python packaging explicit and reduce accidental import behaviour from the repository root.

---

# 9. Technology stack

## Data engineering and analytics

- Python 3.12
- pandas
- DuckDB
- Parquet

## Configuration and validation

- Pydantic
- Pydantic Settings
- YAML configuration

## Optimisation

- Google OR-Tools CP-SAT

## Application layer

Planned:

- FastAPI
- Streamlit

## Engineering quality

- pytest
- pytest-cov
- Ruff
- mypy
- pandas-stubs
- structured logging
- GitHub-based version control

---

# 10. Python version

The project targets:

```text
Python >=3.12,<3.13
```

Python 3.11 is intentionally outside the supported project constraint.

Check your interpreter with:

```bash
python --version
```

---

# 11. Local installation

## Clone the repository

```bash
git clone <repository-url>
cd qld-elective-surgery-optimiser
```

## Create a Python 3.12 virtual environment

Windows:

```bat
py -3.12 -m venv virtual12
virtual12\Scripts\activate
```

macOS or Linux:

```bash
python3.12 -m venv virtual12
source virtual12/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

---

# 12. Configuration health check

The CLI includes a configuration health check.

```bash
python -m qld_surgery_optimiser.cli doctor
```

The health check validates local configuration and package wiring.

It does **not** prove that the latest live CKAN resources have successfully passed through the complete pipeline.

---

# 13. Data ingestion design

The ingestion layer is designed around reproducibility and lineage.

For every downloaded resource, the project records information including:

```text
resource_id
source_url
retrieved_at
local_path
sha256
```

Raw data is treated as immutable.

Instead of casually overwriting previously downloaded files, source content is stored in a way that allows the exact input used for later analytical processing to be identified.

This allows downstream observations to retain traceability to the source material from which they were derived.

---

# 14. CKAN resource discovery

The CKAN client calls Queensland Government package metadata and classifies supported elective-surgery resources.

The configuration distinguishes source families using reviewed naming patterns for:

```text
specialty
category
```

Unsupported source formats are excluded from the current CSV ingestion path.

Historical resources may also be retained according to configuration.

## Latest-resource selection

A CKAN resource's creation or modification timestamp is not necessarily equivalent to the healthcare reporting period represented by its data.

For that reason, the project treats reporting-period semantics separately from generic CKAN modification metadata.

Further live-source verification remains necessary before `latest_only` behaviour should be considered semantically verified across the current public resource catalogue.

---

# 15. Validation philosophy

The project uses explicit validation instead of silently coercing every source value.

Raw values are converted through controlled rules and quality failures are represented explicitly.

## Parse failures

```text
PARSE_FAILURE
```

## Missing required values

```text
MISSING_REQUIRED_VALUE
```

## Negative patient volumes

```text
NEGATIVE_VOLUME
```

## Percentage outside the accepted range

```text
PERCENTAGE_OUT_OF_RANGE
```

## Long waits exceeding total waiting volume

If:

```text
Vol_LongWaits > Vol_Waiting
```

the record receives:

```text
LONG_WAITS_EXCEED_WAITING
```

with error severity.

## Long-wait component mismatch

Where component fields exist, the project compares:

```text
Vol_LongWaits_RFS + Vol_LongWaits_NRFS
```

with:

```text
Vol_LongWaits
```

A mismatch produces:

```text
LONG_WAIT_COMPONENT_MISMATCH
```

as a warning.

This warning is intentionally separate from the invalid condition where total long waits exceed total waiting volume.

## Duplicate business keys

```text
DUPLICATE_BUSINESS_KEY
```

Business-key definitions depend on the source family.

---

# 16. Quarantine model

Rows failing error-level quality rules are not silently discarded.

They are written to quarantine outputs with information such as:

- the original source observation;
- original row index;
- source path;
- resource kind;
- triggered quality-rule identifiers; and
- quality messages.

This creates an auditable boundary between accepted analytical observations and rejected source records.

---

# 17. Canonical analytical model

Validated specialty and category data are converted into a shared canonical representation.

Core fields include:

```text
record_id
resource_kind
source_resource_id
source_sha256
source_url
source_retrieved_at
source_file
facility_code
facility_name
report_month
service_code
service_name
vol_treated
pct_treated_in_time
pct_variation_treated_prior_year
vol_waiting
vol_long_waits
pct_waiting_in_time_total
data_last_update
vol_long_waits_rfs
vol_long_waits_nrfs
pct_waiting_in_time_rfs
```

The canonical model isolates downstream analytical code from publisher-specific source column naming.

---

# 18. Facility identity resolution

Facility identities are resolved deterministically through reviewed mappings.

Expected alias information includes fields such as:

```text
alias_name
canonical_name
canonical_code
hhs
region
active
```

The project intentionally avoids uncontrolled fuzzy matching.

If no reviewed alias exists, the observed source identity is retained rather than guessing a canonical facility.

This design favours auditability over aggressive entity resolution.

---

# 19. Longitudinal measures

The processing layer constructs descriptive measures across reporting periods.

## Previous waiting volume

```text
previous_vol_waiting
```

## Backlog change

```text
backlog_change =
    vol_waiting - previous_vol_waiting
```

## Previous long waits

```text
previous_vol_long_waits
```

## Long-wait change

```text
long_wait_change =
    vol_long_waits - previous_vol_long_waits
```

## Long-wait share

```text
long_wait_share =
    vol_long_waits / vol_waiting
```

## Treatment-to-waiting ratio

```text
treatment_to_waiting_ratio =
    vol_treated / vol_waiting
```

Zero waiting denominators are represented as missing for ratio calculations rather than generating infinite values.

These measures are descriptive and should not be interpreted as causal effects.

---

# 20. Analytical warehouse

The DuckDB analytical warehouse includes:

## Dimensions

```text
dim_facility
dim_specialty
dim_urgency_category
dim_reporting_period
dim_source_resource
```

## Facts

```text
fact_elective_surgery_performance
fact_data_quality_event
```

Warehouse loading uses explicit named columns rather than positional `SELECT *` insertion.

This reduces the risk of schema-position corruption if DataFrame and DuckDB column orders differ.

---

# 21. Source lineage

Analytical records retain source lineage including:

- source resource identifier;
- source URL;
- SHA-256 hash;
- retrieval timestamp; and
- source file.

A deterministic source-resource key is also generated for warehouse use.

The objective is for an analytical observation to remain traceable to its originating source resource.

---

# 22. Reconciliation

Warehouse construction produces reconciliation metadata that makes analytical completeness visible.

Measures include values such as:

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

Successful database writes alone are not treated as proof of analytical correctness.

---

# 23. Phase 5 optimisation design

The optimisation layer is intentionally downstream of the validated analytical warehouse.

It does not directly optimise unvalidated raw files.

## Optimisation grain

The current baseline model uses:

```text
facility_key × service_key
```

for one published resource family at a time.

The baseline scenario currently targets:

```text
resource_kind = specialty
```

Category-level optimisation is also supported by the domain contract.

The model does **not** currently create a synthetic:

```text
facility × specialty × urgency category
```

cube.

The specialty and urgency-category publications are separate aggregate views, so combining them into a joint distribution without supporting source data would introduce fabricated structure.

---

# 24. Optimisation input contract

The Phase 5 contract converts analytical data into typed optimisation rows.

Each row contains:

```text
facility_key
facility_name
service_key
service_name
resource_kind
reporting_period
vol_waiting
vol_long_waits
```

The contract enforces:

- required facility identity;
- required service identity;
- valid reporting periods;
- one source family at a time;
- non-negative waiting volume;
- non-negative long-wait volume;
- whole-number patient volumes;
- long waits not exceeding total waiting volume; and
- uniqueness of the facility/service optimisation grain.

If no reporting period is supplied, the contract selects the latest reporting period present in the selected analytical dataset.

---

# 25. Why capacity is not stored in the analytical input row

The public analytical data describes observed elective-surgery performance.

Hypothetical incremental capacity is different.

For that reason, fields such as:

```text
total_additional_cases
max_additional_cases_per_service
```

are not treated as observed warehouse metrics.

They belong to scenario configuration.

This is a deliberate separation between:

```text
what was observed
```

and:

```text
what the model assumes
```

---

# 26. Baseline optimisation scenario

The baseline scenario is configured in:

```text
configs/scenarios/baseline.yml
```

The current synthetic example contains assumptions equivalent to:

```yaml
scenario:
  name: baseline_capacity_recovery
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

The capacity values above are modelling assumptions for development and demonstration.

They are **not observed Queensland Health capacity values**.

---

# 27. Scenario validation

Scenario YAML is loaded into immutable typed configuration objects.

The scenario layer validates conditions including:

- required sections;
- scenario name;
- resource kind;
- synthetic-assumption declaration;
- non-negative total capacity;
- non-negative service capacity;
- non-negative objective weights;
- at least one positive objective weight;
- Boolean constraint switches;
- positive solver time limit;
- positive search-worker count; and
- non-negative random seed.

This allows invalid optimisation assumptions to fail before model construction.

---

# 28. Baseline capacity constraints

The current baseline model defines integer allocation decisions:

```text
additional_cases[facility, service]
```

subject to explicit constraints.

## Non-negative allocation

```text
additional_cases[i] >= 0
```

## Global additional-capacity ceiling

```text
Σ additional_cases[i]
    <= total_additional_cases
```

## Optional service-level ceiling

When enabled:

```text
additional_cases[i]
    <= default_max_additional_cases_per_service
```

## Optional waiting-volume ceiling

When enabled:

```text
additional_cases[i]
    <= vol_waiting[i]
```

The waiting-volume ceiling prevents a scenario from allocating more additional activity to a service than the observed waiting population represented in that optimisation row.

The constraint layer also verifies that solver decision variables correspond exactly to the supplied facility/service optimisation keys.

---

# 29. Baseline waitlist-recovery objective

The current objective prioritises:

1. general waitlist recovery; and
2. services with a higher share of long-wait patients.

For each facility/service row:

```text
long_wait_share =
    vol_long_waits / vol_waiting
```

CP-SAT uses integer coefficients, so the share is scaled:

```text
scaled_long_wait_share =
    floor(
        vol_long_waits
        × 1000
        / vol_waiting
    )
```

For zero waiting volume:

```text
scaled_long_wait_share = 0
```

The current per-case objective coefficient is:

```text
objective_coefficient =
    waiting_weight × 1000
    +
    long_wait_weight × scaled_long_wait_share
```

The model then maximises:

```text
Σ additional_cases[i]
  × objective_coefficient[i]
```

Using the baseline scenario:

```text
waiting_weight = 1
long_wait_weight = 3
```

a service with no long waits receives:

```text
1000
```

objective points per additional case.

A service where 50% of the waiting population is long-waiting receives:

```text
1000 + (3 × 500)
= 2500
```

objective points per additional case.

This makes the baseline objective relatively simple and explainable.

---

# 30. Why long-wait share is used

The current baseline objective uses relative long-wait burden rather than simply using raw long-wait count as the priority coefficient.

For example:

```text
Service A
waiting = 100
long waits = 50
long-wait share = 50%

Service B
waiting = 500
long waits = 100
long-wait share = 20%
```

A raw-count objective would tend to favour the larger service simply because it has more patients.

The share-based term instead captures the relative concentration of long waits.

This is not presented as the uniquely correct policy objective.

It is a transparent baseline formulation that can later be compared with alternatives through scenario and sensitivity analysis.

---

# 31. Current optimisation limitations

The Phase 5 model should currently be interpreted as a modelling framework, not a validated operational planning model.

Important limitations include:

### Synthetic capacity assumptions

The model does not currently possess verified operating-theatre, workforce, bed, equipment or service-specific incremental-capacity data.

Capacity values in the baseline scenario are synthetic.

### No patient-level urgency modelling

The current optimiser operates on aggregate service rows.

It does not schedule or rank individual patients.

### Separate source families

Specialty and urgency-category publications are not combined into an unsupported joint distribution.

### Static scenario representation

The baseline formulation does not yet model dynamic queue arrivals, stochastic demand, cancellations or capacity uncertainty.

### Simplified capacity

One allocated case is currently treated as one unit of incremental activity.

The model does not yet represent procedure duration, theatre minutes, staffing intensity, bed-days or specialty-specific resource consumption.

### Objective interpretation

A larger optimisation score means a solution better satisfies the configured mathematical objective.

It does not imply a guaranteed clinical or operational outcome.

---

# 32. Planned solver implementation

The next optimisation component is the CP-SAT solver layer.

Its intended responsibilities are:

```text
validated OptimisationInputRow objects
        ↓
decision-variable creation
        ↓
capacity constraints
        ↓
waitlist-recovery objective
        ↓
CpSolver
        ↓
typed AllocationDecision outputs
        ↓
OptimisationSummary
```

The solver layer should remain separate from:

- warehouse extraction;
- YAML parsing;
- scenario persistence;
- diagnostics; and
- user-interface code.

This separation keeps the optimisation engine independently testable.

---

# 33. Planned optimisation diagnostics

Future diagnostics are expected to include metrics such as:

```text
total scenario capacity
total allocated capacity
unused capacity
number of services receiving capacity
allocation by facility
allocation by service
baseline waiting volume
baseline long-wait volume
allocation concentration
binding service ceilings
binding waiting-volume ceilings
solver status
objective value
```

Diagnostics should distinguish clearly between:

```text
observed inputs
scenario assumptions
solver decisions
derived scenario metrics
```

---

# 34. Planned scenario analysis

Later scenarios may vary:

- total additional activity;
- service-level capacity ceilings;
- objective weights;
- long-wait prioritisation;
- source family;
- future facility-specific constraints; and
- future policy constraints.

Possible scenario types include:

## Capacity-constrained recovery

A fixed amount of additional activity is available.

## Long-wait priority

Greater objective weight is assigned to long-wait burden.

## General backlog recovery

Long-wait weighting is reduced or disabled.

## Sensitivity analysis

Capacity and objective parameters are varied systematically to determine whether allocation patterns are robust to modelling assumptions.

Scenario results should never disguise synthetic parameters as measured operational capacity.

---

# 35. Processed outputs

Typical Phase 4 outputs include files such as:

```text
data/processed/canonical_performance.parquet
data/processed/longitudinal_performance.parquet
data/processed/<warehouse>.duckdb
reports/outputs/warehouse_reconciliation.json
reports/outputs/warehouse_build_summary.json
```

Validation may additionally produce:

```text
data/interim/
data/quarantine/
reports/outputs/data_quality_summary.json
```

Later Phase 5 outputs are expected to include artefacts such as:

```text
data/processed/optimisation/allocations.parquet
reports/outputs/optimisation_summary.json
```

These Phase 5 persisted outputs should not be considered implemented until the optimisation pipeline is completed and verified.

---

# 36. Automated testing

Run the complete test suite with:

```bash
pytest
```

The project includes tests for areas such as:

- configuration;
- CKAN discovery;
- downloading;
- manifest behaviour;
- coercion;
- validation rules;
- quarantine construction;
- facility identity resolution;
- canonical normalisation;
- longitudinal processing;
- ingestion integration;
- validation integration;
- warehouse integration;
- optimisation input contracts;
- optimisation scenario validation;
- optimisation constraints; and
- optimisation objective behaviour.

The last confirmed full-project checkpoint before the current Phase 5 additions had the complete local pytest suite passing.

New Phase 5 changes should be revalidated through the complete quality gate before their verification status is promoted in this README.

---

# 37. Static-quality checks

## Ruff

Run:

```bash
ruff check .
```

## mypy

Run:

```bash
mypy src tests
```

## Full local quality gate

```bash
pytest
ruff check .
mypy src tests
```

All three should pass before a Phase 5 implementation milestone is treated as a known-good checkpoint.

---

# 38. Development workflow

A typical workflow is:

```text
1. make one coherent implementation change
2. add or update focused tests
3. run the focused test
4. run the complete pytest suite
5. run Ruff
6. run mypy
7. inspect Git changes
8. stage only related files
9. review the staged diff
10. create a focused commit
```

Recommended checks:

```bash
pytest
ruff check .
mypy src tests
```

Then inspect Git:

```bash
git status
git diff --stat
git diff
```

Before committing:

```bash
git diff --cached
```

---

# 39. Phase 5 commit structure

The Phase 5 implementation is intentionally being developed through focused commits.

The intended history includes:

```text
Create Phase 5 optimisation package
Define optimisation domain models
Define optimisation input data contract
Add synthetic baseline optimisation scenario
Add typed optimisation scenario loading
Test optimisation input contract
Test optimisation scenario validation
Implement baseline capacity allocation constraints
Test baseline optimisation constraints
Implement baseline waitlist recovery objective
Test waitlist recovery objective behaviour
Implement CP-SAT capacity allocation solver
Test CP-SAT capacity allocation solver
Add optimisation diagnostics and allocation metrics
Add Phase 5 optimisation pipeline
Add optimisation pipeline integration test
Expose optimisation pipeline through CLI
Document Phase 5 optimisation model
```

This creates a reviewable progression from model definition to solver implementation rather than introducing the entire optimisation layer in one large change.

---

# 40. Design principles

## Reproducibility over convenience

Source inputs are versioned and hashed rather than casually overwritten.

## Explicit quality failures

Invalid observations are quarantined instead of silently repaired.

## Deterministic entity resolution

Facility mappings require reviewed aliases rather than uncontrolled fuzzy matching.

## Source lineage

Processed observations retain metadata connecting them to their source.

## Aggregate decision support

The optimisation boundary is aggregate facility/service capacity planning rather than individual patient prioritisation.

## Configuration over hidden assumptions

Source behaviour, validation rules and optimisation scenarios should be configurable where practical.

## Synthetic assumptions remain explicit

Capacity parameters not observed in public source data are labelled as modelling assumptions.

## One published source family at a time

Specialty and urgency-category datasets are not combined into a fabricated joint distribution.

## Typed optimisation boundaries

Invalid data or assumptions should fail before reaching the solver.

## Explainability before complexity

The first objective and constraint set is deliberately simple enough to describe mathematically and audit.

## Tests before claims

Passing unit tests or static checks must not be presented as live-source or operational validation.

---

# 41. Known limitations and open engineering items

## Live CKAN verification

The current complete pipeline still needs to be executed and inspected against the latest live Queensland Government resources.

## Semantic latest-resource selection

CKAN modification timestamps may not correspond to the most recent healthcare reporting period.

Reporting-period-aware resource selection should be verified against live metadata.

## Legacy formats

The current ingestion path is primarily CSV-focused.

Some historical resources may use other formats.

## Facility reference coverage

The project does not fabricate facility alias mappings.

Reviewed mappings are required before richer canonical facility metadata can be considered complete.

## Canonical schema enforcement

The processing layer uses explicit transformations and validations, but stronger formal schema enforcement at the canonical boundary may be added later.

## Reconciliation implementation

Programmatic reconciliation and any SQL reconciliation assets should remain aligned as the warehouse evolves.

## Published metric semantics

Source schemas and metric meanings may change over time.

Live-source verification may therefore require controlled updates to aliases, parsing or validation rules.

## Capacity realism

The current Phase 5 scenario uses hypothetical case counts rather than independently verified operational capacity.

## Resource consumption

Different procedures consume different combinations of theatre time, staff, beds and supporting resources. The current baseline model does not yet represent those differences.

## Dynamic queues

Waiting lists evolve through additions, removals, transfers and treatments. The current baseline optimisation is not yet a dynamic queueing model.

---

# 42. Current verification boundary

The following capabilities were verified at the last known-good Phase 4 checkpoint:

- package imports under the supported Python environment;
- configuration health checking;
- automated pytest execution;
- mocked CKAN behaviour;
- ingestion integration;
- validation integration;
- warehouse integration;
- Ruff linting; and
- mypy static type checking.

Phase 5 code is currently being added incrementally.

Until the complete quality gate is rerun successfully after the current optimisation additions, the README should not imply that every new Phase 5 component has passed full-project verification.

The project also does **not yet claim**:

> The current version has successfully processed the latest live Queensland elective surgery dataset end to end.

Nor does it yet claim:

> The optimisation outputs represent verified Queensland Health operational capacity recommendations.

Those are substantially stronger claims and require additional evidence.

---

# 43. Immediate roadmap

The current Phase 5 implementation sequence is:

1. finalise objective tests;
2. implement the CP-SAT solver;
3. test deterministic solver behaviour;
4. implement optimisation diagnostics;
5. implement the optimisation pipeline;
6. persist allocation and summary outputs;
7. add optimisation integration tests;
8. expose the pipeline through the CLI;
9. run the full Phase 5 quality gate;
10. document verified optimisation behaviour.

Separate live-data work remains:

1. run current CKAN discovery;
2. inspect the live resource catalogue;
3. verify reporting-period semantics;
4. execute live raw ingestion;
5. inspect manifest lineage;
6. execute validation against downloaded resources;
7. inspect quarantine and quality reports;
8. construct live canonical and longitudinal datasets;
9. rebuild the DuckDB warehouse;
10. inspect reconciliation outputs; and
11. record any required source-schema adaptations.

---

# 44. Reproducing the engineering checks

From an activated Python 3.12 environment:

```bash
pip install -e ".[dev]"
```

Run the configuration check:

```bash
python -m qld_surgery_optimiser.cli doctor
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Run static typing:

```bash
mypy src tests
```

A known-good development checkpoint should have all relevant engineering gates passing.

---

# 45. Intended portfolio signals

This repository is intended to demonstrate capability across multiple disciplines.

## Data engineering

- API-based ingestion;
- immutable raw storage;
- checksums;
- provenance;
- Parquet;
- DuckDB;
- analytical modelling.

## Data quality

- controlled coercion;
- explicit validation;
- quarantine;
- rule-level reporting;
- reconciliation.

## Analytics engineering

- canonical modelling;
- dimensions and facts;
- deterministic identifiers;
- longitudinal measures.

## Software engineering

- typed Python;
- modular package design;
- configuration management;
- exception handling;
- automated tests;
- linting;
- static typing;
- reproducible environments.

## Operations research

- explicit decision variables;
- objective-function design;
- capacity constraints;
- scenario parameters;
- CP-SAT-compatible formulation;
- planned diagnostics;
- planned sensitivity analysis.

## Technical product thinking

- a defined decision problem;
- explicit system boundaries;
- auditable assumptions;
- staged architecture;
- separation of source data from scenario assumptions;
- separation of analytical outputs from model decisions;
- cautious communication of verification status.

---

# 46. Healthcare analytics principles

Healthcare decision-support systems require careful communication.

This project follows several principles:

1. **Do not fabricate observed results.**
2. **Do not silently repair invalid source records.**
3. **Do not present assumptions as source data.**
4. **Do not infer facility identities through uncontrolled fuzzy matching.**
5. **Do not manufacture unsupported combinations of published aggregate datasets.**
6. **Do not present optimisation recommendations as guaranteed outcomes.**
7. **Do not make patient-level treatment decisions from aggregate public data.**
8. **Maintain source lineage wherever practical.**
9. **Separate warnings from invalidating quality errors.**
10. **Validate optimisation inputs before model construction.**
11. **Keep mathematical assumptions configurable and visible.**
12. **Document what has and has not been verified.**

---

# 47. Licence and source attribution

The underlying Queensland Government data is published under the licence identified by the source dataset and is currently expected to use Creative Commons Attribution 4.0.

Users of this repository should independently confirm the applicable source licence and attribution requirements before redistributing source data or derived outputs.

The software licence for this repository should be defined separately in the repository's `LICENSE` file.

---

# 48. Disclaimer

This project is an analytical, data-engineering and optimisation portfolio project.

It is not:

- a clinical system;
- a medical device;
- an official Queensland Health planning tool;
- an operational hospital scheduling system; or
- a production healthcare resource-allocation platform.

Future optimisation results depend on:

- source-data quality;
- analytical transformations;
- reporting-period selection;
- synthetic scenario assumptions;
- capacity definitions;
- objective-function design;
- constraint definitions; and
- solver configuration.

Model outputs should therefore be interpreted as **decision-support scenarios under explicit assumptions**, not clinical recommendations or guaranteed operational outcomes.