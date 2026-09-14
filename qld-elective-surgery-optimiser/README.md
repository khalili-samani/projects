# Queensland Elective Surgery Capacity and Waitlist Recovery Optimiser

A reproducible decision-support project for analysing Queensland elective surgery performance data and, in later phases, modelling how additional surgical capacity could be allocated across facilities, specialties and urgency categories to support waitlist recovery.

The project is designed as an end-to-end data and optimisation system rather than a standalone notebook. It includes source discovery, immutable ingestion, validation and quarantine, canonical data modelling, longitudinal analytics, a DuckDB analytical warehouse, automated tests and static-quality controls.

> **Development status:** Code is implemented through Phase 4. Configuration health checks, the automated pytest suite, Ruff linting and mypy static type checking are verified locally. Live end-to-end execution against the current Queensland Government CKAN resources is still pending.

---

## 1. Project objective

Queensland publishes aggregate elective surgery performance information across facilities, specialties and urgency categories. The operational question behind this project is:

> Given limited additional elective surgery capacity, where should that capacity be allocated to achieve the strongest waitlist-recovery outcome while respecting operational and policy constraints?

The completed phases build the trustworthy analytical foundation required to answer that question.

The planned optimisation layer will use these processed data together with explicitly labelled scenario assumptions to explore capacity-allocation strategies.

The project does **not** make patient-level clinical decisions.

---

## 2. What the project currently does

The implemented system can:

* discover Queensland elective surgery resources through the Queensland Government Open Data CKAN API;
* classify relevant resources into specialty and urgency-category source families;
* download source CSVs into immutable, checksum-versioned storage;
* record source lineage and SHA-256 hashes in an ingestion manifest;
* validate source schemas and controlled data types;
* quarantine invalid observations instead of silently discarding them;
* generate machine-readable data-quality reporting;
* normalise validated data into a canonical analytical model;
* deterministically resolve facility identities through reviewed aliases;
* construct longitudinal waitlist and treatment measures;
* build analytical dimensions and facts;
* persist processed Parquet datasets;
* load a DuckDB analytical warehouse;
* generate warehouse reconciliation metadata;
* test ingestion, validation, entity resolution, normalisation, longitudinal processing and warehouse construction;
* enforce linting and static type checking.

Optimisation, scenario analysis and the user-facing decision-support application are subsequent phases.

---

## 3. Why this project exists

Public healthcare datasets are often easy to download but considerably harder to use reliably.

A credible optimisation model depends on more than an objective function. It requires confidence in:

* source provenance;
* resource selection;
* schema consistency;
* type coercion;
* facility identity;
* reporting-period interpretation;
* duplicate handling;
* lineage;
* validation failures;
* longitudinal comparability; and
* reconciliation between source and analytical outputs.

This repository therefore treats data engineering and data quality as first-class parts of the optimisation problem.

The intention is to demonstrate how a public-sector analytical use case can be developed as a reproducible data product rather than as an opaque collection of notebook transformations.

---

## 4. Data sources

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

The source publishes aggregate elective surgery information including resource families broadly corresponding to:

* elective surgery by specialty; and
* elective surgery by urgency category.

The ingestion layer discovers resources through CKAN rather than relying on manually copied download URLs.

### Secondary contextual source

Australian Institute of Health and Welfare (AIHW) material may be used later for contextual interpretation and benchmarking.

AIHW data is not currently treated as the primary transactional input to the warehouse.

---

## 5. Data governance and modelling boundary

This project operates on aggregate public data.

It does not use:

* patient identifiers;
* patient-level records;
* individual clinical histories; or
* automated patient treatment recommendations.

Future optimisation inputs that are not directly available from published source data — for example hypothetical additional operating capacity — must be represented as **scenario assumptions** and clearly identified as synthetic inputs.

The project intentionally avoids fabricating observed healthcare results.

---

## 6. Current development status

### Phase 1 — Architecture and project design

**Complete**

Established:

* project objective;
* analytical scope;
* system architecture;
* repository structure;
* technology choices;
* reproducibility principles;
* data-governance boundary; and
* phased delivery plan.

### Phase 2 — Source discovery and ingestion

**Implemented and covered by automated tests**

Includes:

* CKAN source discovery;
* resource classification;
* CSV download handling;
* retry behaviour;
* raw file versioning;
* SHA-256 checksums;
* immutable raw-data paths;
* ingestion manifest creation; and
* source metadata retention.

Focused ingestion and CKAN tests are passing.

Live end-to-end execution against the current public CKAN resources remains to be verified.

### Phase 3 — Validation and quarantine

**Implemented and covered by automated tests**

Includes:

* required-column validation;
* controlled null handling;
* numeric parsing;
* percentage parsing;
* date parsing;
* negative-volume checks;
* percentage-range checks;
* long-wait consistency checks;
* long-wait component reconciliation warnings;
* duplicate business-key detection;
* valid-row output;
* quarantine output; and
* JSON data-quality summaries.

Invalid rows are quarantined with source observations and machine-readable quality-rule metadata.

### Phase 4 — Canonical model and analytical warehouse

**Implemented and covered by automated tests**

Includes:

* canonical source normalisation;
* facility identity handling;
* deterministic surrogate keys;
* longitudinal measures;
* analytical dimensions;
* performance facts;
* quality-event facts;
* processed Parquet outputs;
* DuckDB loading; and
* reconciliation reporting.

### Phase 5 — Capacity and waitlist optimisation

**Planned**

Expected work includes:

* explicit decision variables;
* scenario capacity parameters;
* objective-function design;
* policy and operational constraints;
* OR-Tools CP-SAT implementation;
* baseline and sensitivity scenarios;
* infeasibility diagnostics; and
* optimisation validation.

### Phase 6 — Decision-support interface

**Planned**

Potential components include:

* FastAPI service layer;
* Streamlit analytical interface;
* scenario configuration;
* allocation visualisation;
* before/after backlog comparison;
* explainability outputs; and
* scenario export.

---

## 7. System architecture

The intended flow is:

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
 Future optimisation layer
      |
      v
 Decision-support application
```

---

## 8. Repository structure

A simplified view of the repository is:

```text
qld-elective-surgery-optimiser/
|
|-- configs/
|   |-- base.yml
|   |-- facilities.yml
|   `-- scenarios/
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

The repository follows a `src/` package layout to reduce accidental local-import behaviour and keep packaging explicit.

---

## 9. Technology stack

### Core

* Python 3.12
* pandas
* DuckDB
* Parquet

### Configuration and validation

* Pydantic
* Pydantic Settings
* YAML configuration

### Optimisation

Planned:

* Google OR-Tools CP-SAT
* NumPy

### Application layer

Planned:

* FastAPI
* Streamlit

### Engineering quality

* pytest
* pytest-cov
* Ruff
* mypy
* pandas-stubs
* GitHub Actions
* structured logging

---

## 10. Python version

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

## 11. Local installation

### Clone the repository

```bash
git clone <repository-url>
cd qld-elective-surgery-optimiser
```

### Create a Python 3.12 virtual environment

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

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Install the package and development dependencies

```bash
pip install -e ".[dev]"
```

---

## 12. Configuration health check

The CLI includes a configuration health check.

Run:

```bash
python -m qld_surgery_optimiser.cli doctor
```

A healthy configuration should report that the project configuration can be loaded and required directories are available.

The health check validates local configuration and package wiring. It does **not** prove that the live CKAN pipeline has executed successfully.

---

## 13. Data ingestion design

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

Rather than repeatedly overwriting a source file, downloaded content is versioned using content-derived information so that the exact input used in a later analytical run can be identified.

This allows downstream outputs to retain a traceable relationship to their source resource.

---

## 14. CKAN resource discovery

The CKAN client calls the Queensland Government package metadata endpoint and classifies supported resources.

The configuration distinguishes source families using reviewed naming patterns for:

* category resources; and
* specialty resources.

Unsupported resource formats are excluded from the current CSV ingestion path.

Historical resources can also be retained depending on configuration.

### Known source-selection consideration

The current `latest_only` selection logic should not yet be treated as semantically verified against all live resources.

A future improvement is to derive reporting periods explicitly from source metadata or filenames rather than relying solely on CKAN metadata timestamps such as creation or modification dates.

This distinction matters because the most recently modified CKAN resource is not necessarily the most recent healthcare reporting period.

---

## 15. Validation philosophy

The project uses explicit validation rather than silently coercing all source values.

Raw values are initially treated conservatively and then converted through controlled rules.

Examples include:

### Parse failures

A non-null value that cannot be converted to its expected numeric, percentage or date type produces:

```text
PARSE_FAILURE
```

### Missing essential values

```text
MISSING_REQUIRED_VALUE
```

### Negative patient volumes

```text
NEGATIVE_VOLUME
```

### Percentage outside configured range

```text
PERCENTAGE_OUT_OF_RANGE
```

### Long waits exceeding the total waiting list

If:

```text
Vol_LongWaits > Vol_Waiting
```

the row receives:

```text
LONG_WAITS_EXCEED_WAITING
```

with error severity and is considered invalid.

### Long-wait component mismatch

Where component fields are available, the project compares:

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

The warning is intentionally distinct from the error condition where total long waits exceed total waiting volume.

### Duplicate business keys

```text
DUPLICATE_BUSINESS_KEY
```

Business-key definitions depend on the source family.

---

## 16. Quarantine model

Rows failing error-level quality rules are not silently dropped.

They are written to quarantine data with:

* the original source observation;
* original row index;
* source path;
* resource kind;
* triggered rule identifiers; and
* quality messages.

This provides an auditable boundary between accepted analytical data and rejected source observations.

---

## 17. Canonical analytical model

Validated specialty and category files are converted into a common canonical representation.

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

The model supports both currently observed and selected legacy publisher column names.

This canonical layer isolates downstream analytical code from publisher-specific source column naming.

---

## 18. Facility identity resolution

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

The current strategy intentionally avoids fuzzy facility matching.

If no reviewed alias exists, the observed source identity is retained rather than guessing a mapping.

This favours auditability over aggressive automated entity resolution.

An empty alias file containing only headers is valid.

---

## 19. Longitudinal measures

The processing layer adds descriptive measures across reporting periods.

Examples include:

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

Zero waiting denominators are treated as missing for ratio calculation rather than producing infinite values.

These measures are descriptive and should not be interpreted as causal effects.

---

## 20. Analytical warehouse

The DuckDB warehouse currently includes:

### Dimensions

```text
dim_facility
dim_specialty
dim_urgency_category
dim_reporting_period
dim_source_resource
```

### Facts

```text
fact_elective_surgery_performance
fact_data_quality_event
```

Warehouse loading uses explicit named columns rather than positional `SELECT *` insertion.

This reduces the risk of silent schema-position mismatches when DataFrame and DuckDB column orders differ.

---

## 21. Source lineage

Analytical records retain source lineage including:

* resource identifier;
* source URL;
* SHA-256 hash;
* retrieval timestamp; and
* validated source file.

A deterministic `source_resource_key` is also generated for warehouse use.

The objective is that an analytical observation can be traced back to the source material from which it was derived.

---

## 22. Reconciliation

Warehouse construction writes reconciliation metadata describing row counts and consistency between analytical stages.

Current reporting includes measures such as:

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

This is intended to make pipeline completeness visible instead of assuming successful writes imply correct analytical output.

---

## 23. Processed outputs

Typical Phase 4 outputs include:

```text
data/processed/canonical_performance.parquet
data/processed/longitudinal_performance.parquet
data/processed/<warehouse>.duckdb
reports/outputs/warehouse_reconciliation.json
reports/outputs/warehouse_build_summary.json
```

Validation can additionally produce:

```text
data/interim/
data/quarantine/
reports/outputs/data_quality_summary.json
```

Generated outputs should generally not be treated as hand-maintained source code.

---

## 24. Automated testing

Run the complete test suite with:

```bash
pytest
```

The suite currently covers areas including:

* configuration;
* CKAN discovery;
* downloading;
* manifest behaviour;
* coercion;
* validation rules;
* quarantine construction;
* facility entity resolution;
* canonical normalisation;
* longitudinal processing;
* ingestion integration;
* validation integration; and
* warehouse integration.

At the current verified checkpoint, the full local pytest suite passes.

---

## 25. Static-quality checks

### Ruff

Run:

```bash
ruff check .
```

Current verified state:

```text
All checks passed
```

### mypy

Run:

```bash
mypy src tests
```

Current verified state:

```text
Success: no issues found in 39 source files
```

These checks should be run together with pytest before significant commits or pull requests.

A useful local quality gate is:

```bash
pytest
ruff check .
mypy src tests
```

---

## 26. Design principles

### Reproducibility over convenience

Raw source inputs are versioned and hashed rather than overwritten casually.

### Explicit quality failures

Invalid observations are quarantined instead of silently repaired.

### Deterministic entity resolution

Facility aliases are reviewed explicitly rather than resolved with unreviewed fuzzy matching.

### Source lineage

Processed observations retain enough metadata to identify their source.

### Aggregate decision support

The optimisation boundary is facility/service-level capacity planning rather than individual patient prioritisation.

### Configuration over hard-coded assumptions

Source behaviour, validation rules and future optimisation scenarios should be configurable wherever practical.

### Synthetic assumptions remain explicit

Capacity scenarios or parameters not observed directly in public data must be labelled as assumptions.

### Tests before claims

A passing configuration health check, unit test or static-quality gate should not be described as live-source verification.

---

## 27. Known limitations and open engineering items

The repository is deliberately explicit about work that is not yet verified.

### Live CKAN verification

The automated CKAN behaviour is tested, but the current complete pipeline still needs to be executed against the live Queensland Government resources.

### Semantic latest-resource selection

Selecting the latest resource based only on CKAN creation/modification metadata may not identify the latest healthcare reporting period.

Reporting-period-aware resource selection should be implemented and tested against live metadata.

### Legacy formats

The current ingestion path is CSV-focused.

Some historical resources may use formats such as XLS and are not currently part of the standard ingestion path.

### Facility reference coverage

The project does not fabricate facility alias mappings.

The reference file requires reviewed mappings before richer canonical facility metadata such as HHS or region can be considered complete.

### Canonical schema enforcement

The project uses explicit canonical-processing logic, but further formal schema enforcement at the canonical boundary may be worthwhile.

### Reconciliation SQL

SQL reconciliation assets exist in the project, while the current warehouse build also produces reconciliation information programmatically. These approaches should be aligned as the warehouse matures.

### Metric interpretation

Published metrics may change over time.

Live-source verification may expose schema or semantic differences that require controlled updates to aliases or validation rules.

---

## 28. Planned optimisation model

The optimisation layer is intentionally downstream of the validated analytical foundation.

A future model may define allocation decisions such as:

```text
additional_cases[facility, specialty, urgency_category]
```

Possible objectives include weighted combinations of:

* waiting-list reduction;
* long-wait reduction;
* urgency-weighted backlog reduction;
* service-level recovery;
* equitable allocation; and
* efficient utilisation of incremental capacity.

Potential constraints include:

* total additional capacity;
* facility capacity limits;
* specialty capacity limits;
* urgency-category rules;
* minimum service guarantees;
* maximum feasible throughput changes;
* fairness constraints; and
* scenario-specific policy restrictions.

These assumptions will be explicit and configurable rather than presented as observed facts.

OR-Tools CP-SAT is the planned optimisation engine.

---

## 29. Planned scenario framework

Possible future scenarios include:

### Baseline

Current or reference capacity with no additional intervention.

### Constrained recovery

A fixed amount of additional activity is available.

### Long-wait priority

Greater objective weight is placed on reducing long waits.

### Urgency-sensitive recovery

Capacity is weighted toward urgency categories according to explicit scenario parameters.

### Balanced recovery

Capacity allocation balances backlog reduction, long-wait reduction and distributional considerations.

Scenario outputs should always distinguish:

* observed source metrics;
* derived analytical measures;
* model assumptions; and
* optimisation recommendations.

---

## 30. Planned decision-support outputs

The eventual application may expose outputs such as:

* recommended additional activity by facility;
* recommended activity by specialty;
* urgency-category allocation;
* estimated backlog movement under scenario assumptions;
* long-wait reduction;
* capacity utilisation;
* constraint utilisation;
* binding constraints;
* infeasibility diagnostics;
* scenario comparisons; and
* downloadable allocation tables.

Optimisation outputs will represent modelled scenarios, not guaranteed healthcare outcomes.

---

## 31. Development workflow

A typical engineering workflow is:

```text
1. discover source metadata
2. ingest raw resources
3. validate source observations
4. inspect quarantine and quality reporting
5. normalise accepted data
6. construct longitudinal measures
7. build warehouse
8. reconcile outputs
9. run automated quality gates
10. inspect changes
11. commit a known-good checkpoint
```

Recommended quality checks before committing:

```bash
pytest
ruff check .
mypy src tests
```

Then inspect Git changes:

```bash
git status
git diff --stat
git diff
```

---

## 32. Current verification boundary

The following claims are currently supported by local automated execution:

* the package imports successfully under the supported Python environment;
* configuration health checks work;
* automated pytest tests pass;
* CKAN behaviour is exercised through mocked tests;
* ingestion integration tests pass;
* validation integration tests pass;
* warehouse integration tests pass;
* Ruff static checks pass;
* mypy static type checks pass.

The following claim is **not yet made**:

> The current version has successfully processed the latest live Queensland elective surgery dataset end to end.

That verification is the next major engineering checkpoint.

---

## 33. Next steps

The immediate roadmap is:

1. run CKAN discovery against the current live Queensland dataset;
2. inspect the live resource catalogue and reporting-period naming;
3. improve semantic latest-period selection;
4. execute live raw ingestion;
5. inspect manifest lineage and checksums;
6. execute validation against real downloaded resources;
7. inspect quarantine and quality summaries;
8. build the live canonical and longitudinal datasets;
9. build the DuckDB warehouse;
10. review reconciliation outputs;
11. document any live-source schema adaptations;
12. establish a verified reproducible data snapshot;
13. begin Phase 5 optimisation design.

---

## 34. Reproducing the current engineering checks

From an activated Python 3.12 environment:

```bash
pip install -e ".[dev]"
```

Then:

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

A clean development checkpoint should have all three engineering gates passing.

---

## 35. Intended portfolio signals

This repository is intended to demonstrate capability across several disciplines.

### Data engineering

* API-based ingestion;
* immutable raw storage;
* checksums;
* lineage;
* Parquet;
* DuckDB;
* analytical modelling.

### Data quality

* controlled coercion;
* explicit validation;
* quarantine;
* rule-level reporting;
* reconciliation.

### Analytics engineering

* canonical data modelling;
* dimensions and facts;
* deterministic keys;
* longitudinal measures.

### Software engineering

* typed Python;
* modular package design;
* configuration management;
* exception handling;
* automated tests;
* linting;
* static typing;
* reproducible environments.

### Operations research

Planned:

* formal decision variables;
* objective functions;
* constraints;
* scenario analysis;
* sensitivity analysis;
* optimisation diagnostics.

### Technical product thinking

* clear decision problem;
* explicit modelling boundary;
* auditable assumptions;
* staged architecture;
* separation between observed data and modelled recommendations.

---

## 36. Project principles for healthcare analytics

Healthcare decision-support systems require careful communication.

This project therefore follows several principles:

1. **Do not fabricate observed results.**
2. **Do not silently repair invalid source records.**
3. **Do not present assumptions as source data.**
4. **Do not infer facility identities through uncontrolled fuzzy matching.**
5. **Do not present optimisation recommendations as guaranteed outcomes.**
6. **Do not make patient-level treatment decisions from aggregate public data.**
7. **Maintain source lineage wherever practical.**
8. **Separate warnings from invalidating quality errors.**
9. **Test analytical contracts before relying on them.**
10. **Document what has and has not been verified.**

---

## 37. Licence and source attribution

The underlying Queensland Government data is published under the licence identified by the source dataset, currently expected to be Creative Commons Attribution 4.0.

Users of this repository should independently confirm the applicable source licence and attribution requirements when redistributing source data or derived outputs.

The software licence for this repository should be defined separately in the repository's `LICENSE` file.

---

## 38. Disclaimer

This project is an analytical and engineering portfolio project.

It is not a clinical system, medical device, official Queensland Health planning tool or production healthcare allocation platform.

Any future optimisation results depend on:

* source-data quality;
* modelling assumptions;
* objective-function design;
* constraint definitions; and
* scenario parameters.

Model outputs should therefore be interpreted as decision-support scenarios rather than clinical or operational directives.