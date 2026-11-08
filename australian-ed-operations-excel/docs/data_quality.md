# Data quality

## Initial quality issues

The raw synthetic source contains deliberate issues so the project demonstrates practical data-quality work:

| Issue | Count |
|---|---:|
| Exact duplicate rows | 36 |
| Blank wait-to-clinician values | 68 |
| Invalid age values | 36 |
| Unmatched facility codes | 21 |
| Region labels with trailing whitespace | 59 |
| Non-standard sex labels | 82 |

## Cleaning decisions

- Exact duplicate source rows were excluded from the cleaned output.
- Blank waits were treated as unavailable rather than imputed.
- Invalid ages were flagged and reconciled to the underlying synthetic truth record for the supplied cleaned file.
- Unmatched facility codes were identified against the facility lookup.
- Whitespace was removed before category matching.
- `F` and `M` labels were mapped to `Female` and `Male`.
- High waiting times and long lengths of stay were retained when plausible; they were not automatically removed as outliers.

## Row-count reconciliation

- Raw rows: 12,115
- Cleaned unique encounters: 12,079
- Documented exact duplicates: 36
- Reconciliation: 12,079 + 36 = 12,115

## Remaining concerns

The dataset is synthetic and cannot validate against external clinical systems. Facility-level comparisons are not adjusted for acuity, transfer patterns, staffing, bed occupancy, socioeconomic status or population size. Data-quality checks confirm internal consistency, not real-world accuracy.
