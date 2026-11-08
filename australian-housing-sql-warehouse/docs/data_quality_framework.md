# Data quality framework

## Dimensions and rules

| Dimension | Example rule | Severity | Handling |
|---|---|---:|---|
| Completeness | `listing_id` and parseable `sale_date` required for fact loading | Critical | Reject |
| Validity | `year_built` between 1800 and sale year | High | Set null and flag |
| Conformity | State mapped to an official abbreviation | High | Reject when unresolved |
| Consistency | Postcode first digit broadly consistent with state | Medium | Retain and flag |
| Uniqueness | Exact source-row hash unique within a batch | Medium | Keep first, classify duplicate |
| Plausibility | Sale price between $50,000 and $50,000,000 | High | Set null; reject if no usable price |
| Timeliness | Sale date not after batch load date | High | Reject |

## Quality score

Each standardised row starts at 100 and loses weighted points:

- 30: missing listing identifier;
- 25: unparseable sale date;
- 20: invalid or missing usable price;
- 15: unresolved state;
- 10: invalid postcode/state relationship;
- 10: invalid construction year;
- 5: unresolved property type;
- 5: malformed boolean or numeric fields.

The score supports prioritisation. It does not replace rule-level evidence.

## Disposition

- **accepted**: critical fields valid and row selected by survivorship.
- **duplicate_exact**: same canonical row hash as another row in the batch.
- **duplicate_variant**: repeated listing identity with changed attributes.
- **rejected**: critical fields cannot be resolved safely.
- **accepted_with_warning**: loaded with non-critical defects retained as flags.

## Test policy

Critical structural tests must return zero failures. Profiling tests may return observations but must be reviewed. Thresholds are explicit in SQL so reviewers can distinguish hard failures from monitored drift.
