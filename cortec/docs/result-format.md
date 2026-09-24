# The result record

Every run of `cortec` and `cortec-hybrid` can be printed and exported in one layout, through
`cortec.report`. The printed form is a stage header, a block of named values, one or more tables,
the warnings, and a verdict. The exported form is one JSON record with a declared format version,
a Markdown rendering of the same content, one CSV of the named values, one CSV per table, and a
figure in the paper's style when matplotlib is installed. This document is the contract for the
JSON record. The other exports are derived from it.

## Producing a record

```python
from cortec import show, record, evaluate

rec = show(release, n_private_rows=len(private_df))      # Stage A: prints, returns the record
rec = show(gen, n_rows=len(synthetic))                    # Stage B: a Generator after generate()
rec = show(report, schema=schema.name)                    # Stage C: a BoundReport
rec = evaluate(schema, {"cortec": synthetic}, train=train_df, holdout=holdout_df)   # step 8
rec = show(correct(schema, private_df, synthetic_df, columns=("age",), epsilon=0.5))   # cortec-hybrid
files = rec.save("results/run-2026-09-24", "stage_c")   # {name: path}
```

`record(obj, ...)` builds the record without printing. `RunRecord.from_json(path)` reads one back.
Colour is used only on a terminal; `NO_COLOR` or `CORTEC_COLOR=0` disables it and `CORTEC_COLOR=1`
forces it. The exported files never carry colour.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `format` | string | `cortec-result/1`. A reader checks this first and refuses anything else |
| `tool` | string | `cortec` or `cortec-hybrid` |
| `version` | string | the package version that wrote the record |
| `stage` | string | `Stage A`, `Stage B`, `Stage C`, `Evaluation` or `Correction` |
| `title` | string | the stage's name in words: `Release`, `Generate`, `Utility transmission bound`, `Fidelity and utility beside real references`, `Relabel the target column` |
| `schema` | string | the schema's `name` |
| `created` | string | UTC time, ISO 8601, to the second |
| `values` | object | the named values of the run, `name: value`; numbers are numbers, booleans are booleans, text is text |
| `value_order` | array | the names in `values` in display order |
| `epsilon` | object | the budget the stage spent or inherited: `declared`, `accounted`, `per_row`, `per_person` for Stage A; `per_row: 0` for Stage B; `release`, `bound`, `per_row`, `per_person` for Stage C; `table`, `accounted`, `per_row` for the correction |
| `tables` | array | one object per table: `name`, `columns`, `rows` (arrays in column order), `reference_rows` (indices of rows that are references, such as a real sample or a permuted floor), `note`, `decimals` |
| `warnings` | array | the run's warnings, each a sentence; empty when there were none |
| `notes` | array | standing statements that travel with the stage, such as "This bounds utility. It is not a privacy audit." |
| `verdict` | string or null | the stage's one-sentence verdict, or null when the stage has none |
| `verdict_role` | string | `ok`, `warning` or `refusal`: how the verdict is to be read, and the colour it is printed in |
| `files` | object | the paths `save()` wrote, by name: `json`, `markdown`, `values_csv`, `table_csv:<table name>`, and `figure` (a path, or `not written: ...` when matplotlib is absent) |

Numbers are stored at full precision. The printed and Markdown forms show three decimal places,
the paper's convention, and integers with a thousands separator.

## The tables each stage writes

| Stage | Table | Columns |
|---|---|---|
| Stage A | `cohorts` | cohort, released size, positive rate, class-conditional |
| Stage A | `conditional table, level k` (the finest released level) | cell, rate, support |
| Stage A | `privacy accounting` | group, queries, partitions, epsilon |
| Stage B | `reasoning evidence` | measure, count |
| Stage C | `bound per condition` | condition, cells, covered, uncovered, thin, mean bound, worst bound, within tolerance; rows 1 and 2 are the real-sample ceiling and the permuted-target floor |
| Evaluation | `fidelity and utility` | table, rows, 1-way TV, TSTR-LR, TSTR-RF, TSTR-GBM, absent categories; the last two rows are the real sample and the permuted floor |
| Correction | `released conditional table` | cell, rate, support |

The evaluation table is the paper's Table 6 layout. A synthetic row is read against the two
reference rows, never against a threshold: the real sample is the ceiling at that size, and the
permuted floor is what data with no usable target information scores.

## Figures

`rec.save()` writes `<stem>.png` for Stage A (cohort rates and the finest conditional level),
Stage C (the bound per condition against the tolerance), the evaluation (fidelity and utility
beside the references) and the correction (conditional error before and after). Stage B has no
figure. The style is the paper's: `cortec.plots.paper_rcparams()` returns the same rc parameters
the paper's figure script uses, and the palette is the paper's validated categorical palette,
assigned in fixed order. Reference rows are drawn in grey, results in blue, and the tolerance in
the paper's reference-line colour.
