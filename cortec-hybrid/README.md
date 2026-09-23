# cortec-hybrid

Conditional-signal correction for marginal DP synthesisers (AIM, MST, PrivBayes). No LLM is
required; the heaviest dependency is scikit-learn.

**It does not generate synthetic data.** The name is misleading, and this is the clarification that
matters most. `cortec-hybrid` contains no synthesiser. It never calls AIM, MST or anything else, it
will not produce a synthetic dataset for you, and it has no dependency on smartnoise-synth. It is a
post-processor: it releases a DP conditional target table from your private data and rewrites only
the target column of synthetic data you already generated, so that each cell's rate matches the
table. Every feature column comes out byte-identical to what you put in; the tool verifies this and
the test suite asserts it.

```
your private data ──► DP conditional table ──┐
                                             ├──► relabel target column only ──► corrected output
your AIM/MST output ─────────────────────────┘
```

| Step | What you do |
|---|---|
| 1. Install | Install `cortec` first, then this package |
| 2. What you need | Your private table, your synthesiser's output, and a public schema |
| 3. Run the correction | One call, or the four steps it is made of |
| 4. Read the result | What `worth_it` measures, and what it does not |
| 5. Before you run: feasibility | Why `check_feasible()` refuses some schemas |
| 6. Privacy | The ledger, `n_min`, and the privacy unit |
| 7. Why rate matching alone | The measurement that turned rank preservation off |

## 1. Install

`cortec-hybrid` reuses `cortec`'s audited privacy accounting, so install `cortec` first (its
`Schema` and `PrivacyLedger` are all it needs; no vendor SDK is required). Python 3.10 or later;
the packages are not on PyPI:

```bash
git clone https://github.com/Calyie/cortec-framework
cd cortec-framework
pip install ./cortec
pip install ./cortec-hybrid
python3 -m pytest cortec-hybrid/tests -q     # 26 tests, offline
```

## 2. What you need

1. **Your private data**, one pandas DataFrame with the columns the schema declares. One row per
   person; section 6 explains why.
2. **Synthetic data you already generated** with whatever DP synthesiser you use (AIM, MST,
   PrivBayes, a vendor product), with the same columns. This tool does not produce it.
3. **A `Schema`** from the `cortec` package: the column list, public bounds, public bin edges, the
   target and its two labels, declared from public knowledge only. The README of `cortec` (step 4)
   explains each field.

## 3. Run the correction

The one-call form:

```python
from cortec import Schema
from cortec_hybrid import check_feasible, correct

check_feasible(schema, n_rows=len(private_df))   # refuses BEFORE you burn hours on a schema AIM cannot fit

# `synthetic_df` is YOUR synthesiser's output; this tool does not produce it.
corrected, table, gain = correct(
    schema, private_df, synthetic_df,
    columns=("admission_type", "a1c_result"),   # the cells: richer is better and costs no extra ε
    epsilon=0.5,
)
print(gain["verdict"])
# -> "the correction materially improved conditional calibration"
#    or "the correction changed little; ... consider keeping the budget instead"
```

| Parameter of `correct()` | Default | What it does |
|---|---|---|
| `columns` | required | the columns whose cells the table is released over; numeric columns are binned by the schema's public edges, categoricals by declared value or by the schema's `coarsen` map |
| `epsilon` | required | the budget for the table; cells partition the data, so the whole table costs this once, however many cells it holds |
| `n_min` | `150` | cells with fewer private records are not released; the floor is 50 |
| `seed` | `None` | tests only: a seeded release is reproducible and carries no guarantee |

`correct()` is four steps, each available on its own:

1. `check_feasible(schema, n_rows=...)` refuses a schema AIM is unlikely to converge on (section 5).
2. `release_conditional_table(schema, private_df, columns, epsilon=..., n_min=...,
   max_rows_per_person=...)` spends the budget and returns a `ConditionalTable` with `cells`
   (rate per cell), `support` (noised size per cell), `epsilon` and `audit`. Store it; it contains
   no private record.
3. `relabel(schema, synthetic_df, table)` rewrites the target column so each cell matches its
   released rate, with stochastic rounding of the count and i.i.d. assignment within the cell. It
   is pure post-processing of two already-DP artefacts and costs nothing beyond the table.
4. `estimate_gain(schema, synthetic_df, corrected, table)` measures what changed, using released
   quantities only.

## 4. Read the result

`estimate_gain()` returns a dictionary:

| Key | Meaning |
|---|---|
| `conditional_error_before`, `conditional_error_after`, `conditional_error_reduction` | mean absolute gap between each cell's rate in the data and the released rate, before and after |
| `feature_marginal_shift` | mean total-variation change of the categorical marginals between input and output (it should be zero, since features are untouched) |
| `n_cells_corrected` | how many released cells were matched |
| `worth_it`, `verdict` | true when conditional error fell by more than 0.02, with a sentence saying so |
| `worth_it_measures`, `worth_it_does_not_predict` | what the criterion is, and what it is not |

**Read `worth_it` for what it is.** It is a calibration criterion, meaning that conditional error
against the released table fell by more than 0.02, and it does not predict downstream utility. We
measured the two coming apart. Across 11 datasets the correction improved calibration in essentially
every case while downstream AUC moved in both directions, and of 9 runs flagged `worth_it` the
downstream AUC fell on 6 in one run and on 3 of 10 in a repeat. Three datasets, a renal registry,
hospital readmission and cervical cancer, were reliably negative: calibration reliably improved and
downstream AUC reliably dropped. `worth_it_does_not_predict` says exactly this. Validate on your
own held-out data before adopting the correction, and keep your budget if the gain is small.

**What to expect.** Marginal-based DP synthesisers are excellent at the statistic they optimise,
and they are evaluated on it in their own papers. Neither AIM's nor MST's paper reports a
downstream predictive-utility experiment, and in our measurements that omission is consequential:
at matched budget and matched sample size on UCI Adult, models trained on their output reached
0.675 to 0.728 AUC where a real sample of the same size reached 0.834 to 0.872 (technical report
§7.2). This tool spends a small, separately accounted slice of budget on the one thing those
mechanisms do not target, a conditional target table `P(y | cell)` over a disjoint partition, and
relabels the output to match it. Because the cells partition the data, the whole table costs one
query's worth of ε however many cells it holds: accuracy is set by cell support, not cell count,
which is what makes a rich table affordable. It is not "CoRTeC without the LLM", and it is not a
general improvement over its inputs. A coarse 12-cell table, the obvious implementation, fixes
calibration and adds little utility in our runs (+0.025 AUC on AIM and +0.037 on MST, on Adult);
the large gain arrives only with a richer table (§8.2). Use it in one specific situation: your
marginal synthesiser's output has weak conditional target structure and you need a calibrated rate.

## 5. Before you run: the feasibility check

`check_feasible()` refuses a schema AIM is unlikely to converge on, so you learn in the first second
rather than the third hour. Pass `n_rows=len(private_df)`, because the row count is one of the
three factors it uses, and `raise_on_fail=False` to receive the report and proceed anyway.

It used to gate on domain size, and our own ablation refuted that. UCI Adult has the largest domain
of every schema we measured, 4.4 × 10¹³, six million times larger than a healthcare prefix that
times out, and Adult is the one that fits. Within a single dataset, the arm with the larger domain
fit 425 seconds faster than the arm with the smaller one. Domain volume does not predict
convergence, and gating on it refuses the wrong schemas. Three factors did predict it, each
demonstrated by a controlled flip:

- **attribute count**: a 19-attribute schema times out at a row count where a 15-attribute one fits;
- **row count**: the same schema at the same domain moves from a 26-minute fit to a timeout purely
  by going from 26,048 to 81,410 rows;
- **high-cardinality correlated numeric columns**: one dataset has Adult's width, fewer rows and a
  30× smaller domain, and it still fails. Dropping any three columns does not rescue it; dropping
  three amount columns does. A width-matched control (dropping three demographics instead) still
  times out, which is the only way to separate those two causes.

The report includes the full observation table it is derived from, and the estimated domain size,
reported for reference and explicitly not used to decide. This is a heuristic from ten fits under
one AIM implementation, not a theory.

## 6. Privacy

The tool reuses `cortec`'s audited `PrivacyLedger`, so the composition rules are implemented once
and audited once. `table.audit` is the same machine-readable trail: every query with its epsilon,
sensitivity, composition rule and partition key. The cell sizes it publishes are noised and
charged, in their own group, beside the rates. Validation runs before any budget is spent, and
`n_min` has a hard floor of 50 because a rate over fewer records is dominated by its own Laplace
noise.

**ε protects one ROW, and this tool is the one most likely to be pointed at data where that is not
one person.** It corrects a synthesiser you already run, so it gets aimed at the table your
institution already has, including encounter-level clinical data, where one patient contributes
many rows. Under group privacy a person contributing `k` rows receives `k · ε`: on a real hospital
dataset whose heaviest patient has 40 encounters, a declared ε of 2.0 is an ε of 80 for that
patient, which carries no meaningful guarantee.

Declare the privacy unit with `max_rows_per_person`. Where you declare one whose ε per person is
vacuous, the release is refused, not annotated:

```python
# refused: eps_person = 80
release_conditional_table(schema, df, ("grp",), epsilon=2.0, max_rows_per_person=40)

# proceed with a ROW-level guarantee, acknowledgement recorded in the audit trail
tbl = release_conditional_table(schema, df, ("grp",), epsilon=2.0, max_rows_per_person=40,
                                acknowledge_vacuous_privacy_unit=True)
tbl.audit["epsilon_per_person"]          # -> 80.0
tbl.audit["epsilon_per_person_vacuous"]  # -> True
```

Leave it out and the audit records `"UNDECLARED"` rather than assuming 1. If your data has more
than one row per person, aggregate before releasing, or report `ε_person` rather than the per-row
number.

## 7. Why rate matching alone

We previously held that rank-preserving assignment, ordering the rows within a cell by a model of
the synthetic data's own target, was necessary for this correction to pay for itself. Testing this
package disproved that in one regime, and the code changed as a result: `preserve_ranking=False` is
the default of `relabel()`.

Rank preservation only helps when the input synthetic data already carries within-cell structure.
When its target is close to noise, precisely the case this tool targets, a ranking model fitted on
that target learns the noise, and ordering by it injects spurious structure that a downstream model
mistakes for signal. In our fixture that cost 0.09 AUC against plain i.i.d. assignment.

The guard we first wrote for this did not work, and why is worth knowing. `_rank_scores` gated on
whether the synthetic target is predictable from the synthetic features (cross-validated AUC
against `RANK_INFORMATIVENESS_FLOOR = 0.55`). But a synthesiser generates its target as a function
of those features, so that score is near-perfect almost always: 0.981 on one dataset whose real
downstream AUC is 0.631. It measures self-consistency, not validity, and it is highest exactly when
the synthesiser has confidently learned a wrong relationship. It therefore selected for the failure
mode it was built to prevent.

Replicated over three datasets at three synthesiser seeds each, i.i.d. assignment scored higher
than rank preservation in 9 of 9 runs:

| dataset | base synthesiser | + rank-preserving | + i.i.d. within cell |
|---|---|---|---|
| renal registry | 0.631 | 0.546 (−0.085) | **0.614 (−0.017)** |
| diabetes | 0.581 | 0.528 (−0.053) | **0.549 (−0.032)** |
| adult | 0.705 | 0.788 (+0.083) | **0.821 (+0.116)** |

Preserving the within-cell ordering requires evidence that the ordering is real, and nothing
available to a post-processing tool provides that. The transferable lesson is not specific to this
tool. A check fitted on a system's own output measures self-consistency. It cannot detect that the
system is confidently wrong, and when confidence and wrongness correlate it will actively recommend
the wrong action.

## Licence

Apache-2.0.
