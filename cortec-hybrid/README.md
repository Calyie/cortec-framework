# cortec-hybrid

Conditional-signal correction for marginal DP synthesisers (AIM, MST, PrivBayes). No LLM is
required; the heaviest dependency is scikit-learn.

## It does not generate synthetic data

The name is misleading, and this is the clarification that matters most. `cortec-hybrid` contains
no synthesiser. It never calls AIM, MST or anything else, it will not produce a synthetic dataset
for you, and it has no dependency on smartnoise-synth.

It is a post-processor. You bring:

1. your private data, and
2. synthetic data you already generated with whatever DP synthesiser you use: AIM, MST, PrivBayes,
   a vendor product, anything.

It releases a DP conditional target table from (1), and rewrites only the target column of (2) to
match it. Every feature column comes out byte-identical to what you put in; the tool verifies this
and the test suite asserts it. If your synthesiser is already well calibrated on conditional
structure, the tool will tell you so, and you should keep your budget.

```
your private data ──► DP conditional table ──┐
                                             ├──► relabel target column only ──► corrected output
your AIM/MST output ─────────────────────────┘
```

## The narrow claim

Marginal-based DP synthesisers are excellent at the statistic they optimise, and they are evaluated
on it in their own papers. Neither AIM's nor MST's paper reports a downstream predictive-utility
experiment, and in our measurements that omission is consequential. At matched budget and matched
sample size on UCI Adult, models trained on their output reached 0.675 to 0.728 AUC where a real
sample of the same size reached 0.834 to 0.872 (technical report §7.2).

This tool spends a small, separately accounted slice of privacy budget on the one thing those
mechanisms do not target, a conditional target table `P(y | cell)` over a disjoint partition, and
relabels the synthesiser's output to match it. Because the cells partition the data, the whole table
costs one query's worth of ε however many cells it holds: accuracy is set by cell support, not cell
count. That is what makes a rich table affordable. Relabelling is pure post-processing of two
already-DP artefacts, so it costs nothing beyond the table itself.

## What this is not

It is not "CoRTeC without the LLM", and it is not a general improvement over its inputs. A coarse
12-cell table, the obvious implementation, fixes calibration and adds little utility in our runs
(+0.025 AUC on AIM and +0.037 on MST, on Adult); the large gain arrives only with a richer table
(technical report §8.2). Use it in one specific situation: your marginal synthesiser's output has
weak conditional target structure and you need a calibrated rate. Then measure, and keep your budget
if the gain is small. `estimate_gain()` will tell you so in as many words.

**Read `worth_it` for what it is.** It is a calibration criterion, meaning that conditional error
against the released table fell by more than 0.02, and it does not predict downstream utility. We
measured the two coming apart. Across 11 datasets the correction improved calibration in essentially
every case while downstream AUC moved in both directions, and of 9 runs flagged `worth_it` the
downstream AUC fell on 6 in one run and on 3 of 10 in a repeat. Three datasets, a renal registry,
hospital readmission and cervical cancer, were reliably negative: calibration reliably improved and
downstream AUC reliably dropped. `estimate_gain()` returns `worth_it_does_not_predict` saying
exactly this. Validate on your own held-out data before adopting the correction.

## Use

```python
from cortec import Schema
from cortec_hybrid import check_feasible, correct

check_feasible(schema)          # refuses BEFORE you burn hours on a schema AIM cannot fit

# `synthetic_df` is YOUR synthesiser's output; this tool does not produce it.
corrected, table, gain = correct(
    schema, private_df, synthetic_df,
    columns=("admission_type", "a1c_result"),   # richer is better and costs no extra ε
    epsilon=0.5,
)
print(gain["verdict"])
# -> "the correction materially improved conditional calibration"
#    or "the correction changed little; ... consider keeping the budget instead"
```

## The up-front feasibility check

`check_feasible()` refuses a schema AIM is unlikely to converge on, so you learn in the first second
rather than the third hour.

It used to gate on domain size, and our own ablation refuted that. UCI Adult has the largest domain
of every schema we measured, 4.4 × 10¹³, six million times larger than a healthcare prefix that
times out, and Adult is the one that fits. Within a single dataset, the arm with the larger domain
fit 425 seconds faster than the arm with the smaller one. Domain volume does not predict
convergence, and gating on it refuses the wrong schemas.

Three factors did predict it, each demonstrated by a controlled flip:

- **attribute count**: a 19-attribute schema times out at a row count where a 15-attribute one fits;
- **row count**: the same schema at the same domain moves from a 26-minute fit to a timeout purely
  by going from 26,048 to 81,410 rows;
- **high-cardinality correlated numeric columns**: one dataset has Adult's width, fewer rows and a
  30× smaller domain, and it still fails. Dropping any three columns does not rescue it; dropping
  three amount columns does. A width-matched control (dropping three demographics instead) still
  times out, which is the only way to separate those two causes.

```python
check_feasible(schema, n_rows=len(private_df))   # pass n_rows: it is one of the three factors
```

The report includes the full observation table it is derived from, and the estimated domain size,
reported for reference and explicitly not used to decide. This is a heuristic from ten fits under
one AIM implementation, not a theory; pass `raise_on_fail=False` to proceed and measure it
yourself.

## A correction to our own earlier belief

We previously held that rank-preserving assignment was necessary for this correction to pay for
itself. Testing this package disproved that in one regime, and the code changed as a result.

Rank preservation only helps when the input synthetic data already carries within-cell structure.
When its target is close to noise, precisely the case this tool targets, a ranking model fitted on
that target learns the noise, and ordering by it injects spurious structure that a downstream model
mistakes for signal. In our fixture that cost 0.09 AUC against plain i.i.d. assignment.

The guard we first wrote for this did not work, and why is worth knowing. `_rank_scores` gated on
whether the synthetic target is predictable from the synthetic features (cross-validated AUC against
`RANK_INFORMATIVENESS_FLOOR = 0.55`). But a synthesiser generates its target as a function of those
features, so that score is near-perfect almost always: 0.981 on one dataset whose real downstream
AUC is 0.631. It measures self-consistency, not validity, and it is highest exactly when the
synthesiser has confidently learned a wrong relationship. It therefore selected for the failure mode
it was built to prevent.

Replicated over three datasets at three synthesiser seeds each, i.i.d. assignment scored higher than
rank preservation in 9 of 9 runs:

| dataset | base synthesiser | + rank-preserving | + i.i.d. within cell |
|---|---|---|---|
| renal registry | 0.631 | 0.546 (−0.085) | **0.614 (−0.017)** |
| diabetes | 0.581 | 0.528 (−0.053) | **0.549 (−0.032)** |
| adult | 0.705 | 0.788 (+0.083) | **0.821 (+0.116)** |

So `preserve_ranking=False` is the default, and rate matching alone is the mechanism. Preserving the
within-cell ordering requires evidence that the ordering is real, and nothing available to a
post-processing tool provides that.

The transferable lesson is not specific to this tool. A check fitted on a system's own output
measures self-consistency. It cannot detect that the system is confidently wrong, and when
confidence and wrongness correlate it will actively recommend the wrong action.

## Privacy

The tool reuses `cortec`'s audited `PrivacyLedger`, so the composition rules are implemented once
and audited once. `table.audit` is the same machine-readable trail. Validation runs before any
budget is spent, and `n_min` has a hard floor of 50 because a rate over fewer records is dominated by
its own Laplace noise.

**ε protects one ROW, and this tool is the one most likely to be pointed at data where that is not
one person.** It corrects a synthesiser you already run, so it gets aimed at the table your
institution already has, including encounter-level clinical data, where one patient contributes many
rows. Under group privacy a person contributing `k` rows receives `k · ε`: on a real hospital dataset
whose heaviest patient has 40 encounters, a declared ε of 2.0 is an ε of 80 for that patient, which
carries no meaningful guarantee.

Declare the privacy unit. Where you declare one whose ε per person is vacuous, the release is
refused, not annotated:

```python
# refused: eps_person = 80
release_conditional_table(schema, df, ("grp",), epsilon=2.0, max_rows_per_person=40)

# proceed with a ROW-level guarantee, acknowledgement recorded in the audit trail
tbl = release_conditional_table(schema, df, ("grp",), epsilon=2.0, max_rows_per_person=40,
                                acknowledge_vacuous_privacy_unit=True)
tbl.audit["epsilon_per_person"]          # -> 80.0
tbl.audit["epsilon_per_person_vacuous"]  # -> True
```

Leave it out and the audit records `"UNDECLARED"` rather than assuming 1. If your data has more than
one row per person, aggregate before releasing, or report `ε_person` rather than the per-row number.

## Licence

Apache-2.0.
