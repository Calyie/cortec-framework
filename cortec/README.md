# cortec

Differentially private synthetic tabular data from a **frozen** language model conditioned only on
DP statistics.

The privacy budget is spent once, on a statistics release. The generator reads only that release
and never a private record, so generation is *post-processing*: it adds no privacy cost, and you
can draw unlimited datasets from a single release.

```
private data ──► DP release (Stage A) ──►│──► frozen LLM (Stage B) ──► synthetic records
                 the budget is spent     │     reads only the release, ε = 0
                 here, once              │     repeatable for free
                                    nothing right of this line
                                    ever sees a private record
```

## Deploying it

[`docs/deployment.md`](docs/deployment.md) is the operational manual: the controls mapped to standards, every
operational failure mode with the guardrail that enforces it, the deployment checklist, the two
deployment patterns, and the cost model.

## Install

```bash
pip install cortec[anthropic]     # Claude — recommended on AWS Bedrock
pip install cortec[openai]        # GPT — recommended on Azure OpenAI
pip install cortec[gemini]        # Gemini — recommended on Google Vertex AI
pip install cortec[ollama]        # scientific controls only — NOT a deployment path; see below
```

**On where this runs.** The prompt contains only the DP release — noisy histograms, class balances
and a conditional table — and never a private record, so by post-processing immunity the trust
boundary is crossed *before* the model call. That is what makes it safe to point at a
tenant-isolated enterprise endpoint inside your own account or project. It is also why the choice
of model is a **quality** decision rather than a compliance one.

## Use

```python
from cortec import Schema, Band, release_statistics, Generator

schema = Schema(
    name="encounters",
    numerical={"age": (18, 95), "length_of_stay": (1, 30)},
    categorical={"admission_type": ["Emergency", "Elective", "Urgent"],
                 "a1c_result": ["None", "Norm", ">7", ">8"]},   # 'None' is a real category
    target="readmitted_30d", positive="YES", negative="NO",
    bins={"age": [18, 40, 55, 70, 95], "length_of_stay": [1, 3, 6, 10, 30]},
    stratify=[("length_of_stay", [Band("los1-2", 1, 3), Band("los3-5", 3, 6),
                                  Band("los6+", 6, 31)])],
    conditional=[("admission_type",), ("admission_type", "a1c_result")],
)

# Stage A — the only code that touches your private data.
release = release_statistics(schema, private_df, epsilon_total=2.0, n_min=150)
print(release.audit["epsilon_accounted"])      # 2.0
release.to_json("release.json")                # contains no private data

# Stage B — post-processing. Run it as often as you like, at no further privacy cost.
# Every batch is given the exact per-column counts it owes (on by default), and generate_selected
# draws a 3x pool and keeps the rows whose marginals match the release.
gen = Generator(schema, backend="anthropic", model="claude-fable-5", reasoning="on")
synthetic = gen.generate_selected(release, n_rows=5000, pool_factor=3)

# Verify reasoning actually fired. A reported count of ZERO and NO count at all are different
# states, and some transports return neither -- reading thinking_tokens alone once produced a
# published claim that a model's reasoning was off when it had in fact run on every call.
print(gen.stats.calls_with_reasoning_block, "of", gen.stats.calls, "calls reasoned")
print(gen.stats.calls_reasoning_unmeasured, "calls could not be verified either way")
for w in gen.stats.warnings:
    print("WARN:", w)
```

`generate()` is the same without the pool (one call's worth of rows per batch, still with exact
counts). `generate_by_cell` generates per released conditional cell instead of per cohort; it
refuses a release whose conditioning columns are not substantially covered by the released cells,
because that path emits rows only for released cells and silently drops the rest of the population.

## What this is for, and what it is not for

**Use it** when your private data plausibly departs from what a general-purpose model would assume
— institution-specific coding, local case mix, proprietary product structure — and when a
downstream model or a calibrated rate is the deliverable.

**Do not use it** for a published marginal table when a marginal synthesiser will do: with exact
counts and selection (below) the output reaches MST's 1-way error and beats both MST and AIM on
2-way and conditional error, but the marginal synthesisers pay once and sample freely, and AIM led
on its own 3-way workloads in our measurements of the earlier configuration.

**How close to real data does it actually get?** This is the number the method is usually adopted
for, so here is the boundary rather than the best case. On UCI Adult **at n = 300**, models trained
on CoRTeC's output are statistically indistinguishable from models trained on a real sample of the
same size — differences of +0.007, −0.001 and +0.011 AUC across three students, every p > 0.27.
**That result was measured on one dataset first, and we measured where it stopped.** On a finance
dataset the same comparison was a *significant* shortfall under the pooled release: TSTR-LR 0.652
against a real sample's 0.695, about 94% of real-sample utility (Welch p = 0.0034). The
class-conditional release takes the two tree students to the real-sample floor on that dataset, and
the configuration this package now ships by default (class-conditional release, exact-count batches,
selection from a threefold pool; see below) brings all three students within 0.011 AUC of the floor
on both datasets, with 1-way marginal error at the level of MST and below a real sample of the
same size (paper §7.12). That is a result at n = 300 under one generator family. On Adult at
n = 1,000 the point estimates favour the real sample (0.827 against 0.855), underpowered at two
draws but in the direction the mechanism predicts: a fixed release does not get richer as you ask
for more records. **Do not assume parity: measure it on your own data against a real sample of
matched size, and read the conditional measures beside the aggregate ones.**

**Do not use an unconditioned LLM** for any regulated quantity. Asked for synthetic hospital
records with no conditioning, a frontier model reported a 30-day readmission rate of 98.5% for a
group whose true rate was 21.4% — and reported that same 98.5% regardless of what the private data
said. No rank-based utility metric will reveal that, because the ordering is right and only the
magnitude is catastrophic.

## Where the trust boundary sits

**No private record is ever sent to the model.** The prompt carries only released statistics —
noised histograms, class balances and a conditional table. Generation is post-processing of a DP
release, so by post-processing immunity whoever runs the model learns nothing beyond what the
release already discloses, and the ε guarantee is indifferent to *where* that computation happens.

This matters for the reflexive objection to any LLM-based method — "you are sending data to a third
party" — which does not apply here, **not because the endpoint is trustworthy but because there is
no private data in the request.** The decoder's trust level is not load-bearing. It may be a
tenant-isolated enterprise endpoint (Bedrock in your own VPC, Azure OpenAI, Vertex AI) under a BAA,
or an air-gapped model on your own hardware with no network egress: the argument is identical. What
differs between those is *capability*, not trust — see the sweep-only tier below, which is why the
deployment recommendation names enterprise platforms even though the privacy argument permits local
weights.

**What this does not cover.** The guarantee says nothing about the model's pretraining corpus. If a
private record was in it, that happened before this tool ran. The sharper version of the concern is
that conditioning on true marginals for a narrow stratum resembles a prompt-based extraction attack
— it tells the model which region to sample from, and a memorised record there becomes more
reachable. DP does not exclude this, because such a record is not a function of your release. In
the research behind this tool, four membership-inference attacks (nearest-neighbour, exact-match,
shadow-model, and a per-record likelihood-ratio test, each validated on a positive control) found
**no advantage above chance and zero exact matches** on any dataset. That is evidence about the
published output, not a proof about the corpus, and it is the strongest statement available.

## The guarantees, and how to audit them

Record-level ε-differential privacy under add/remove-one adjacency, in the central model.

**ε protects one ROW, and on some data that is not one person.** Under group privacy a person
contributing `k` rows receives `k · ε`. On encounter-level hospital data — one admission per row,
several per patient — this is not a technicality: in a real dataset of 101,766 encounters from
71,518 patients the heaviest patient contributes **40 encounters**, so a declared ε of 2.0 is an
**ε of 80** for that patient, which is outside any range normally considered meaningful.

**So this package is for one-row-per-person data.** On unaggregated longitudinal records the
guarantee degrades to a number that carries no meaning, and that is a limit on what the mechanism is
for rather than a box to tick. The *magnitude* is a property of row-level DP rather than of CoRTeC —
any row-level synthesiser on the same data inherits the same factor — but that is a reason not to
prefer a competitor, not a reason to ship the claim.

`k` cannot be computed for you without spending budget on it, so you declare it — and where you
declare one whose ε per person is vacuous, **the release is refused rather than annotated**:

```python
# refused: eps_person = 80
cortec.release_statistics(schema, df, epsilon_total=2.0, max_rows_per_person=40)
# ReleaseError: REFUSED: max_rows_per_person=40 at epsilon_total=2.0 gives
#   epsilon_per_person=80.0, past the 10 beyond which a per-person guarantee carries no
#   meaning ... Either aggregate to one row per person before Stage A ...

# proceed anyway, with the acknowledgement recorded in the audit trail that ships with the release
release = cortec.release_statistics(schema, df, epsilon_total=2.0, max_rows_per_person=40,
                                    acknowledge_vacuous_privacy_unit=True)
release.audit["epsilon_per_person"]           # -> 80.0
release.audit["epsilon_per_person_vacuous"]   # -> True
release.audit["warnings"]                     # -> ["VACUOUS PER-PERSON GUARANTEE ACKNOWLEDGED: ..."]

# the ordinary case
release = cortec.release_statistics(schema, df, epsilon_total=2.0, max_rows_per_person=1)
release.audit["epsilon_per_person"]           # -> 2.0
```

Leave it out and the audit records `"UNDECLARED"` with `privacy_unit_declared: false` and a vacuity
verdict of `"UNKNOWN"` — never `false`, and never a quiet assumption of 1, which is how an
encounter-level release comes to carry a per-person claim it has not earned. **If your data has more
than one row per person, either aggregate to one row per person before Stage A, or report
`ε_person` rather than the per-row number.**

One thing the tool cannot do for you: auto-configuration ranks conditioning columns by mutual
information and **has no concept of a person**, so it can select a column that is itself a proxy for
how many rows someone contributes (on the hospital dataset above it chose `number_inpatient`,
correlation 0.733 with encounter count). That does not change ε, but it means the release is
stratified by contribution count. Declare your conditioning columns yourself if that matters to you.

Every privatised query is recorded in a ledger, and `PrivacyLedger.spend()` is the only function
in the package that returns a noise scale — so nothing can be released without appearing in the
audit trail. `release.audit` is a complete, machine-readable record designed so a privacy engineer
can recompute the total by hand:

```python
import json; print(json.dumps(release.audit, indent=2))
```

It records every query with its epsilon, sensitivity, composition rule and partition key; the
per-group totals; and the composition rules applied. `cortec/accounting.py` is written to be read
end to end by someone auditing it, independently of the rest of the package.

One deliberate conservatism: suppressing cohorts below `n_min` is a decision made by looking at
private counts, so by default we **charge** for it rather than treating cohort sizes as public.
Pass `charge_suppression=False` for the literature-standard treatment; the choice is recorded.

## Class-conditional histograms: what the release carries, and why

The release used to carry one *pooled* feature histogram per cohort plus the target rate inside
the conditional hierarchy's cells — and nothing about how any other feature relates to the target.
The generator filled that in from its prior, and on the finance benchmark the filled-in columns
measurably *degraded* the downstream model (paper §F.3.1). Since 2026-09-18 the release carries
one histogram block per **(cohort, outcome)** wherever both outcomes clear `n_min`:

- **Same ε.** The two outcome blocks of a cohort are disjoint, so they compose in parallel at the
  same ε per query the pooled block spent; the pooled histogram is their mixture under the released
  class balance — post-processing, no query. The ledger charges each record the class balance plus
  *one* block, through nested partition keys (`cohort/outcome`), and `epsilon_accounted` is
  unchanged. Where either outcome falls short of `n_min` the cohort keeps a pooled block.
- **What the model sees.** "Rows with `readmitted_30d = YES` look like this; rows with `NO` look like
  that" for every column, with the instruction to decide each row's outcome first and draw its
  other columns from that outcome's distributions. Cell-wise prompts show the same per-outcome
  blocks for the *k* positives and *n − k* negatives they ask for.
- **Measured.** Decoded by a naive independent sampler with no model at all, the class-conditional
  release reaches the real-sample floor on two of three downstream students on both finance and
  Adult, where the pooled release trails it by 0.03–0.06 AUC; held-out conditional error improves
  by 31–39% (`cortec/paper/audit/release_sufficiency.py`). Through the generator, on the
  finance benchmark, three draws per release: TSTR 0.680 / 0.712 / 0.704 (LR / RF / GBM) from the
  class-conditional release against 0.651 / 0.662 / 0.662 from the pooled one with Gemini 3.5 Flash
  (Welch p = 0.007 / 0.017 / 0.029), and 0.670 / 0.723 / 0.716 against 0.652 / 0.673 / 0.664 with
  Claude Fable 5, where the tree students reach the real-sample floor of 0.695 / 0.727 / 0.717 and
  held-out conditional error rises 0.008; on Adult two Fable 5 draws score 0.846 / 0.885 / 0.873
  against a real sample's 0.830 / 0.870 / 0.840 (paper §7.11).
- `release_statistics(..., class_conditional=False)` restores the pooled release.

## Exact counts and selection: how the output reaches the release's own fidelity

Two steps, both post-processing of the release, both on by default.

- **Exact-count batches.** Every batch is told the number of rows it owes per bin of each numeric
  column, per category of each categorical column, and per outcome, apportioned from the released
  histograms for the cohort as a whole and updated after each accepted batch so that a batch
  returning more or fewer valid rows than asked cannot leave the cohort short. A frontier model
  given these counts reproduces them exactly; given shares, it matches about two thirds of them.
  `Generator(quota=False)` turns this off.
- **Selection from a pool.** `generate_selected(release, n_rows, pool_factor=3)` asks for three
  times the rows and keeps the `n_rows` whose cell counts match the release: inclusion weights by
  iterative proportional fitting over every released cell, systematic sampling, then a greedy
  exchange of rows that lowers the weighted distance to the released counts, with the pooled
  marginals weighted above the per-cell counts and any suppressed category given zero mass. A
  cohort whose release carries only a pooled block is constrained at cohort level, not per class
  (imposing a pooled histogram on each class erases the feature-to-target structure; that is a
  regression test). The release carries the cohort rule it was built under, so selection works on
  auto-configured releases whose cohorts the caller's schema does not name.
- **A pool that stopped early is refused.** Selection can only choose among rows that exist, so a
  generation run that hit its spend cap or lost its connection leaves its later cohorts unfilled,
  and no reweighting can repair a cohort share from rows that are not there. `select_to_release`
  checks every released cohort's rows against the rows it owes (`pool_coverage` reports them) and
  raises, naming the short cohorts, unless `allow_short_pool=True`. Generate the full pool.
- **Categorical cells are checked against the declared set.** A model that writes an integer-coded
  category with a decimal point (`2.0` for a column declared as `1`/`2`) would otherwise re-type
  the whole pool as floats on concatenation, and the selection step and every consumer would then
  see a category the data does not have. The parser normalises integer-valued floats to the
  declared string and drops rows whose value is not declared, counted in
  `stats.rows_undeclared_category`, exactly as an out-of-domain number is dropped.

The `mock` backend honours the exact-count block, so the whole generate-and-select loop runs end
to end with no API and no cost; the offline test in `tests/test_selection.py` does exactly that.

One more step is part of the default and follows from the release. A released histogram fixes
bin counts and nothing finer, so where a value sits inside its bin is never released information.
In cohorts whose release carries class-conditional blocks, `generate_selected` redraws every
numeric value uniformly inside its released bin (never crossing the row's stratification band),
so the output's structure below bin resolution is the release's rather than the generator's; in
cohorts with only a pooled block the generator's values are kept, because there they are the only
carrier of the class signal. Measured on every stored arm (paper, Appendix H.17): it lifted NHANES's
logistic-regression student from 0.729 to 0.752 against a real sample's 0.773 and changed no
binned measure anywhere; `sub_bin="generator"` disables it.

Measured in the paper (§7.12) on Adult and finance with Gemini 3.5 Flash: 1-way marginal error within
0.005 of MST's and below a real sample of the same size, 2-way error below AIM and MST, conditional
error a third of a real sample's, and downstream utility at the real-sample level. The ceiling is
the release's own distance from the truth, which is why the conditional table's share of the
budget defaults to a fifth (`conditional_fraction=0.2`): the histograms take the rest, and the
release's marginal error halves for no measurable loss elsewhere. Cost: `pool_factor` times the
generation spend, and roughly twice the thinking per call on models that reason over the counts.

## Guardrails you cannot turn off by accident

These exist because each one corresponds to a defect that produced a plausible but wrong
conclusion during the research behind this tool.

| Guardrail | The wrong conclusion it prevents |
|---|---|
| Model capability gating | A 7B model scored a transmission slope of 0.21 while emitting perfectly valid CSV. It fails *silently*, so it is refused rather than warned about. |
| Hash-locked prompts | Removing the conditional table or the "even if counterintuitive" instruction collapses the mechanism to an unconditioned model, while the pipeline keeps producing plausible rows. |
| Context-fit check | Ollama's 4096-token default silently truncated the prompt, cutting off the conditional table — it looked like "this model family ignores the statistics". |
| Empty-content detection | A reasoning model spent its whole output budget on hidden chain-of-thought and returned `content=""` — it looked like "this model cannot follow the schema". |
| `keep_default_na=False` | pandas reads the string `'None'` as missing. `A1Cresult='None'` means the test was not ordered; the default deleted 83% of rows at a 100% call-success rate. |
| Domain-bounds rejection | A degenerate response produced a credit limit of 34 against a declared floor of 10,000; without the check it entered the dataset silently. |
| Declared-category check | One batch wrote an integer-coded category as `2.0`; concatenation re-typed the whole pool, and the selection step and the evaluator then saw a category the data does not have (conditional error 0.22 against 0.01 from a pool whose generation was fine). |
| Pool-coverage guard | A run that stopped at its spend cap left two cohorts with 7 rows each; selection cannot fill a cohort from rows that do not exist, and the output's cohort shares were wrong while every per-row check passed. |
| Bands tile the domain | Half-open autoconfig bands left the records at the domain maximum in an `oob` cohort that was released like any other and that no generated row could ever join. |
| Published cohort size floored at `n_min` | At ε = 0.3 a 469-record cohort's noised size clipped to zero and the generator gave it 1 of 600 rows: a whole age band missing from the output while every per-row check passed. A cohort is released only because it holds `n_min` records, so the published size is clamped there (post-processing). |
| Proportional row allocation | Uniform allocation over-represented a 2%-of-population cohort by 12×, so every marginal measured afterwards described a deliberately wrong mixture. |
| Validation before spending | Privacy budget cannot be refunded, so schema conflicts are raised before the first query. |
| `n_min` floor | A conditional rate over too few records is dominated by its own Laplace noise. |

Run them: `pytest tests/` — each test is named after the defect it prevents.

## Getting accurate absolute rates: use cell-wise generation

`Generator.generate()` asks the model for a mixed batch of records and lets it allocate them
across the released conditional cells. That caps how finely any one cell's rate can be expressed.
On UCI Adult with 12 rows per call, `education = Masters` receives **0.65 rows per call** — it can
only emit 0% or 100%, and since its released rate of 0.554 exceeds a half, it rounds to 100% every
time. Measured end to end this gave **MAE 0.253, slope 1.64**: rates below ~0.19 accurate, rates
above ~0.24 saturating toward 1.0. The apparent "rate threshold" was an artifact — on this dataset
the *rare* education values happen to be the *high-income* ones.

`Generator.generate_by_cell()` fixes it. Each call covers exactly one released cell, so the row
count is known, and the prompt states the outcome as an **integer count** ("exactly 7 of these 12
records must have income = '>50K'") rather than a probability. The model has no discretion over the
target column, and the residual error is bounded by the stochastic rounding of a single row.

| | released | `generate()` | `generate_by_cell()` |
|---|---|---|---|
| Masters | 0.554 | 1.000 | **0.625** |
| Bachelors | 0.419 | 0.968 | **0.408** |
| Assoc-acdm | 0.238 | 0.818 | **0.300** |
| Some-college | 0.190 | 0.189 | 0.206 |
| HS-grad | 0.159 | 0.148 | 0.155 |
| | | MAE **0.253**, slope 1.64 | MAE **0.037**, slope 1.19, r 0.99 |

```python
synthetic = gen.generate_by_cell(release, n_rows=5000)   # accurate absolute rates
```

**Use `generate_by_cell()` whenever absolute conditional rates matter** — a readmission
probability, a default rate, anything a regulated decision consumes. It costs more calls for the
same output size (34 calls for 300 rows here, against 25), which is the trade. `generate()`
remains available and is adequate when only relative ordering is needed.

Two caveats we have not resolved. Rounding is stochastic, so a cell allocated very few rows still
carries up to half a row of error in a single draw — draw more records, which is free. And this was
measured on one dataset with one model; run `tools/acceptance_test.py` against your own schema
before relying on the numbers, which is what it is for.

### The coverage guard, and what it does not cover

Cell-wise generation emits rows **only for released cells**, so any band of a conditioning column
that no released cell names is produced at essentially rate zero. On one health-survey release that
erased four of six racial groups to exactly 0.000 while aggregate fidelity and downstream utility
both stayed healthy. So `generate_by_cell()` refuses when any conditioning column has less than
**90%** of its released marginal mass inside released bands, names the offending column, and points
at the knobs that change it (`n_min`, conditional depth). The check reads the release only, so it
costs no budget. Override with `allow_low_coverage=True` — for measurement, not for use.

**Treat this as a heuristic mitigation, not a solved problem.** The guard answers one question —
*which bands of a column did the released cells span?* — soundly and for free. It does not answer
the one next door: *what is inside a band they did span?* On a coarsened high-cardinality
categorical those differ, and a column scoring **100% coverage** was still destroyed: its cells were
keyed on an opaque group label, the generator was told which values were legal and not how often
each occurs, and it spread them near-uniformly — a discharge code at 59.2% of real records came out
at 14.3%, which is 1/7 on a seven-member group. This tool now expands every coarsened group into its
members **with their within-group shares** (a renormalisation of proportions already in the release,
so it costs no budget), which repairs it on both vendors' models we tested. But a fix for one named
failure mode is not a guarantee about the mapping in general: **establishing when a DP conditional
table can be conveyed to a language model without silent loss is an open problem.** A passing guard
is evidence that one failure mode is absent. Compare per-column output against the release anyway.

## Model support

Two things decide output quality here, and only one of them is the model.

**Enable reasoning.** This is the single largest effect we measured, and it is free to get right.
On one model tested against itself — same release, same prompts, one flag changed, **three draws
per arm** — conditional error moved **0.173 ± 0.032 → 0.045 ± 0.007, a 3.8× improvement** (Welch
p = 0.005). That takes it from *worse than a no-information control* (0.152) to among the best
generators measured. `Generator(..., reasoning="on")` is the default.

The trap is the economics, and it is sharper than it first looks. The suppressed setting is
**14.8× cheaper**, and across the same three draws per arm **downstream utility does not separate
at all**: TSTR-LR reads 0.786 ± 0.050 suppressed against 0.802 ± 0.025 enabled, well inside one
standard deviation. So a team optimising cost against a standard TSTR check does not see a small
difference and accept it — it sees **no measurable difference**, concludes correctly on that
evidence, and ships a pipeline carrying none of their data's conditional structure. Only a
conditional measure reveals it. (We previously reported reasoning as improving downstream utility
too; at one draw it appeared to. It does not survive replication, and we withdrew it.)

**And verify that reasoning actually fired.** Setting the flag is not evidence. We once read a
model's reasoning as *intermittent* — a large count in one batch, none in another — and it was our
instrument: the two batches ran on different transports, and one of them reports no count at all.
A count that was never reported is not a count of zero. The tool keeps the two states apart and
warns you:

```python
gen.stats.thinking_tokens            # total reported reasoning tokens
gen.stats.calls_without_reasoning    # calls that REPORTED zero while reasoning="on"
gen.stats.calls_reasoning_unmeasured # calls on which the vendor reported no count at all
gen.stats.calls_with_reasoning_block # calls where a reasoning block was present (Anthropic)
gen.stats.positives_off_by_more_than_one  # cell-wise calls whose emitted positive count missed the
                                          # "exactly k" it asked for by more than one row
gen.stats.warnings                   # explicit warnings for each of the above
```

Both generation methods append one provenance column to their output — `_cohort` for
`generate()`, `_cell` for `generate_by_cell()` — naming the released unit each row was generated
for. Drop it before use if your consumer expects the schema's columns only.

### Backends

| backend | enterprise surface | reasoning control |
|---|---|---|
| `anthropic` | **Claude on AWS Bedrock** (recommended); the public `api.anthropic.com` also works but is not the recommended deployment surface | on by default |
| `openai` | **GPT on Azure OpenAI** (recommended); the public `api.openai.com` also works but is not the recommended deployment surface | on by default; `reasoning="suppressed"` sends `reasoning_effort="minimal"` — note this vendor's `"low"` is *not* low, only `"minimal"` reaches zero |
| `gemini` | **Gemini on Google Vertex AI** (recommended); the public `generativelanguage.googleapis.com` also works but is not the recommended deployment surface | on by default; suppression is best-effort, since this vendor rejects a zero thinking budget outright |
| `ollama` | **not an enterprise surface — out of deployment scope** | no reasoning mode on the models we measured; provided for reproducing the paper's scientific controls, and the capability gate refuses these models by default |

**Which endpoint to point this at.** For regulated data, use the **enterprise-hosted, tenant-isolated**
surface in your own cloud account — Bedrock, Azure OpenAI, or Vertex AI — with private networking, data
residency, contractual exclusion of training on inputs, and a BAA where HIPAA applies. **The vendors'
public developer APIs are supported by these backends but are not recommended for regulated
deployment.** The weights are the same; the contract and the network path are not, and those are what a
compliance review assesses. CoRTeC's *privacy* argument does not depend on the endpoint — the request
carries only the DP release — but its *compliance* argument does. (The paper's own experiments ran
against the public APIs; §6.2 of the paper discloses this and says which results that affects.)

**Reaching the enterprise surface.** Pass `surface=` and, where the surface names models differently,
`surface_model=`; `model` stays the validated profile name that the capability gate and the price
table key on. The request, the reasoning control and the response handling are identical to the
public path — only the client class, its credential flow and the model identifier change:

```python
# Claude on AWS Bedrock: AWS credentials from the standard chain (env, profile, instance role)
#   pip install 'cortec[bedrock]'   AWS_REGION=eu-central-1
gen = Generator(schema, backend="anthropic", model="claude-fable-5", surface="bedrock",
                surface_model="anthropic.claude-fable-5-v1:0")      # model id or inference-profile ARN

# GPT on Azure OpenAI: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY (AZURE_OPENAI_API_VERSION optional)
gen = Generator(schema, backend="openai", model="gpt-5", surface="azure",
                surface_model="gpt5-prod")                          # your DEPLOYMENT name

# Gemini on Vertex AI: Application Default Credentials; GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
gen = Generator(schema, backend="gemini", model="gemini-3.5-flash", surface="vertex")
```

A missing setting or a surface/backend mismatch fails at construction, before any client exists;
`gen.describe_surface()` is one line for your audit trail; and a run on the public surface says so in
`gen.stats.warnings`. **What is verified, exactly:** the tests construct the SDKs' real client classes
offline and replace only the transport, so an SDK constructor drift fails in CI rather than at your
first call; and **Vertex AI is verified live** — Gemini 3.5 Flash through a service account holding
only `roles/aiplatform.user`, both generation paths, reasoning reported on every call. Two things
that run taught us: the newest Gemini models are served from the **`global`** location and return
404 in every regional one (set `GOOGLE_CLOUD_LOCATION=global`), and a model 404 is now a fatal,
non-retried error. **What is not verified:** a live round trip on Bedrock or Azure OpenAI — no
account was available to us. Treat your first run there as that test: generate a small batch, check
`gen.stats.calls_with_reasoning_block` / `thinking_tokens`, and compare the output to the release
before generating at scale.

### Capability gating

| Model | Tier | Magnitude error (transmission sweep) |
|---|---|---|
| Claude Fable 5 / Opus / Sonnet | validated | 0.009 |
| Gemini 3.x Flash | validated | 0.008 |
| llama3.3:70b-instruct | **sweep-only — refused** | 0.020 |
| qwen2.5:72b-instruct | **sweep-only — refused** | 0.030 |
| mistral-small:24b, gemma2:27b, gpt-oss:20b | adequate | 0.049–0.078 |
| qwen2.5:32b, qwen2.5:14b | adequate | 0.098–0.105 |
| qwen2.5:7b | **insufficient — refused** | 0.406 |

### Which model to use

We recommend the models we measured on full-dataset generation:

```python
from cortec.models import recommended_table
print(recommended_table())
```

An unlisted model is refused unless you pass `allow_unvalidated=True`, and if you do, verify
transmission on your own data before trusting any output — the sweep-only tier below is why.

**Read the sweep-only tier carefully, because it is the trap this table exists to prevent.** The
transmission sweep asks whether a model reproduces a released rate — one large, salient signal in a
short prompt. Two self-hosted 70B models pass it *inside the frontier band*, and on that basis this
table used to call them validated and recommend them. Re-measured on full-dataset generation at
three draws, `llama3.3:70b-instruct` closes **12%** of the conditional-error gap to an
*unconditioned* prompt, closes **nothing at all** on 2-way total variation (0.238 against the
control's identical 0.238), and is **worse than that control** on 1-way total variation. On the
metrics this method exists to improve, its output is close to indistinguishable from asking a model
for plausible records with no conditioning — and no output check reveals it. Passing the sweep is
necessary, not sufficient. These models are now refused by default and run only under
`acknowledge_insufficient=True`.

Capability is family-dependent, not just size-dependent: at comparable scale a 24B model recorded
lower error than a 32B one, so a size threshold alone would be wrong. An unlisted model is refused
unless you pass `allow_unvalidated=True`.

**Some models are measured but not listed, and the gate tells you so.** GPT-5 and Gemini 3.1 Pro
were measured on *full-dataset generation* (conditional error 0.045 and 0.041, both inside the band
of the validated frontier models and below what a real 300-record sample achieves) but not on the
transmission sweep every tier above is scored against. **Those are different experiments and their
numbers are not interchangeable** — a model can track a forced rate in three cohorts while
attending weakly to a full multi-level table, and nothing in the output says which is happening.
So they are still gated, and the refusal message reports exactly what we do and do not know.

### A withdrawn claim

An earlier version of this README stated that 70B-class models are "validated *and* self-hostable,
so a regulated deployment needs no external API and no data leaves the institution." **We withdraw
the first half.** That validation was measured on the *transmission* sweep. On full-dataset
generation the same class of model closes only **12%** of the gap between an unconditioned prompt
and a frontier model on conditional-seen error, **20%** on held-out, **nothing at all** on 2-way
total variation, and is *worse* than an unconditioned prompt on 1-way. The failure is silent and
the marginal metrics do not reveal it.

The second half was never the load-bearing part: because the prompt carries a DP release and never
a record, an enterprise endpoint under your own tenancy is available to you, and the generator
choice is a quality decision. If you do run local weights, verify conditional fidelity on your own
data first.

## Licence

Apache-2.0.
