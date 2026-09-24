# cortec

Differentially private synthetic tabular data from a frozen language model that is conditioned only
on DP statistics.

The privacy budget is spent once, on a statistics release (Stage A). A frozen model then generates
records from that release and never sees a private record (Stage B), so generation is
post-processing: it adds no privacy cost, and unlimited datasets can be drawn from one release. An
optional Stage C attaches a budgeted utility bound to the output.

```
private data ──► DP release (Stage A) ──►│──► frozen LLM (Stage B) ──► synthetic records
                 the budget is spent     │     reads only the release, ε = 0
                 here, once              │     repeatable for free
                                    nothing right of this line
                                    ever sees a private record
```

This README is the setup guide. Read it in order the first time. Each step names the call it uses,
the settings that matter, and what to check before moving on.

| Step | What you do |
|---|---|
| 1. Install and run | Install the package, then run the whole pipeline on your table with one command |
| 2. Choose where the model runs | Pick the surface and set its credentials |
| 3. Prepare the data | One table, one row per person, string categoricals |
| 4. Declare the schema | Public facts only: columns, bounds, bins, target |
| 5. Stage A | Spend the budget once, check the audit, store the release |
| 6. Stage B | Generate from the release and check the run |
| 7. Stage C | Bound how far the output is from the private data |
| 8. Evaluate | Compare against a real sample and a permuted floor |
| 9. Trust boundary | What the model sees, and what the guarantee does not cover |
| 10. Troubleshooting | What each refusal means and what to change |
| 11. Output and exports | One printed layout and one exportable record for every stage |

Two further documents: [`docs/deployment.md`](docs/deployment.md) is the operational manual for a
regulated deployment (the architecture, the checklist, the controls mapped to standards, the
failure modes, the two patterns and the cost model), and
[`docs/design-notes.md`](docs/design-notes.md) records the measurements behind the defaults.

## 1. Install and run

Python 3.10 or later. The package depends on numpy and pandas; each vendor SDK is an optional
extra. The package is not on PyPI, so install from a clone of this repository:

```bash
git clone https://github.com/Calyie/cortec-framework
cd cortec-framework
pip install './cortec[anthropic]'     # Claude
pip install './cortec[openai]'        # GPT
pip install './cortec[gemini]'        # Gemini
pip install './cortec[ollama]'        # local models: scientific controls only; NOT a deployment path (step 2)
```

The enterprise surfaces have their own extras: `bedrock` (adds request signing for AWS), `azure`
and `vertex`. Install `'./cortec[dev]'` as well to run the tests:

```bash
python3 -m pytest cortec/tests -q     # 151 tests, offline, each named after the defect it prevents
```

**Run it.** `run_cortec.py`, in this directory, runs the whole pipeline on your own table: Stage A
(the release), Stage B (generation), Stage C (the bound) and the evaluation, printing each in the
standard layout and writing every file under one folder that it names at the end. It needs two
inputs: the private table as a CSV with one row per person (step 3), and a Python file that
declares the schema from public facts only (step 4), for example:

```python
# my_schema.py
from cortec import Schema

SCHEMA = Schema(
    name="encounters",
    numerical={"age": (18, 95), "length_of_stay": (1, 30)},
    categorical={"admission_type": ["Emergency", "Elective", "Urgent"],
                 "a1c_result": ["None", "Norm", ">7", ">8"]},
    target="readmitted_30d", positive="YES", negative="NO",
    bins={"age": [18, 40, 55, 70, 95], "length_of_stay": [1, 3, 6, 10, 30]},
)
```

```bash
# offline first: no key, no spend, the whole pipeline on the mock backend
python3 run_cortec.py --data private.csv --schema my_schema.py --backend mock

# then a real model, with a spend cap; the run keeps the rows completed within it
export ANTHROPIC_API_KEY=...
python3 run_cortec.py --data private.csv --schema my_schema.py \
    --backend anthropic --model claude-fable-5 --n-rows 300 --budget-usd 5

# every setting, with its default
python3 run_cortec.py --help
```

Without `--holdout`, one fifth of the data is set aside before Stage A for the Stage C ceiling
and the evaluation. The output folder (default `results/<schema name>_<backend>`) holds
`synthetic.csv`, `release.json` (reused by later runs into the same folder, because a release is
made once), `bound.json`, and each stage's record as JSON, Markdown, CSV and a figure. The
sections below explain each stage and every setting; the same calls are available from Python
for your own scripts, and `cortec-hybrid` ships `run_cortec_hybrid.py` for the correction.

For a full tour of the three stages in the standard output, with no API key and no data, run
the bundled example (it builds a small synthetic table in code and uses the offline `mock`
backend, so the release and the bound are real and only the generated numbers are placeholders):

```bash
python3 example.py
```

For the list of everything the package can run, in order, with each function's arguments and
defaults read from the code, print the help page; give it a name for one function's full
documentation. The `cortec` command does nothing else: the package is a library, and the runs
are Python calls.

```bash
cortec --help                        # run order, functions, classes, refusals
cortec bound_with_controls           # one name in full
cortec-hybrid --help                 # the same page for cortec-hybrid
python3 -m cortec --help             # the same, when pip's scripts directory is not on PATH
```

## 2. Choose where the model runs, and set the credentials

The generator is called through one of four backends, and three of them can reach either the
vendor's public API or an enterprise surface inside your own cloud account. The `model` name is the
same on both; the surface changes only the client and the credentials.

| Backend | Surface | Set before running | `Generator(...)` arguments |
|---|---|---|---|
| `anthropic` | public API | `ANTHROPIC_API_KEY` | `backend="anthropic", model="claude-fable-5"` |
| `anthropic` | Claude on AWS Bedrock (recommended) | AWS credentials from the standard chain (environment, profile or instance role) and `AWS_REGION`; the `bedrock` extra | `surface="bedrock", surface_model="<Bedrock model id or inference-profile ARN>"` |
| `openai` | public API | `OPENAI_API_KEY` | `backend="openai", model="gpt-5"` |
| `openai` | GPT on Azure OpenAI (recommended) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, optionally `AZURE_OPENAI_API_VERSION` | `surface="azure", surface_model="<your deployment name>"` |
| `gemini` | public API | `GEMINI_API_KEY` | `backend="gemini", model="gemini-3.5-flash"` |
| `gemini` | Gemini on Google Vertex AI (recommended) | Application Default Credentials, `GOOGLE_CLOUD_PROJECT`, optionally `GOOGLE_CLOUD_LOCATION` (default `global`) | `surface="vertex"` |
| `ollama` | a local Ollama server | `ollama_url` (default `http://localhost:11434`) | **not an enterprise surface — out of deployment scope**; the capability gate refuses these models by default |
| `mock` | none | nothing | offline dry runs of the whole pipeline |

**Which endpoint to point this at.** For regulated data, use the enterprise-hosted, tenant-isolated
surface in your own cloud account (Bedrock, Azure OpenAI or Vertex AI), with private networking,
data residency, contractual exclusion of training on inputs, and a BAA where HIPAA applies. The
vendors' public developer APIs are supported but are not recommended for regulated deployment: the
weights are the same, the contract and the network path are not, and those are what a compliance
review assesses. The privacy argument does not depend on the endpoint, because the request carries
only the DP release (step 9); the compliance argument does. The paper's own measurements were
made on the public APIs; §6.2 of the technical report says which results that affects.

**What is verified.** The tests construct each SDK's real client class offline and replace only the
transport, so an SDK change fails in CI rather than at your first call.
**Vertex AI is verified live**: Gemini 3.5 Flash through a service account holding only
`roles/aiplatform.user`, both generation paths, reasoning reported on every call. The newest Gemini models are served from the
`global` location and return 404 in every regional one, and a model 404 is a fatal, non-retried
error. **What is not verified:** a live round trip on Bedrock or Azure OpenAI, because no account
was available to us. Treat your first run there as that test: generate a small batch, check
`gen.stats.calls_with_reasoning_block` and `gen.stats.thinking_tokens`, and compare the output with
the release before generating at scale.

A missing setting, or a surface that does not serve the chosen backend, fails when the `Generator`
is constructed, before any client exists. `gen.describe_surface()` returns one line for your audit
trail, and a run on a public surface says so in `gen.stats.warnings`.

## 3. Prepare the data

Stage A reads one pandas DataFrame with exactly the columns the schema declares.

- Categorical columns hold strings. Integer codes are fine as strings (`"1"`, `"2"`). Read the file
  with `keep_default_na=False` if a real category is spelled `None`, `NA` or `nan`, because pandas
  otherwise turns it into a missing value.
- Numeric columns hold numbers inside the declared bounds. Values outside are clipped, and the
  validation report says how many; a surprise there means the declared bounds are wrong.
- The target column holds exactly two labels, the ones declared as `positive` and `negative`.
- **One row per person.** ε protects one row. If a person contributes `k` rows, that person's
  guarantee is `k · ε`. Count rows per individual before Stage A; if the maximum is above 1, cap
  each person's rows or aggregate to one row per person. Step 5 explains the declaration.

Validation runs before any budget is spent and reports every problem at once.

## 4. Declare the schema

A `Schema` is a public description of the table. Everything in it must be knowledge you have
without looking at the private file: the column list, domain bounds from documentation or
convention, bin edges from clinical or regulatory practice, the target. Never derive a bound or an
edge from the data's own quantiles. A schema inferred from private values leaks through its bounds,
which is why `Schema.from_dataframe()` does not exist.

```python
from cortec import Schema, Band

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
```

| Field | What it declares | Notes |
|---|---|---|
| `name` | an identifier used in file names and the audit trail | |
| `numerical` | `{column: (lo, hi)}`, the public domain of each numeric column | values outside are clipped |
| `categorical` | `{column: [allowed values]}` | every declared value stays in the release, even when rare |
| `target`, `positive`, `negative` | the binary target and its two labels | the target is not listed as a feature |
| `bins` | `{column: [edges]}`, public histogram edges | default: 10 equal-width bins over the bounds |
| `stratify` | `[(column, [Band(name, lo, hi), ...])]`, the public rule that forms cohorts | the bands must tile the declared domain with no gap; the top band is closed at its upper edge |
| `conditional` | the levels of the conditional target table, coarse to fine, as column tuples | leave it out to have it derived (below) |
| `coarsen` | `{column: {declared value: group}}`, grouping a wide categorical for conditioning | derived by auto-configuration; may also be supplied |

**Leave `conditional` empty and the hierarchy is derived.** Auto-configuration ranks the columns by
mutual information with the target, from one DP-noised contingency table per column. It chooses how
many columns to condition on from the public cardinalities and the number of records you will
request, coarsens wide categoricals, and derives a cohort stratification on the highest-ranked
numeric column when the data can support it. The selection spends a declared share of
`epsilon_total`, between 5% and 30% depending on the schema size and the row count, and is charged
to the same ledger. Declare `conditional` yourself and it is used untouched. Auto-configuration has
no concept of a person, so on multi-row data it can select a column that is a proxy for how many
rows someone contributes; declare the columns yourself if that matters.

## 5. Stage A: release the statistics

```python
from cortec import release_statistics

release = release_statistics(schema, private_df, epsilon_total=2.0, n_min=150,
                             max_rows_per_person=1)
print(release.audit["epsilon_accounted"])      # 2.0
release.to_json("release.json")                # contains no private record
```

This is the only call that reads the private data. It forms the cohorts from the public rule;
releases one Laplace-noised histogram per column per cohort (one per outcome class where both
classes clear `n_min`), the class balance, noised cohort and cell sizes, and the conditional target
table; and records every query in a ledger.

| Parameter | Default | What it does |
|---|---|---|
| `epsilon_total` | `2.0` | the whole row-level budget; every query is charged to it |
| `n_min` | `150` | cohorts and cells with fewer records are suppressed; the floor is 50, below which a rate is dominated by its own noise |
| `conditional_fraction` | `0.2` | the share of the budget for the conditional table; the histograms take the rest |
| `max_rows_per_person` | `None` | the privacy unit: the most rows one person contributes; left out, the audit records `UNDECLARED` |
| `acknowledge_vacuous_privacy_unit` | `False` | required to proceed when `max_rows_per_person × epsilon_total` exceeds 10 |
| `class_conditional` | `True` | one histogram block per (cohort, outcome) where both outcomes clear `n_min`; `False` restores a pooled block |
| `autoconfig` | `True` | derive the hierarchy when `schema.conditional` is empty |
| `n_records` | `1000` | how many records you intend to generate; auto-configuration sizes the table against it |
| `charge_suppression` | `True` | charge the decision of which cohorts and cells appear; `False` is the literature-standard treatment, and the choice is recorded |
| `seed` | `None` | tests only: a seeded release is reproducible, so its noise can be subtracted and it carries no guarantee; the audit says so |

**The privacy unit.** ε protects one ROW, and on some data that is not one person. Under group
privacy a person contributing `k` rows receives `k · ε`. On encounter-level hospital data, one
admission per row and several per patient, this is not a technicality: in a real dataset of 101,766
encounters from 71,518 patients the heaviest patient contributes 40 encounters, so a declared ε of
2.0 is an ε of 80 for that patient, which is outside any range normally considered meaningful. So
this package is for one-row-per-person data. `k` cannot be computed for you without spending
budget, so you declare it, and a declaration whose ε per person is vacuous is refused rather than
annotated:

```python
# refused: eps_person = 80
release_statistics(schema, df, epsilon_total=2.0, max_rows_per_person=40)
# ReleaseError: REFUSED: max_rows_per_person=40 at epsilon_total=2.0 gives
#   epsilon_per_person=80.0, past the 10 beyond which a per-person guarantee carries no
#   meaning ... Either aggregate to one row per person before Stage A ...

# proceed anyway, with the acknowledgement recorded in the audit trail
release = release_statistics(schema, df, epsilon_total=2.0, max_rows_per_person=40,
                             acknowledge_vacuous_privacy_unit=True)
release.audit["epsilon_per_person"]           # -> 80.0
release.audit["epsilon_per_person_vacuous"]   # -> True
```

If your data has more than one row per person, aggregate to one row per person before Stage A, cap
each person's rows, or report `ε_person` rather than the per-row number.

**Check the release before generating.**

- `release.audit["epsilon_accounted"]` equals what you declared and `within_budget` is true. The
  audit lists every query with its epsilon, sensitivity, composition rule and partition key, so a
  privacy engineer can recompute the total by hand. `PrivacyLedger.spend()` is the only function
  in the package that returns a noise scale, so nothing is released without appearing there.
- `release.audit.get("warnings", [])` is empty, or you have read each entry. A seeded release and
  an undeclared or vacuous privacy unit are recorded there. Two further conditions are printed at
  release time rather than recorded: an under-spent conditional budget (some declared levels
  released no cell) and a noise-dominated conditional table (the noise on a level's rates is at or
  above their spread). Read the console.
- `release.conditional_levels` is non-empty and its finest level has cells. An empty table is
  refused, because the output would be unconditioned.
- Store `release.json` and hash it. Generation reads only this file, so every later draw and every
  comparison must reuse it. A release cannot be regenerated identically, because its noise is
  unseeded, and re-running Stage A spends the budget again.

## 6. Stage B: generate

```python
from cortec import Generator

gen = Generator(schema, backend="anthropic", model="claude-fable-5", reasoning="on",
                budget_usd=25.0)
synthetic = gen.generate_selected(release, n_rows=5000, pool_factor=3)
```

| Parameter | Default | What it does |
|---|---|---|
| `backend`, `model` | `"anthropic"`, `"claude-fable-5"` | the vendor and the model; the model must pass the capability gate (below) |
| `surface`, `surface_model` | `"public"`, `None` | the enterprise surface and the identifier it expects (step 2) |
| `reasoning` | `"on"` | `"suppressed"` is measured to be 3.8× worse on conditional error; the tool warns if you choose it |
| `rows_per_call` | `25` | rows requested per model call |
| `quota` | `True` | give every batch the exact per-bin, per-category and per-outcome counts it owes |
| `budget_usd` | `None` | a spend cap from the vendor's reported token counts at list prices; the run stops when it is reached and keeps what it has |
| `request_timeout`, `max_retries` | `120.0`, `3` | the per-call timeout in seconds, and the retries per batch |
| `allow_unvalidated` | `False` | run a model not in the measured set; verify transmission on your own data first |
| `acknowledge_insufficient` | `False` | run a model measured as insufficient or sweep-only, for reproducing that finding rather than for use |

Three generation methods:

- **`generate_selected(release, n_rows, pool_factor=3)`** is the default recommendation. It
  generates `pool_factor × n_rows` records cohort by cohort, keeps the `n_rows` whose cell counts
  match the release, and redraws each numeric value inside its released bin in the cohorts that
  carry class-conditional blocks (`sub_bin="generator"` keeps the model's values). Both steps read
  only the release, so they cost no privacy budget; the cost is `pool_factor` times the generation
  spend.
- **`generate(release, n_rows)`** is the same without the pool. It is adequate when relative
  ordering is all you need.
- **`generate_by_cell(release, n_rows, level=0)`** makes one call per released conditional cell
  with an integer positive count. Use it when absolute conditional rates matter, such as a
  readmission probability or a default rate. It refuses when any conditioning column has less than
  90% of its released marginal mass inside released bands, because rows outside those bands would
  be missing from the output while every aggregate check still looked healthy;
  `allow_low_coverage=True` overrides, for measurement rather than for use. Treat that guard as a
  heuristic mitigation, not a solved problem: it checks which bands the released cells span and not
  what happens inside a band, and establishing when a DP conditional table can be conveyed to a
  language model without silent loss is an open problem. Compare per-column output against the
  release anyway.

**Which model.** Capability is gated before any call, because a model too weak to follow the
release produces well-formed, plausible rows that carry none of its structure, and no output check
reveals that.

| Model | Tier | Magnitude error (transmission sweep) |
|---|---|---|
| Claude Fable 5 / Opus / Sonnet | validated | 0.009 |
| Gemini 3.5 Flash (matched by the `gemini.*flash` pattern) | validated | 0.008 |
| llama3.3:70b-instruct | **sweep-only — refused** | 0.020 |
| qwen2.5:72b-instruct | **sweep-only — refused** | 0.030 |
| mistral-small:24b, gemma2:27b, gpt-oss:20b | adequate | 0.049–0.078 |
| qwen2.5:32b, qwen2.5:14b | adequate | 0.098–0.105 |
| qwen2.5:7b | **insufficient — refused** | 0.406 |

`print(cortec.models.recommended_table())` lists the models we recommend, measured on
full-dataset generation: Gemini 3.1 Pro, GPT-5 at its default reasoning, Claude Fable 5, Claude
Opus 5 and Gemini 3.5 Flash. A sweep-only model reproduces one forced rate in a short prompt but,
on full-dataset generation, is close to indistinguishable from an unconditioned prompt; the design
notes give the measurements. An unlisted model is refused unless you pass `allow_unvalidated=True`.

**Enable reasoning, and verify that it fired.** Reasoning is the largest single effect measured. On
one model tested against itself, conditional error moved from 0.173 ± 0.032 to 0.045 ± 0.007 with
one flag changed, while downstream utility did not separate (0.786 ± 0.050 against 0.802 ± 0.025).
A cost comparison on utility alone would therefore choose the 14.5× cheaper setting and ship output
carrying none of the data's conditional structure. Setting the flag is not evidence that it fired:
some transports report no reasoning-token count at all, and a count that was never reported is not
a count of zero. The stats keep the states apart:

```python
gen.stats.thinking_tokens              # total reported reasoning tokens
gen.stats.calls_without_reasoning      # calls that REPORTED zero while reasoning="on"
gen.stats.calls_reasoning_unmeasured   # calls on which the vendor reported no count at all
gen.stats.calls_with_reasoning_block   # calls where a reasoning block was present (Anthropic)
gen.stats.positives_off_by_more_than_one   # cell-wise calls that missed the "exactly k" by > 1 row
gen.stats.warnings                     # one explicit warning for each condition above
```

**After the run**, read `gen.stats`: `calls`, `parse_ok`, `rows`, `rows_requested`,
`rows_out_of_bounds`, `rows_undeclared_category`, `rate_limit_waits`, `spend_usd`. Rows outside
the declared domain or carrying an undeclared category are dropped and counted, never kept. The
output carries one provenance column, `_cohort` (or `_cell` for cell-wise generation), naming the
released unit each row came from; drop it if your consumer expects the schema's columns only.

**Dry run first.** `backend="mock"` needs no key and honours the exact-count block, so the whole
generate-and-select loop runs end to end offline; `tests/test_selection.py` does exactly that.

## 7. Stage C: attach a utility bound

For each released cell, Stage C releases a Laplace estimate of the private target rate at a
declared budget `epsilon_cert` and turns the noise into a one-sided bound on the gap between the
private and the synthetic rates. The bound holds simultaneously over every cell with probability at
least `1 − alpha`. It is a utility bound computed under DP, and it is not a privacy audit.

```python
from cortec import bound_with_controls

report = bound_with_controls(schema, release, private_df, synthetic, holdout_df,
                             epsilon_cert=1.0, alpha=0.05, tolerance=0.15)
print(report.summary())
report.to_json("bound.json")       # readable by a third party without the private data
```

- `holdout_df` is real data that was not used to build the release. It supplies the ceiling (a real
  sample, which should clear the tolerance) and the floor (the same sample with its target
  permuted, which must not). If the ceiling fails or the floor clears, the test did not
  discriminate, `report.verdict` is `None`, and that run is not reported.
- `epsilon_cert` is spent once, through the ledger. The deployment total is
  `epsilon_release + epsilon_cert`, stated per row and per person. The Laplace scale uses the
  public floor `n_min`, never the private cell size, so the bound is pure ε-DP.
- A released cell the synthetic data never covers scores the trivial bound of 1.0, so a bound
  cannot be obtained by covering a convenient subset; cells with fewer than 20 synthetic rows are
  reported as thin. The noise is unseedable. The report's `_what_this_is` and
  `_standards_not_claimed` blocks say what it is and which privacy frameworks it must not be mapped
  to.

## 8. Evaluate the output

Never against a bare threshold. Score the synthetic data beside two references built from real
data of the same size: a real sample, which is the ceiling a synthetic method can reach at that
size, and the same sample with its target column permuted, which is the floor of data that carries
no usable information. Report fidelity and utility together, because a method can score well on
one and poorly on the other, and read the conditional measures beside the aggregate ones. Also
compare each declared categorical's full support with the release: a category present in the
release and absent from the output is a representativeness failure that no aggregate metric
reports.

The package ships this comparison. `evaluate()` scores every table you give it beside a real
sample of the same size drawn from `train` and that sample with its target permuted, on the
paper's measures (1-way total variation on the public bins, and TSTR AUC under logistic
regression, a random forest and gradient boosting on the holdout), and counts the declared
categories absent from each table. It needs scikit-learn, the `dev` extra.

```python
from cortec import evaluate

rec = evaluate(schema, {"cortec": synthetic}, train=private_df, holdout=holdout_df)
rec.show()                      # the paper's head-to-head table layout, reference rows marked
rec.save("results", "evaluation")
```

**What to expect.** On UCI Adult at n = 300, models trained on this package's output were
statistically indistinguishable from models trained on a real sample of the same size, with
differences of +0.007, −0.003 and +0.012 AUC across three students and every p > 0.18. On a
finance dataset the pooled release fell short (TSTR-LR 0.652 against a real sample's 0.695, about
94% of real-sample utility), and the configuration this package ships by default left all three
students within 0.015 AUC of the floor on both datasets. That is a result at n = 300 under one
generator family. At larger sizes the point estimates favour the real sample, because a fixed
release does not get richer as you ask for more records. Do not assume parity: measure it on your
own data against a real sample of matched size. The design notes carry the full measurements.

## 9. Where the trust boundary sits

No private record is ever sent to the model. The prompt carries only released statistics: noised
histograms, class balances and a conditional table. Generation is post-processing of a DP release,
so by post-processing immunity whoever runs the model learns nothing beyond what the release
already discloses, and the ε guarantee does not depend on where that computation happens. The
decoder may be a tenant-isolated enterprise endpoint under a BAA or an air-gapped model on your own
hardware; the argument is the same, and what differs between those is capability, not trust.

The guarantee says nothing about the model's pretraining corpus. If a private record was in it,
that happened before this tool ran. The sharper version of the concern is that conditioning on
true marginals for a narrow stratum resembles a prompt-based extraction attack, because it tells
the model which region to sample from. DP does not exclude this. In the research behind this tool,
four membership-inference attacks, each validated on a positive control, found no advantage above
chance and zero exact matches on any dataset. That is evidence about the published output, not a
proof about the corpus.

## 10. Troubleshooting

Each row below is a refusal: the tool stopping on purpose, before spending budget or writing
output it cannot stand behind. In a run script, print refusals in the standard layout instead of
a traceback by adding one line at the top, or by wrapping the run:

```python
from cortec import install_guard
install_guard()          # a refusal prints as a boxed block and exits with status 2

from cortec import guard
with guard():            # the same, for one block (works in notebooks too)
    ...
```

```
── cortec 0.4.0 · Refused · Data Validation ────────────────────────────────────────────
  refused by  DataValidationError
  reason
    column 'income' is missing from the data. NO privacy budget was spent.

  The tool stopped on purpose; this is a refusal, not a fault in the tool. Change the input
  or the setting the reason names and run again.
```

Only the classes in this table (and `cortec-hybrid`'s) are treated as refusals; any other
exception is a fault and keeps its traceback.

| What you see | Why | What to do |
|---|---|---|
| `DataValidationError` before any query | a column is missing, the target has an undeclared label or one class only, or a numeric column has no usable values | fix the data or the schema; no budget was spent |
| `SchemaError: stratify bands ... must tile its declared domain` | a gap or an overlap between bands | make the bands contiguous from `lo` to `hi` |
| `ReleaseError: n_min=... is below the safe floor of 50` | a rate over fewer records is dominated by its noise | raise `n_min`, or coarsen the conditional levels |
| `ReleaseError: no cohort reached n_min` or `the conditional target table is empty` | too few records per cohort or cell | coarsen the stratification or the levels, lower `n_min`, or supply more records |
| `ReleaseError: REFUSED: max_rows_per_person=...` | the per-person ε would exceed 10 | cap or aggregate rows per person, lower `epsilon_total`, or acknowledge a row-level guarantee |
| a printed warning that part of the conditional budget was not spent | some declared levels released no cell | use fewer levels, lower `n_min`, or supply more records |
| a printed warning that a level's noise is at or above the spread of its rates | the table is noise-dominated | raise ε, coarsen to fewer cells, or accept that conditioning will not help on this data |
| `ModelCapabilityError` | the model is unmeasured, sweep-only or insufficient | use a recommended model, or opt in with the flag the message names and verify on your own data |
| `PromptIntegrityError` | a prompt template was modified in memory, or the conditional table is empty | do not edit the templates; register a variant under a new name if you must |
| `ContextTruncationError` | the prompt plus the output budget exceeds a local model's context | raise `num_ctx` or lower `rows_per_call` |
| `EmptyContentError` | a reasoning model spent its output budget before writing any CSV | raise the output budget; `ModelRefusalError` is the different case of a refusal, which needs a coarser cell or a different generator |
| `FatalAPIError` | billing, quota, credentials, or a model the surface does not serve | fix the account or the model name; nothing was retried and the release is unchanged |
| the run waits and `rate_limit_waits` rises | the vendor asked it to wait | nothing; the waits back off from 15 s to 4 min before an error surfaces |
| a "budget cap reached" warning and fewer rows than requested | the spend cap was reached | not an error: the output holds the rows completed within the budget; raise `budget_usd` or use a cheaper validated model for the full request |
| `GenerationError` naming a yield or parse rate | most returned rows were dropped, or too few calls parsed | a genuine fault, so the run stops before spending more; check the schema against the rows being dropped |
| `select_to_release` refuses the pool | a run that stopped early without a budget cap (a lost connection) left cohorts short of the rows they owe | a budget cap is handled automatically, returning the partial pool with a warning; for other truncations pass `allow_short_pool=True` |
| `generate_by_cell` refuses the release | a conditioning column has under 90% of its mass in released bands | lower `n_min` or use a coarser level, or generate cohort-wise |

## 11. Output and exports

Every stage prints one layout and exports one record, through `cortec.report`. The layout is a
stage header, a block of named values, one or more tables, the warnings, and a verdict. On a
terminal the numbers the run produced are printed in blue, reference rows (a real sample, a
permuted floor) in grey, warnings in amber, and a verdict against the output or a refusal in
orange. Colour is used only on a terminal; `NO_COLOR` or `CORTEC_COLOR=0` turns it off and
`CORTEC_COLOR=1` forces it. The exported files never carry colour.

```python
from cortec import show, record

rec = show(release, n_private_rows=len(private_df))   # Stage A
rec = show(gen, n_rows=len(synthetic))                 # Stage B, after generate()
rec = show(report, schema=schema.name)                 # Stage C
files = rec.save("results/run-01", "stage_c")          # {name: path}
```

`show()` prints and returns a `RunRecord`; `record()` builds one without printing. `save()`
writes `<stem>.json` (the record, format `cortec-result/1`), `<stem>.md` (the same content as
Markdown), `<stem>.csv` (the named values), one CSV per table, and `<stem>.png` when matplotlib
is installed. `RunRecord.from_json()` reads a record back. Tables follow the paper's conventions:
a label column on the left, numbers to three decimal places, reference rows marked. Figures use
the paper's own rc parameters and validated palette, available to your own scripts as
`cortec.plots.paper_rcparams()`. [`docs/result-format.md`](docs/result-format.md) is the contract
for the record and lists the tables each stage writes.

## Licence

Apache-2.0.
