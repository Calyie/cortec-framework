# Deploying CoRTeC: the operational manual

This manual is for the team that runs CoRTeC on regulated data. It is ported from Appendix G and
Appendix H.8 of the CoRTeC technical report (`paper/CoRTeC.md` in the research repository), which
is the authoritative source; section references of the form §x.y and bracketed citations [n] refer
to that report, and the arXiv paper carries a condensed version in its Appendix C. Every control
here is enforced in the `cortec` package rather than documented as advice, and each failure mode
corresponds to a regression test in `tests/` named after the defect it prevents.

Read it in this order. Section 1 is the architecture on one page. Section 2 is the checklist, step
by step. Sections 3 and 4 are what a compliance review asks for: the controls mapped to standards,
and the failure modes with the guardrail for each. Sections 5 and 6 are the two deployment
patterns and the cost model.

## 1. The architecture on one page

Three trust zones, and one artefact that crosses between them.

**Zone 1, regulated.** The institution's own VPC or premises. It holds the system of record (the
EHR, core banking or registry table with PHI or PII), the schema declaration (column list, public
bounds, public bin edges, target, and the maximum rows per person), and Stage A, the DP release
engine. Stage A forms the cohorts from the public stratification rule at ε = 0, releases DP
histograms per cohort and class, the DP conditional table and DP sizes, charges auto-configuration
where it is used, and writes the release artefact `R`: noisy counts only, with an audit trail
recording every query, its ε, its sensitivity and its composition rule. No private record leaves
this zone.

**The trust boundary.** Only `R` crosses. Everything below it is post-processing and adds nothing
to ε.

**Zone 2, the tenant-isolated model platform.** Bedrock, Azure OpenAI or Vertex AI in the
institution's own account, with private networking, no training on inputs, data residency and a
BAA. It runs the guardrails (the hash-locked prompt, the capability gate, the domain and category
checks, the coverage guards, the spend cap, rate-limit backoff and yield checkpoints) and Stage B:
generation from `R` by a frozen model with reasoning enabled, as exact-count batches per cohort,
then a k× pool, selection to `R`, and the sub-bin redraw. It is repeatable without limit, and ε is
unchanged.

**Zone 3, synthetic data, freely shareable.** The synthetic dataset, at any size and with unlimited
redraws; Stage C, the utility transmission bound, which is a utility claim under DP and not a
privacy audit; and the release package (the dataset, the audit trail, the bound and the DP claim
block) for training, vendor evaluation or sharing.

## 2. The deployment checklist

Each step says what to do, why, and which call or setting does it.

**Step 1. Declare the schema from public knowledge only.** The column list, the domain bounds, bin
edges from clinical or regulatory convention (WHO BMI categories, ACC/AHA blood-pressure stages),
and the target. Never from quantiles of your own file. Call: `Schema(...)`.

**Step 2. Establish the privacy unit, and reduce it if you must.** Count rows per individual. If
the maximum `k` is 1, `ε_person = ε_row` and there is nothing to do. If it is greater,
`ε_person = k · ε_row`, and on encounter-level data that is routinely vacuous, so, in this order:

- **(a) Cap each person's contribution** to their first `C` rows before Stage A, choosing the
  smallest `C` whose `ε_person = C · ε_row` your regulator will accept. This is pure preprocessing.
  It leaves the quantity you are estimating alone, and on Diabetes 130 it is 6.3× more accurate
  than aggregating at the same ε_person (§4.4).
- **(b) Aggregate to one row per person before Stage A** only when a per-patient quantity is what
  you actually want. It fixes `k` exactly, but it changes the estimand, and nothing downstream will
  tell you so.
- **(c) Failing either, report `ε_person = max_rows · ε_row`** rather than the per-row number, and
  make no per-person claim.

After capping, re-check released-cell coverage (step 5): capping shrinks cells, and cells that fall
under `n_min` stop being released. This step is the answer to the ε_person = 80 figure of §4.4,
and it is enforced rather than recommended: `certify.py` requires `--max-rows-per-person` and will
not produce a bound report without it, and the reference implementations refuse a declared privacy
unit whose `ε_person` is vacuous unless the caller acknowledges it into the audit trail. Call:
`release_statistics(..., max_rows_per_person=k)`.

**Step 3. Choose ε and n.** Output quality was essentially unchanged over ε ∈ [0.3, 8] on Diabetes
130 (81,410 records). On NHANES (2,999 records) the shipped configuration lost a fifth of its
utility between ε = 2 and ε = 0.3, and the release's own noise-to-signal ratio predicts this before
generation (§7.6, §H.2). Pick the tightest budget your regulator will accept, and read the
release-time check of step 5 before generating. Settings: `epsilon_total`, `n_records`.

**Step 4. Run Stage A once.** Store `R` and its audit trail as the controlled artefact. Hash it. `R`
cannot be regenerated: the noise source is cryptographically secure and ignores any seed you set,
which we verified directly, so the stored artefact is the only copy of that release. Every draw and
every comparison must reuse it by file, not by re-running Stage A. Call: `release_statistics(...)`,
`release.to_json(path)`.

**Step 5. Check the release before generating.** Is the finest conditional table non-empty? Is the
noise-to-signal ratio below 1? How many cells cleared `n_min`, and for each conditioning column,
what share of its mass lies in bands the surviving cells actually name? Cell-wise generation
produces essentially nothing outside them, so an uncovered band silently deletes that slice of the
population (§7.10). This is computable from the release itself, so the check costs no budget. Read:
`release.conditional_levels`, the release-time warnings, and the coverage guard's refusal message.

**Step 6. Select a reasoning-capable model on an enterprise-hosted, tenant-isolated platform, and
enable reasoning.** The scope note at the head of the technical report applies here: the
recommendation is Claude on Bedrock, GPT on Azure OpenAI, or Gemini on Vertex AI, in the
institution's own account, under the institution's own contract. The vendors' public developer
APIs are not a substitute: the model is the same, the contract is not, and the contract is what a
compliance review assesses. Our own measurements were made on those public APIs; §6.2 says so and
says exactly which of our results that does and does not affect. Open-weight models appear in the
technical report as scientific controls only. None of those we measured was run with a reasoning
mode engaged (seven have none; `gpt-oss:20b` has one and ran uninstrumented at its default), and
§H.15.3 reports what that costs. Verify that reasoning fired, not that you asked: read the
reasoning-token count, and treat "no count reported" as unverified rather than as zero, because
§7.5 documents a transport on which the count silently disappears. Budget for it: this is where the
cost is. Settings: `Generator(backend=..., model=..., surface=..., reasoning="on")`; read
`gen.stats.calls_with_reasoning_block` and `gen.stats.calls_reasoning_unmeasured`.

**Step 7. Generate**, reusing `R`. Draw as many datasets as you need; they are free in ε. Call:
`gen.generate_selected(release, n_rows, pool_factor=3)`.

**Step 8. Evaluate against floors and a ceiling**, never against a bare threshold: a real sample at
matched n, and the same data with the target permuted. Report fidelity and utility together. Also
compare each declared categorical's full support against the release: a category present in the
release and absent from the output is a representativeness failure that no aggregate metric
reports.

**Step 9. Bound** (Stage C), and attach the utility bound report, the audit trail and the DP claim
block to the release package. Call: `bound_with_controls(...)`, `report.to_json(path)`.

**Step 10. Re-verify after any change to the generator or its configuration.** A settings change
alone moved conditional error by a factor of 3.8 in our measurements.

## 3. Controls, mapped to standards

| Control | Implementation | Standard |
|---|---|---|
| Stated DP parameters | variant (pure ε-DP, central), δ = 0, neighbouring relation (add/remove, unbounded), **privacy unit**, composition rules, mechanism, ε per stage, **ε per person** | NIST SP 800-226 [34] |
| Auditable accounting | `PrivacyLedger.spend()` is the sole source of noise scales; every query records ε, sensitivity, composition rule and partition key; ledger **seals** after release | NIST SP 800-226 [34] |
| Documented gaps | floating-point Laplace (Mironov [32]), pretraining provenance, privacy unit, uncharged-suppression assumption if opted into: carried **in the release audit itself**, not only in this table | NIST SP 800-226 [34] §"where the guarantee does not hold" |
| Documented utility claim in the release package | Stage C **utility transmission bound**: bounded claim at stated confidence and stated ε_cert. **Not a disclosure review**: it measures fitness for use, not disclosure risk | NIST SP 800-188 [33] (governance and documentation) |
| Re-identification risk | aggregates over cells of ≥ n_min; **zero exact matches** measured on all four datasets; four attacks validated on a positive control | ISO/IEC 27559 [23], ISO/IEC 20889 [22] |
| Data protection by design | private data never leaves Zone 1; prompt carries only `R` | GDPR [16] Art. 25 / Recital 26 |
| Expert determination route | DP release (ε stated per person) + measured re-identification evidence: **not** the Stage C bound, which speaks to utility only | HIPAA Expert Determination [49] |
| No training on inputs | contractual, via the tenant-isolated platform | vendor terms / BAA |

Three notes on reading this table.

- **The Expert Determination row deliberately excludes Stage C.** HIPAA Expert Determination
  requires a statistical assessment of re-identification risk. The Stage C bound is a statement
  about how much conditional structure survived generation, a utility quantity, and mapping it to
  that route would be a category error with compliance consequences. The privacy weight in that
  row is carried by the DP release and the measured re-identification evidence alone.
- **The utility-claim row is labelled carefully for the same reason.** A disclosure review assesses
  disclosure risk, which is a privacy function the Stage C bound does not perform. Calling the bound
  a disclosure review would repeat the Expert Determination error one row up. The bound is the
  documented utility half of a release package, fitness for use, budgeted and stated; the
  disclosure-risk half is carried by the re-identification row.
- **The bound report enforces that exclusion rather than relying on this table.** Its artefact
  claims alignment with NIST SP 800-226 and SP 800-188 only, and carries a `_standards_not_claimed`
  block that names HIPAA Expert Determination, ISO/IEC 27559, ISO/IEC 20889 and GDPR Art. 25
  explicitly, with the reason for each and a pointer to where that weight actually sits. Naming
  them is better than omitting them: a compliance reader who sees a short list cannot otherwise tell
  a considered exclusion from an oversight.

Two things this architecture does not do, and a reviewer will ask. It does not make the generator's
pretraining corpus part of the guarantee; if a private record was in that corpus it was compromised
before CoRTeC ran. And the Stage C output is a utility transmission bound computed under DP: it
bounds how much conditional structure was transmitted, and it must never be presented as a privacy
audit.

## 4. Operational failure modes, and the guardrail for each

These are not hypothetical. Each corresponds to a defect that produced a plausible but wrong result
during this work, and each is enforced in code rather than documented as advice.

| Failure mode | What it looks like | Guardrail |
|---|---|---|
| **Generator below the capability floor** | well-formed, plausible records carrying none of the release's structure | capability gate: measured-insufficient models refused; unmeasured models require an explicit override |
| **Reasoning suppressed** | conditional error at or below the no-information floor, while aggregate utility still reads ~95% | set reasoning effort explicitly; verify it **fired**: a reported count of zero and no count at all are different states, and some transports report neither (§7.5) |
| **A second DP release fired silently** | generating into a fresh output directory re-spends the whole budget; output is indistinguishable from correct | loud banner on a fresh release; reuse path prints a confirmation; **verify the release artefact by hash before every run** |
| **Context truncation** | the conditional table is cut off; looks like "this model ignores the statistics" | context-fit check before the call; `num_ctx` set explicitly |
| **Reasoning starves the output** | empty content, no CSV; looks like "this model cannot follow the schema" | empty-content detection with a specific diagnostic; stream above the token threshold |
| **Missing-value convention** | `'None'` read as NaN; a later dropna deleted 83% of rows at a 100% call-success rate | `keep_default_na=False`; assert the affected category's share after loading |
| **Out-of-domain rows** | a credit limit of 34 against a declared floor of 10,000 | domain-bounds rejection (bounds are public, so filtering costs no ε) |
| **Whole subpopulations absent** | cell-wise generation emits nothing outside released cells; four of six racial groups at exactly 0.000 while fidelity *and* utility dashboards stay green | **per-column coverage guard** (*partial: see §H.7*): every conditioning column must have ≥ 90% of its released marginal mass inside released bands, or the cell path is refused; read from the release, so the check costs no ε. It catches band erasure and **does not** catch destruction inside a covered coarsened category (§7.10) |
| **Uniform row allocation** | a 2%-of-population cohort supplies 25% of rows | allocation proportional to released mass |
| **Noise-dominated conditional table** | conditioning cannot help, and nothing says so | release-time warning when Laplace sd ≥ the entire spread of released rates (§H.2) |
| **Budget silently under-spent** | release noisier than the caller paid for | warning naming the shortfall; viable-level budget split (§7.7) |
| **Spend runaway** | credits exhausted mid-run | per-run cap, 2-call parse/yield checkpoints, partial-output preservation |
| **A pool that stopped early** | a run hits its cap or loses its connection; selection fills the early cohorts and starves the late ones while every per-row check passes (§7.12) | pool-coverage guard: every released cohort must hold 1.1× the rows it owes, or the pool is refused and the short cohorts named |
| **A category the schema does not declare** | one batch writes an integer code with a decimal point; concatenation re-types the whole pool and every consumer sees a category the data lacks | parser normalises to the declared strings and drops undeclared values, counted like out-of-domain rows |
| **A transient rate limit counted as failure** | two "try again later" responses in a row abort a run under the yield rule | rate limits waited out with exponential backoff before they count; billing and credential faults stay fatal |
| **Stratification bands with a gap** | records at the domain maximum fall into a cohort no generated row can join | bands must tile the declared domain; the top band is closed at its upper edge |

## 5. Two deployment patterns

**Pattern A, CoRTeC.** The institution has enterprise model access and no public transfer set. This
gives the highest standalone conditional fidelity and downstream utility; the cost is per record.

**Pattern B, hybrid correction.** The institution already runs AIM or MST and will not put a
language model in the data path at all. Keep the existing synthesiser; release a DP conditional
table and relabel only its target column, with the `cortec-hybrid` package. This is pure
post-processing of two already-DP artefacts. §8.2 reports what it does and does not achieve,
including three datasets where it reliably improves calibration and reliably reduces downstream
AUC.

## 6. Computational and monetary cost

| method | fit cost | per-record generation cost | scaling behaviour |
|---|---|---|---|
| MST | 23–135 s | free after fit | stable across our datasets |
| **AIM** | 23–44 min (Adult) · 56 s (Titanic) · 16–29 min (finance, 12-column schema) · **> 3 h, did not complete** (healthcare; finance on all 15 columns) | free after fit | governed jointly by attributes, rows, and correlated heavy-tailed columns (§F.2.2) |
| PATE-CTGAN | 42–65 min | free after fit | stable |
| DP-CTGAN | **2.2 h** (Adult) | free after fit | — |
| **CoRTeC (Gemini 3.5 Flash)** | **seconds** (the DP release) | **$1.03 / 1,000 records** | linear in records requested |
| **CoRTeC (Gemini 3.5 Flash), shipped configuration (§7.12)** | **seconds** | $1.4–2.3 / 1,000 generated rows (NHANES, Adult, finance); ×2–3 for the pool, **$4.3–7.0 / 1,000 kept records** | linear; reasoning tokens per call roughly double under exact counts |
| **CoRTeC (Claude Fable 5)** | **seconds** | $4.84 / 1,000 records | linear |
| **CoRTeC (Gemini 3.1 Pro)** | **seconds** | $5.21 / 1,000 records | linear |
| CoRTeC (GPT-5, reasoning on) | seconds | $9.36 / 1,000 records | linear |

Generation costs are measured from the token counts each vendor reported over our own runs, priced
at list rates. They are indicative rather than a benchmark. The shipped configuration's row is the
NHANES, Adult and finance runs of §7.12 (nine draws, eight through Vertex AI and one through the
public API, $16.6 in all): the exact-count prompt roughly doubles the reasoning tokens a call spends,
and the pool multiplies the rows generated per row kept. The figures exclude the reasoning-suppressed
configurations of §7.5, because those produce output at or below the no-information floor and their
apparent cheapness is not a saving.

**The cheapest generator we measured is not the weakest.** Gemini 3.5 Flash records TSTR-LR 0.568
against the frontier models' 0.578 on healthcare, with the lowest 1-way total variation of any
CoRTeC configuration (0.035) and conditional error matching Fable 5's (0.010 against 0.009), at one
fifth of the cost. It also spends more reasoning per call than the Pro tier, which is consistent with
§7.5: what the method needs is a reasoning-capable generator, not an expensive one. A deployment
generating 100,000 records pays roughly $103 rather than $521.

**The asymmetry, in both directions.** AIM's and MST's cost is one-off and then sampling is free,
which is a real and decisive advantage at the scale of millions of records. CoRTeC's release takes
seconds, its generation is trivially parallel and resumable, it does not constrain where the compute
runs, and AIM's fit did not complete within three hours on either regulated-industry dataset.

**One reporting discipline on spend.** Our own spend meter under-counted by roughly 1.85× at one
point, for two reasons worth passing on: one vendor bills reasoning tokens separately from output
tokens and they must be added, and a price-table prefix match charged a cheap model at a frontier
model's rate (6.7× over). Report tokens, and treat any dollar figure as an estimate.
