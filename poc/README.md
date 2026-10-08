# CoRTeC end to end: the proof of concept

This folder runs CoRTeC on two public datasets, NHANES 2017-2018 and UCI Adult, from the raw
download to a trained model, and writes every artefact under `artifacts/<dataset>/` so the paper's
two results can be checked on your own machine. It is a test feature of this repository: the
artefacts are produced by running it, with the offline mock backend or a model of your own, and
none are shipped here.

- **Marginal fidelity.** The synthetic table's 1-way error against real held-out data is
  close to what a real sample of the same size records.
- **Downstream utility.** Models trained on the synthetic table score, on real held-out
  data, close to models trained on a real sample of the same size.

Both are read against the two reference rows the tool always prints, a real sample of the same
size and the same sample with its target permuted, never against a threshold. One draw is a signal;
the paper's claims rest on three draws per dataset and Holm-corrected families.

## What a run writes

| path | what it is |
|---|---|
| `artifacts/<dataset>/REPORT.md` | the run's facts, the results panel (marginal fidelity and downstream utility against the real-sample range, the Stage C verdict, the privacy spent, the cost), the published models and how to read them |
| `artifacts/<dataset>/RESULTS.md`, `results.json` | the tool's results panel as a document and as a record |
| `artifacts/<dataset>/release.json` | the DP release (Stage A), ε = 2.0; it contains no private record |
| `artifacts/<dataset>/synthetic.csv` | the synthetic table (Stage B), 300 rows selected from a threefold pool |
| `artifacts/<dataset>/bound.json` | the Stage C transmission bound with its real-sample ceiling and permuted floor |
| `artifacts/<dataset>/stage_*.*`, `evaluation.*` | the tool's records of each stage: JSON, Markdown, CSV and a figure |
| `artifacts/<dataset>/models/*.joblib` | the three students (logistic regression, random forest, gradient boosting) trained on the synthetic table, with the evaluation's estimators and seed |
| `artifacts/<dataset>/models/scores.json` | their AUCs on the real holdout, beside the mean AUC of the same students trained on real samples of the same size and on their permuted copies |

The datasets are public benchmarks: `data.py` downloads them from the CDC and the UCI repository
and builds the same tables, with the same 80/20 split (seed 42), as the paper's research harness.
The holdout split is never shown to Stage A.

## Run it

Install the package with its evaluation extra, from the repository root:

```
pip install -e './cortec[dev]' joblib
```

**Without a key (replay).** Re-scores a synthetic table an earlier run left under
`artifacts/<dataset>/` beside fresh real references through the tool's evaluation, retrains the
students on it, scores them on the holdout you build locally, and rewrites the results panel and
the report:

```
python poc/run_poc.py --dataset nhanes --replay
python poc/run_poc.py --dataset adult --replay
```

**Without a key (verify).** Scores the models an earlier run saved on your copy of the holdout and
prints them beside the AUCs that run recorded:

```
python poc/run_poc.py --dataset nhanes --verify
```

**The full run (needs a key).** Stage A, B and C through `cortec run`, then the students and the
report. Set the key for the backend as the package README describes (for example
`GEMINI_API_KEY`), and cap the spend:

```
python poc/run_poc.py --dataset nhanes --backend gemini --model gemini-3.5-flash --budget-usd 7
python poc/run_poc.py --dataset adult  --backend gemini --model gemini-3.5-flash --budget-usd 9
```

A fresh `--out` folder means a fresh release, which spends the budget again on the same data; the
tool says so when it does. Re-running into the same folder reuses the release and generates again
at no privacy cost.

**Offline smoke test.** `python poc/run_poc.py --dataset nhanes --backend mock --out /tmp/poc_mock`
runs every step with the rule-based mock generator; its numbers mean nothing and it exists to show
the pipeline works before a key is spent.

## How to read the report

`REPORT.md` carries the tool's results panel. Its first block gives one line each for marginal
fidelity and downstream utility, the Stage C verdict, the privacy spent and the cost. The marginal
fidelity table places the synthetic table's 1-way error beside the mean over five real samples of
the same size and their permuted copies, and its reading says whether the synthetic value lies
within the range those real samples span. The downstream utility table does the same for each
student's AUC on the holdout, with the gap to the real sample and the share of real-sample utility,
(synthetic − permuted) / (real sample − permuted). The headline under the tables is the reading at
this draw, `ok` when every measure lies within or beyond the real-sample range and a warning
otherwise; it is not a pass or a fail against a threshold. The Stage C line reports the bound on the
private-versus-synthetic conditional gap over the released cells, with its derived tolerance; `no
verdict` means the controls could not discriminate or a released cell held too few synthetic rows,
and is itself the finding. The saved-models table at the end gives the AUC of each saved model
on the holdout, which `--verify` reproduces on your machine.

## What this is not

The release is pure ε-differentially private at the row level; the proof of concept makes no
privacy claim beyond the tool's own accounting record, and the Stage C bound is a utility bound, not
a privacy audit. The schemas here leave the conditional hierarchy to the tool's auto-configuration,
which is the shipped turn-key path; the paper's Adult tables used a hand-tuned hierarchy in the
research harness, so the Adult numbers here are the tool's, not Table 3's. NHANES is the paper's
auto-configured case and matches its protocol.
