#!/usr/bin/env python3
"""CoRTeC end to end on a public dataset: the proof of concept.

Marginal fidelity: the synthetic table's 1-way error against real held-out data is within the
range real samples of the same size record. Downstream utility: models trained on the
synthetic table score, on real held-out data, within the range of models trained on real samples
of the same size. Both are read against references built from real data, a real sample (the
ceiling at that size) and the same sample with its target permuted (the floor of no information),
never against a threshold. The tool's results panel states both readings; this script adds the
trained models, publishes them, and writes the report.

Three ways to run it:

  python poc/run_poc.py --dataset nhanes --backend gemini --model gemini-3.5-flash --budget-usd 7
      the full run: data, Stage A (the release, the only step that reads private data), Stage B
      (generation through a frozen model), Stage C (the DP transmission bound with its controls),
      the evaluation and the results panel through `cortec run`; then the three students trained
      on the synthetic table, saved as joblib files, and REPORT.md. Needs an API key (README).

  python poc/run_poc.py --dataset nhanes --replay
      no key: uses the release and synthetic table an earlier run left under poc/artifacts/<dataset>/,
      re-scores them beside fresh real references, retrains the students and rewrites the report.

  python poc/run_poc.py --dataset nhanes --verify
      no key: loads the saved models and scores them on the real holdout, so the AUCs an earlier
      run recorded can be checked on your machine.

Every artefact is public: the release contains no private record, the synthetic table is the
tool's output, the holdout is a public benchmark's held-out split.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:                                   # installed with `pip install -e './cortec[dev]'` (README)
    import cortec  # noqa: F401
except ImportError:                    # or run from a clone without installing: the package sits beside poc/
    sys.path.insert(0, str(HERE.parent / "cortec"))
from data import build  # noqa: E402

STUDENTS = ("LR", "RF", "GBM")
STUDENT_NAMES = {"LR": "logistic regression", "RF": "random forest", "GBM": "gradient boosting"}
DATASETS = {
    "nhanes": ("NHANES 2017-2018 (CDC)", "diabetes, HbA1c at or above 6.5%, from age, body measures, blood "
                                         "pressure, sex, race or ethnicity, education and income"),
    "adult": ("UCI Adult (1994 census)", "income above 50K from age, work class, education, marital status, "
                                         "occupation, relationship, race, sex, capital gains and losses, hours "
                                         "and native country"),
}


def load_schema(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"poc_schema_{name}", HERE / "schemas" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.SCHEMA


def _xy(schema, df: pd.DataFrame):
    num, cat = schema.numerical_cols, schema.categorical_cols
    x = df[num + cat].copy()
    for c in num:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    y = (df[schema.target].astype(str).str.strip() == schema.positive).astype(int)
    return x, y


def train_students(schema, train_df: pd.DataFrame, *, seed: int = 0) -> dict:
    """The three students of the paper's evaluation, as fitted scikit-learn pipelines; the same
    estimators, settings and seed as `cortec.evaluate`, so their AUCs are the evaluation's."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num, cat = schema.numerical_cols, schema.categorical_cols
    x, y = _xy(schema, train_df)
    if y.nunique() < 2:
        raise SystemExit("the training table holds one class only; no student can be fitted")
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, clf in (("LR", LogisticRegression(max_iter=2000)),
                          ("RF", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
                          ("GBM", HistGradientBoostingClassifier(random_state=seed))):
            pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
            models[name] = make_pipeline(pre, clf).fit(x, y)
    return models


def score(schema, models: dict, holdout: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    xt, yt = _xy(schema, holdout)
    return {n: float(roc_auc_score(yt, m.predict_proba(xt)[:, 1])) for n, m in models.items()}


def save_models(models: dict, folder: Path) -> list[str]:
    import joblib
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for n, m in models.items():
        p = folder / f"{n.lower()}.joblib"; joblib.dump(m, p); out.append(str(p))
    return out


def load_models(folder: Path) -> dict:
    import joblib
    return {n: joblib.load(folder / f"{n.lower()}.joblib") for n in STUDENTS}


def _record(out: Path, stem: str):
    """A stage record from the folder, or None when the file is not there."""
    from cortec.report import RunRecord
    p = out / f"{stem}.json"
    return RunRecord.from_json(str(p)) if p.exists() else None


def _utility_reference(results) -> dict:
    """The real-sample and permuted AUCs of the results panel, by student."""
    t = results.table("downstream utility result") if results is not None else None
    if t is None:
        return {}
    return {r[0]: {"real_sample": r[2], "permuted": r[3]} for r in t.rows}


def write_report(out: Path, dataset: str, results, facts: list, saved: dict, verify_cmd: str) -> Path:
    """REPORT.md: the run's facts, the results panel, the saved models and how to read it."""
    title, task = DATASETS[dataset]
    lines = [f"# CoRTeC proof of concept: {title}", "",
             f"Task: {task}. Every number below is one draw of one run; a signal, not a claim.", "",
             "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in facts]
    lines += ["", "## Results: marginal fidelity and downstream utility against real references", ""]
    if results is not None:
        body = results.to_markdown().split("\n", 3)[3]      # drop the record's own heading and time line
        lines += [body.strip(), ""]
    else:
        lines += ["The results panel was not written; run without --replay, or --replay in the folder of an earlier run.", ""]
    lines += ["## The saved models", "",
              "The three students were trained on the synthetic table alone (`synthetic.csv`), with the same "
              "estimators, settings and seed as the tool's evaluation, and saved under `models/`. Their AUCs on "
              "the real holdout are beside the mean AUC of the same students trained on real samples of the "
              "same size and on their permuted copies.", "",
              "| student | file | AUC on the holdout | real sample (mean) | permuted (mean) |",
              "|---|---|---:|---:|---:|"]
    for st in STUDENTS:
        ref = saved.get("reference", {}).get(st, {})
        f = lambda x: "" if x is None else f"{x:.3f}"   # noqa: E731
        lines.append(f"| {STUDENT_NAMES[st]} ({st}) | `models/{st.lower()}.joblib` | {f(saved['synthetic'].get(st))} "
                     f"| {f(ref.get('real_sample'))} | {f(ref.get('permuted'))} |")
    lines += ["", f"Check them on your own copy of the holdout with `{verify_cmd}`; it rebuilds the holdout from "
              "the public source, scores the saved files and prints both numbers side by side.", "",
              "## How to read this", "",
              "**Marginal fidelity.** The 1-way total-variation distance between the synthetic table and "
              "the holdout, averaged over every column, beside what real samples of the same size record on the "
              "same holdout. The synthetic value is read against the range those real samples span: inside it, "
              "the synthetic table is indistinguishable from a real sample of that size on this measure.", "",
              "**Downstream utility.** Train on the table, test on the real holdout: the AUC of logistic "
              "regression, a random forest and gradient boosting. Each synthetic-trained student is read against "
              "the range of the same student trained on real samples of the same size, and against the permuted "
              "floor, the AUC of a student trained on data whose target carries no information. The share of "
              "real-sample utility is the part of a real sample's utility above that floor that the synthetic "
              "table reaches.", "",
              "**Stage C.** A differentially private bound on how far the synthetic conditional rates can lie from "
              "the private ones over the released cells, with its own two controls; it bounds utility and is not "
              "a privacy audit. A `no verdict` means the controls could not discriminate at this size and is "
              "itself the finding.", "",
              "The paper's claims rest on three draws per dataset and Holm-corrected comparisons; one run here is "
              "a signal. The release (`release.json`) is the only artefact built from private rows and holds "
              "noised counts and rates only.", ""]
    p = out / "REPORT.md"; p.write_text("\n".join(lines)); return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("nhanes", "adult"), default="nhanes")
    ap.add_argument("--backend", default="mock", help="mock (offline), anthropic, openai, gemini, ollama")
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--surface", default="public")
    ap.add_argument("--budget-usd", type=float, default=None, help="hard spend cap for Stage B")
    ap.add_argument("--effort", default=None, choices=("low", "medium", "high"), help="thinking level for an Ollama reasoning model")
    ap.add_argument("--max-output-tokens", type=int, default=None, help="output budget per call, reasoning included")
    ap.add_argument("--allow-unvalidated", action="store_true", help="run a model the capability gate has not measured")
    ap.add_argument("--request-timeout", type=float, default=None, help="seconds to wait for one model call (default 120)")
    ap.add_argument("--n-rows", type=int, default=300)
    ap.add_argument("--pool-factor", type=int, default=3)
    ap.add_argument("--epsilon-cert", type=float, default=1.0)
    ap.add_argument("--reference-draws", type=int, default=5, help="real samples the references are drawn over")
    ap.add_argument("--out", default=None, help="default poc/artifacts/<dataset>")
    ap.add_argument("--replay", action="store_true", help="use the release and synthetic table of an earlier run; no key")
    ap.add_argument("--verify", action="store_true", help="score the saved models on the holdout; no key")
    a = ap.parse_args()
    from cortec.report import ResultTable, banner, emit, files_block, fmt, fmt_seconds, header, note, record_results

    t0 = time.time()
    out = Path(a.out) if a.out else HERE / "artifacts" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    schema = load_schema(a.dataset)
    train_csv, holdout_csv = build(a.dataset)
    cols = schema.columns
    train = pd.read_csv(train_csv, keep_default_na=False)[cols]
    holdout = pd.read_csv(holdout_csv, keep_default_na=False)[cols]
    title, task = DATASETS[a.dataset]
    mode = "verify the saved models" if a.verify else ("replay the saved artefacts" if a.replay else "full run")
    emit(banner(f"proof of concept · {a.dataset}", [
        ("dataset", f"{title}: {task}"),
        ("private rows", f"{len(train):,} (read once, by Stage A)"),
        ("holdout rows", f"{len(holdout):,} (never shown to Stage A; every score is on these)"),
        ("mode", mode), ("folder", str(out))]))
    emit("")

    if a.verify:
        models = load_models(out / "models")
        got = score(schema, models, holdout)
        pub = json.loads((out / "models" / "scores.json").read_text())
        rows = [[STUDENT_NAMES[n] + f" ({n})", pub["synthetic"][n], got[n], got[n] - pub["synthetic"][n]] for n in STUDENTS]
        emit(header("Proof of concept", "saved models scored on your copy of the holdout"))
        emit("")
        emit(ResultTable("saved models", ["student", "recorded AUC", "AUC here", "difference"], rows,
                         subtitle="The saved joblib files, loaded and scored on the holdout rebuilt from the "
                                  "public source on this machine.",
                         note="a difference of zero means the holdout rebuilt here is the one the models were "
                              "scored against; a small one means the public file changed").render())
        return 0

    if not a.replay:
        from cortec.run import main as cortec_run
        argv = ["--data", str(train_csv), "--schema", str(HERE / "schemas" / f"{a.dataset}.py"),
                "--holdout", str(holdout_csv), "--backend", a.backend, "--model", a.model, "--surface", a.surface,
                "--n-rows", str(a.n_rows), "--pool-factor", str(a.pool_factor), "--epsilon-cert", str(a.epsilon_cert),
                "--reference-draws", str(a.reference_draws), "--out", str(out), "--exports", "all"]
        if a.budget_usd:
            argv += ["--budget-usd", str(a.budget_usd)]
        if a.effort:
            argv += ["--effort", a.effort]
        if a.max_output_tokens:
            argv += ["--max-output-tokens", str(a.max_output_tokens)]
        if a.allow_unvalidated:
            argv += ["--allow-unvalidated"]
        if a.request_timeout:
            argv += ["--request-timeout", str(a.request_timeout)]
        rc = cortec_run(argv)
        if rc:
            return rc
        emit("")
    for needed in ("release.json", "synthetic.csv"):
        if not (out / needed).exists():
            raise SystemExit(f"{out / needed} is missing; run without --replay (needs a key) or point --out at an earlier run's folder")
    synthetic = pd.read_csv(out / "synthetic.csv", keep_default_na=False)[cols]
    rec_a, rec_b, rec_c = _record(out, "stage_a_release"), _record(out, "stage_b_generate"), _record(out, "stage_c_bound")
    results = None if a.replay else _record(out, "results")
    if results is None:
        # replay: the saved table beside fresh real references, through the tool's own evaluation
        from cortec import evaluate
        rec_e = evaluate(schema, {"synthetic": synthetic}, train=train, holdout=holdout,
                         reference_draws=max(1, a.reference_draws))
        rec_e.show()
        rec_e.save(str(out), "evaluation")
        emit("")
        results = record_results(evaluation=rec_e, bound=rec_c, generation=rec_b, release=rec_a,
                                 schema=schema.name)
        results.show()
        results.save(str(out), "results")
        results.write_markdown(str(out / "RESULTS.md"))
        emit("")

    # the saved models: the students trained on the synthetic table, saved, scored on the holdout
    emit(header("Proof of concept", "the saved models"))
    t1 = time.time()
    models = train_students(schema, synthetic, seed=0)
    paths = save_models(models, out / "models")
    syn_scores = score(schema, models, holdout)
    reference = _utility_reference(results)
    rows = [[STUDENT_NAMES[n] + f" ({n})", syn_scores[n], reference.get(n, {}).get("real_sample"),
             reference.get(n, {}).get("permuted"), f"models/{n.lower()}.joblib"] for n in STUDENTS]
    emit("")
    emit(ResultTable("saved models", ["student", "AUC on the holdout", "real sample", "permuted", "file"], rows,
                     subtitle="Trained on the synthetic table alone, with the evaluation's estimators and seed, "
                              "and saved as joblib files anyone can load and score.",
                     note="real sample and permuted are the means from the results panel above").render())
    gen_vals = dict(rec_b.values) if rec_b is not None else {}
    backend = str(gen_vals.get("backend", f"{a.backend} / {a.model}"))
    (out / "models" / "scores.json").write_text(json.dumps({
        "synthetic": syn_scores, "reference": reference, "holdout_rows": int(len(holdout)),
        "trained_on_rows": int(len(synthetic)), "generator": backend,
        "reference_draws": int(a.reference_draws), "seconds": round(time.time() - t1, 1)}, indent=2))
    rel_vals = dict(rec_a.values) if rec_a is not None else {}
    facts = [("private rows (Stage A reads them once)", f"{len(train):,}"),
             ("holdout rows (every score is on these)", f"{len(holdout):,}"),
             ("release", f"epsilon {fmt(rel_vals.get('epsilon accounted', 2.0))}, n_min {rel_vals.get('n_min', '')}, "
                         f"{rel_vals.get('cohorts', '')} cohorts, {rel_vals.get('conditional levels', '')} conditional levels"),
             ("generator", backend),
             ("synthetic rows", f"{len(synthetic):,}" + (f", selected from a pool of {gen_vals['rows kept']:,}"
                                                         if gen_vals.get("rows kept") and gen_vals["rows kept"] > len(synthetic) else "")),
             ("cost", f"{gen_vals.get('calls', '?')} model calls, ${float(gen_vals.get('spend (USD)', 0) or 0):.2f}"),
             ("reference draws", f"{a.reference_draws} real samples and their permuted copies"),
             ("recorded", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    verify_cmd = f"python poc/run_poc.py --dataset {a.dataset} --verify"
    rep = write_report(out, a.dataset, results, facts, {"synthetic": syn_scores, "reference": reference}, verify_cmd)
    emit("")
    emit(files_block("what the proof of concept adds to the folder", [
        ("folder", str(out)),
        ("REPORT.md", "the run's facts, the results panel, the saved models and how to read them"),
        ("models/", "lr.joblib, rf.joblib, gbm.joblib (trained on the synthetic table) and scores.json"),
        ("RESULTS.md", "the tool's results panel as a document (results.json is the same as a record)")]))
    emit(note(f"check the saved models on your machine: {verify_cmd}"))
    emit(note(f"{fmt_seconds(time.time() - t0)} in all"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
