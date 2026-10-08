"""`cortec run`: the whole pipeline on your own table, from any directory once the package is
installed. Stage A (release), Stage B (generate), Stage C (bound), then the evaluation beside
real references; every stage printed in the standard layout, every file under one folder that is
named at the end.

  cortec run --data private.csv --schema my_schema.py --backend mock
  cortec run --data private.csv --schema my_schema.py \\
      --backend anthropic --model claude-fable-5 --n-rows 300 --budget-usd 5
  cortec run --data private.csv --holdout holdout.csv --schema my_schema.py \\
      --backend gemini --model gemini-3.5-flash --n-rows 1000

`my_schema.py` defines `SCHEMA = Schema(...)` from public facts only (README step 4). `--data`
is the private table, one row per person, with the schema's columns (README step 3). Without
`--holdout`, one fifth of the data is set aside before Stage A, for the Stage C ceiling and the
evaluation; it is never used for the release. Stage A runs once per output folder and stores
release.json there; a later run into the same folder reuses it, because a release cannot be
regenerated identically and re-running Stage A would spend the budget again.

`--backend mock` needs no key and no spend and runs the whole pipeline offline, so run it first.
A refusal by the tool prints as a boxed block and exits with status 2 (README step 10).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import pandas as pd


def load_schema(path: str):
    """The `SCHEMA` object a user's schema file defines."""
    if not os.path.isfile(path):
        sys.exit(f"schema file not found: {path} (looked from {os.getcwd()})")
    spec = importlib.util.spec_from_file_location("user_schema", path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot import {path}; it must be a Python file that defines SCHEMA = Schema(...)")
    folder = os.path.dirname(os.path.abspath(path))
    if folder not in sys.path:               # so the schema file can import a sibling module
        sys.path.insert(0, folder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "SCHEMA"):
        sys.exit(f"{path} must define SCHEMA = Schema(...); see README step 4")
    return module.SCHEMA


def read_table(path: str, columns: list[str]) -> pd.DataFrame:
    """A CSV with the schema's columns, categories kept as the strings they are."""
    if not os.path.isfile(path):
        sys.exit(f"table not found: {path} (looked from {os.getcwd()})")
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        sys.exit(f"{path} lacks the schema's column(s) {missing}; it has {list(df.columns)}")
    return df[columns]


def save_record(rec, out: str, stem: str, exports: str) -> dict:
    """The stage's record on disk: the JSON alone (`exports="json"`), which holds everything
    and renders to Markdown, CSV and a figure later through `RunRecord.from_json`, or all of
    those now (`exports="all"`). Returns {name: path}."""
    if exports == "all":
        return rec.save(out, stem)
    path = os.path.join(out, stem + ".json")
    rec.to_json(path)
    return {"json": path}


def parser(prog: str = "cortec run") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the private table (CSV), one row per person")
    ap.add_argument("--schema", required=True, help="a Python file that defines SCHEMA = Schema(...)")
    ap.add_argument("--holdout", help="real rows not used for the release (CSV); default: one fifth of --data")
    ap.add_argument("--backend", default="mock", help="mock (offline, default), anthropic, openai, gemini, ollama")
    ap.add_argument("--model", default="claude-fable-5", help="the model name; must pass the capability gate")
    ap.add_argument("--surface", default="public", help="public (default), bedrock, azure or vertex (README step 2)")
    ap.add_argument("--surface-model", default=None, help="the identifier the enterprise surface expects")
    ap.add_argument("--effort", default=None, choices=("low", "medium", "high"),
                    help="the thinking level for a reasoning model served by Ollama (gpt-oss); default: the model's")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="output budget per call, reasoning included (default 8192); a reasoning model that "
                         "spends its budget before writing any CSV needs more")
    ap.add_argument("--no-quota-by-class", action="store_true",
                    help="state the exact counts as totals only, not per outcome (the earlier releases' "
                         "prompt); by default each outcome's rows per bin are stated as well, so the "
                         "class-specific marginals are enforced whichever model generates")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="run a model the capability gate has not measured; the output says so")
    ap.add_argument("--request-timeout", type=float, default=None,
                    help="seconds to wait for one model call (default 120); a reasoning model served "
                         "locally can need several minutes per call")
    ap.add_argument("--n-rows", type=int, default=300, help="synthetic rows to produce (default 300)")
    ap.add_argument("--pool-factor", type=int, default=1,
                    help="1 (default): generate exactly n rows; 3: generate 3n and select the n that match "
                         "the release best, the README's recommendation, at three times the spend")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="spend cap; the run stops when it is reached and keeps the rows completed")
    ap.add_argument("--epsilon-total", type=float, default=2.0, help="the Stage A budget (default 2.0)")
    ap.add_argument("--n-min", type=int, default=150, help="cohorts and cells below this size are suppressed")
    ap.add_argument("--max-rows-per-person", type=int, default=1, help="the privacy unit (default 1)")
    ap.add_argument("--epsilon-cert", type=float, default=1.0, help="the Stage C budget (default 1.0)")
    ap.add_argument("--tolerance", default="auto",
                    help="the Stage C tolerance: 'auto' (default) derives it from the release, or a number")
    ap.add_argument("--out", default=None, help="output folder; default results/<schema name>_<backend>")
    ap.add_argument("--no-evaluate", action="store_true", help="skip the evaluation (needs scikit-learn)")
    ap.add_argument("--reference-draws", type=int, default=5,
                    help="real samples (each with a permuted copy) the evaluation scores as references; "
                         "their range is what the results panel reads each synthetic number against "
                         "(default 5; 1 scores one sample, as the earlier releases did)")
    ap.add_argument("--exports", choices=("json", "all"), default="json",
                    help="json (default): one record per stage, complete; all: also the Markdown, "
                         "CSV and figure renderings of each record")
    return ap


def main(argv: list[str] | None = None, prog: str = "cortec run") -> int:
    import time
    from cortec import Generator, Release, bound_with_controls, install_guard, release_statistics, show
    from cortec.report import (banner, emit, files_block, fmt, fmt_seconds, kv_block, note,
                               record_results, warn)

    install_guard()
    a = parser(prog).parse_args(argv)
    t0 = time.time()

    schema = load_schema(a.schema)
    columns = list(schema.numerical) + list(schema.categorical) + [schema.target]
    data = read_table(a.data, columns)
    if a.holdout:
        train, holdout = data, read_table(a.holdout, columns)
        holdout_line = f"{a.holdout}, {len(holdout):,} rows (never shown to Stage A)"
    else:
        holdout = data.sample(frac=0.2, random_state=0)
        train = data.drop(holdout.index).reset_index(drop=True)
        holdout = holdout.reset_index(drop=True)
        holdout_line = (f"{len(holdout):,} of {len(data):,} rows set aside before Stage A (no --holdout given); "
                        f"they are never shown to Stage A")
    out = a.out or os.path.join("results", f"{schema.name}_{a.backend}")
    os.makedirs(out, exist_ok=True)
    outdir = os.path.abspath(out)
    release_path = os.path.join(out, "release.json")
    reuse = os.path.exists(release_path)

    # ---- the banner: what is about to run, before any result ----
    pool = f"{a.n_rows:,} rows" if a.pool_factor <= 1 else \
        f"{a.n_rows:,} rows selected from a pool of {a.n_rows * a.pool_factor:,}"
    emit(banner(schema.name, [
        ("private table", f"{a.data}, {len(train):,} rows, {len(columns)} columns"),
        ("holdout", holdout_line),
        ("backend", f"{a.backend} / {a.model}" + (f" ({a.surface} surface)" if a.surface != "public" else "")
                    + (f", thinking {a.effort}" if a.effort else "")),
        ("privacy budget", f"epsilon {fmt(a.epsilon_total)} for the release (Stage A) + {fmt(a.epsilon_cert)} for "
                           f"the bound (Stage C) = {fmt(a.epsilon_total + a.epsilon_cert)} per row; "
                           f"n_min {a.n_min}"),
        ("synthetic rows", pool + (f"; spend cap ${a.budget_usd:.2f}" if a.budget_usd else "")),
        ("stages", "A release" + (" (reused)" if reuse else "") + ", B generate, C bound"
                   + (", evaluation, results" if not a.no_evaluate else ", results (evaluation skipped)")),
        ("output", outdir)]))
    emit("")

    # ---- Stage A: the only step that reads the private data; the budget is spent here, once ----
    # The schema the release was made from is stored beside it, so a run into the same folder
    # with a different schema (other bins, another hierarchy) is refused instead of silently
    # generating from a release that does not match.
    fingerprint_path = os.path.join(out, "release_schema.txt")
    if reuse:
        stored = open(fingerprint_path, encoding="utf-8").read() if os.path.exists(fingerprint_path) else None
        if stored is not None and stored != repr(schema):
            sys.exit(f"{release_path} was made from a different schema than {a.schema}; a release "
                     f"belongs to its schema. Use another --out for this schema.")
        release = Release.from_json(release_path)
        emit(note(f"Stage A: reusing {release_path}. A release is made once; re-running it would spend "
                  f"the budget again. Use another --out for a new release."))
    else:
        release = release_statistics(schema, train, epsilon_total=a.epsilon_total, n_min=a.n_min,
                                     max_rows_per_person=a.max_rows_per_person, n_records=a.n_rows)
        release.to_json(release_path)
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            f.write(repr(schema))
    rec_a = show(release, n_private_rows=len(train))
    files_a = save_record(rec_a, out, "stage_a_release", a.exports)
    emit("")

    # ---- Stage B: the model sees only the release; generation costs no privacy budget ----
    t_b = time.time()
    gen = Generator(schema, backend=a.backend, model=a.model, surface=a.surface,
                    surface_model=a.surface_model, reasoning="on", budget_usd=a.budget_usd,
                    allow_unvalidated=a.allow_unvalidated, quota_by_class=not a.no_quota_by_class,
                    **({"request_timeout": float(a.request_timeout)} if a.request_timeout else {}))
    if a.effort:
        gen.ollama_think = a.effort
    if a.max_output_tokens:
        gen.max_output_tokens = int(a.max_output_tokens)
        # the same context rule as the research harness (output budget + 4,096 for the prompt), so
        # the two never ask one Ollama server for different context sizes, which makes it reload
        # the model between their requests
        gen.num_ctx = max(gen.num_ctx, gen.max_output_tokens + 4096)
    surface = dict(kv.split("=", 1) for kv in gen.describe_surface().split())
    emit(kv_block([("Stage B", f"generating {pool}"),
                   ("through", f"{surface.get('backend')} / {surface.get('model')}, {surface.get('surface')} surface"
                               + (f", request model {surface['request_model']}"
                                  if surface.get("request_model") not in (None, surface.get("model")) else "")),
                   ("exact counts", "per column and per outcome" if gen.quota_by_class else
                                    "per column only (totals over the outcomes)")]))
    if a.pool_factor <= 1:
        synthetic = gen.generate(release, n_rows=a.n_rows)
    else:
        synthetic = gen.generate_selected(release, n_rows=a.n_rows, pool_factor=a.pool_factor)
    synthetic = synthetic.drop(columns=["_cohort", "_cell"], errors="ignore")
    positive_rate = float((synthetic[schema.target].astype(str) == str(schema.positive)).mean())
    rec_b = show(gen, n_rows=len(synthetic), positive_rate=positive_rate)
    rec_b.values.append(("positive rate in the private data",
                         float((train[schema.target].astype(str) == str(schema.positive)).mean())))
    rec_b.values.append(("generation time", fmt_seconds(time.time() - t_b)))
    files_b = save_record(rec_b, out, "stage_b_generate", a.exports)
    synthetic.to_csv(os.path.join(out, "synthetic.csv"), index=False)
    emit("")

    # ---- Stage C: a DP bound on the private-vs-synthetic conditional gap, with its own controls ----
    report = bound_with_controls(schema, release, train, synthetic, holdout,
                                 epsilon_cert=a.epsilon_cert, alpha=0.05,
                                 tolerance=a.tolerance if str(a.tolerance).lower() == "auto" else float(a.tolerance))
    rec_c = show(report, schema=schema.name)
    files_c = save_record(rec_c, out, "stage_c_bound", a.exports)
    report.to_json(os.path.join(out, "bound.json"))
    emit("")

    # ---- Evaluation: fidelity and utility beside real samples and their permuted floors ----
    files_e: dict = {}
    rec_e = None
    if not a.no_evaluate:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            emit(warn("evaluation skipped: scikit-learn is not installed (pip install 'cortec[dev]')"))
        else:
            from cortec import evaluate
            rec_e = evaluate(schema, {"synthetic": synthetic}, train=train, holdout=holdout,
                             reference_draws=max(1, a.reference_draws))
            rec_e.show()
            files_e = save_record(rec_e, out, "evaluation", a.exports)
            emit("")

    # ---- Results: marginal fidelity, downstream utility, the bound, the privacy and the cost, on one screen ----
    rec_r = record_results(evaluation=rec_e, bound=rec_c, generation=rec_b, release=rec_a,
                           schema=schema.name, seconds=time.time() - t0)
    rec_r.show()
    files_r = save_record(rec_r, out, "results", a.exports)
    rec_r.write_markdown(os.path.join(out, "RESULTS.md"))      # the panel as a document, always
    emit("")

    # ---- Files: the ones a reader opens first, then the records, one line per stage ----
    def kinds(files):
        exts = sorted({os.path.splitext(p)[1].lstrip(".") for p in files.values()
                       if not str(p).startswith("not written")})
        return ", ".join(exts)
    entries = [("folder", outdir),
               ("RESULTS.md", "the results panel above, as a document"),
               ("synthetic.csv", f"the synthetic table, {len(synthetic):,} rows"),
               ("release.json", "the DP release (Stage A), reused by later runs into this folder"),
               ("bound.json", "the Stage C bound with its two controls")]
    for stem, what, files in (("stage_a_release", "the Stage A record (the release)", files_a),
                              ("stage_b_generate", "the Stage B record (generation)", files_b),
                              ("stage_c_bound", "the Stage C record (the bound)", files_c),
                              ("evaluation", "the evaluation record", files_e),
                              ("results", "the results record", files_r)):
        if files:
            entries.append((stem + ".*", f"{what}: {kinds(files)}"))
    emit(files_block("everything this run wrote", entries))
    if a.exports == "json":
        emit(note("each JSON record renders to Markdown, CSV and a figure: --exports all, or "
                  "RunRecord.from_json(path).save(folder)"))
    emit(note("the terms used above (tolerance, ceiling, floor, TSTR, share ...): cortec terms"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
