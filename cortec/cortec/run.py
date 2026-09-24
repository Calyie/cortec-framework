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
    ap.add_argument("--tolerance", type=float, default=0.15, help="the Stage C tolerance (default 0.15)")
    ap.add_argument("--out", default=None, help="output folder; default results/<schema name>_<backend>")
    ap.add_argument("--no-evaluate", action="store_true", help="skip the evaluation (needs scikit-learn)")
    return ap


def main(argv: list[str] | None = None, prog: str = "cortec run") -> int:
    from cortec import Generator, Release, bound_with_controls, install_guard, release_statistics, show
    from cortec.report import emit, header, note, warn

    install_guard()
    a = parser(prog).parse_args(argv)

    schema = load_schema(a.schema)
    columns = list(schema.numerical) + list(schema.categorical) + [schema.target]
    data = read_table(a.data, columns)
    if a.holdout:
        train, holdout = data, read_table(a.holdout, columns)
    else:
        holdout = data.sample(frac=0.2, random_state=0)
        train = data.drop(holdout.index).reset_index(drop=True)
        holdout = holdout.reset_index(drop=True)
        emit(note(f"no --holdout given: {len(holdout)} of {len(data)} rows set aside before Stage A"))
    out = a.out or os.path.join("results", f"{schema.name}_{a.backend}")
    os.makedirs(out, exist_ok=True)
    release_path = os.path.join(out, "release.json")

    # ---- Stage A: the only step that reads the private data; the budget is spent here, once ----
    # The schema the release was made from is stored beside it, so a run into the same folder
    # with a different schema (other bins, another hierarchy) is refused instead of silently
    # generating from a release that does not match.
    fingerprint_path = os.path.join(out, "release_schema.txt")
    if os.path.exists(release_path):
        stored = open(fingerprint_path, encoding="utf-8").read() if os.path.exists(fingerprint_path) else None
        if stored is not None and stored != repr(schema):
            sys.exit(f"{release_path} was made from a different schema than {a.schema}; a release "
                     f"belongs to its schema. Use another --out for this schema.")
        release = Release.from_json(release_path)
        emit(note(f"Stage A: reusing {release_path} (a release is made once; re-running would spend "
                  f"the budget again; use another --out for a new release)"))
    else:
        release = release_statistics(schema, train, epsilon_total=a.epsilon_total, n_min=a.n_min,
                                     max_rows_per_person=a.max_rows_per_person, n_records=a.n_rows)
        release.to_json(release_path)
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            f.write(repr(schema))
    rec_a = show(release, n_private_rows=len(train))
    rec_a.save(out, "stage_a_release")

    # ---- Stage B: the model sees only the release; generation costs no privacy budget ----
    gen = Generator(schema, backend=a.backend, model=a.model, surface=a.surface,
                    surface_model=a.surface_model, reasoning="on", budget_usd=a.budget_usd)
    emit(note(gen.describe_surface()))
    if a.pool_factor <= 1:
        synthetic = gen.generate(release, n_rows=a.n_rows)
    else:
        synthetic = gen.generate_selected(release, n_rows=a.n_rows, pool_factor=a.pool_factor)
    synthetic = synthetic.drop(columns=["_cohort", "_cell"], errors="ignore")
    positive_rate = float((synthetic[schema.target].astype(str) == str(schema.positive)).mean())
    rec_b = show(gen, n_rows=len(synthetic), positive_rate=positive_rate)
    rec_b.values.append(("positive rate in the private data",
                         float((train[schema.target].astype(str) == str(schema.positive)).mean())))
    rec_b.save(out, "stage_b_generate")
    synthetic.to_csv(os.path.join(out, "synthetic.csv"), index=False)

    # ---- Stage C: a DP bound on the private-vs-synthetic conditional gap, with its own controls ----
    report = bound_with_controls(schema, release, train, synthetic, holdout,
                                 epsilon_cert=a.epsilon_cert, alpha=0.05, tolerance=a.tolerance)
    rec_c = show(report, schema=schema.name)
    rec_c.save(out, "stage_c_bound")
    report.to_json(os.path.join(out, "bound.json"))

    # ---- Evaluation: fidelity and utility beside a real sample and a permuted floor ----
    files_e: dict = {}
    if not a.no_evaluate:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            emit(warn("evaluation skipped: scikit-learn is not installed (pip install 'cortec[dev]')"))
        else:
            from cortec import evaluate
            rec_e = evaluate(schema, {"synthetic": synthetic}, train=train, holdout=holdout)
            rec_e.show()
            files_e = rec_e.save(out, "evaluation")

    emit("")
    outdir = os.path.abspath(out)
    emit(header("Files written", outdir))
    emit(note(f"{'synthetic':12s} {'the synthetic table':30s} synthetic.csv"))
    emit(note(f"{'release':12s} {'reused by later runs here':30s} release.json"))
    emit(note(f"{'Stage C':12s} {'bound report':30s} bound.json"))
    for stage, files in (("Stage A", rec_a.files), ("Stage B", rec_b.files),
                         ("Stage C", rec_c.files), ("evaluation", files_e)):
        for name, path in files.items():
            if not str(path).startswith("not written"):
                label = name.replace("table_csv:", "table: ")
                emit(note(f"{stage:12s} {label:30s} {os.path.basename(path)}"))
    emit(note(f"all of the above are inside  {outdir}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
