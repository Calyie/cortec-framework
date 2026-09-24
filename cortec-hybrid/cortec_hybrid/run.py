"""`cortec-hybrid run`: the correction on your own tables, from any directory once the package is
installed. Releases a DP conditional target table from the private data and relabels ONLY the
target column of synthetic data you already generated (with AIM, MST, PrivBayes, or cortec) so
each cell matches the table. Prints the correction in the standard layout, writes corrected.csv
and the record, and, when a holdout is given, scores the table before and after beside real
references.

  cortec-hybrid run --data private.csv --synthetic synthetic.csv \\
      --schema my_schema.py --columns age,sex --epsilon 0.5
  cortec-hybrid run --data private.csv --synthetic synthetic.csv --holdout holdout.csv \\
      --schema my_schema.py --columns age,sex,education --n-min 100

`my_schema.py` defines `SCHEMA = Schema(...)` from cortec (its README, step 4). This tool does not
generate data. `worth_it` in the output is a calibration criterion only; read the evaluation table
beside it before adopting the correction. A refusal prints as a boxed block and exits with 2.
"""
from __future__ import annotations

import argparse
import os
import sys


def parser(prog: str = "cortec-hybrid run") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the private table (CSV), one row per person")
    ap.add_argument("--synthetic", required=True, help="the synthetic table to correct (CSV), same columns")
    ap.add_argument("--schema", required=True, help="a Python file that defines SCHEMA = Schema(...)")
    ap.add_argument("--columns", required=True, help="the cell columns, comma-separated (e.g. age,sex)")
    ap.add_argument("--epsilon", type=float, default=0.5, help="the budget for the table (default 0.5)")
    ap.add_argument("--n-min", type=int, default=150, help="cells below this size are not released; floor 50")
    ap.add_argument("--max-rows-per-person", type=int, default=1, help="the privacy unit (default 1)")
    ap.add_argument("--holdout", help="real rows not in --data (CSV); enables the before/after evaluation")
    ap.add_argument("--out", default=None, help="output folder; default: the synthetic table's folder")
    ap.add_argument("--exports", choices=("json", "all"), default="json",
                    help="json (default): one record per step, complete; all: also the Markdown, "
                         "CSV and figure renderings of each record")
    return ap


def main(argv: list[str] | None = None, prog: str = "cortec-hybrid run") -> int:
    from cortec import install_guard, show
    from cortec.report import emit, header, kv_block, note, warn
    from cortec.run import load_schema, read_table, save_record
    from cortec_hybrid import check_feasible, correct

    install_guard()
    a = parser(prog).parse_args(argv)

    schema = load_schema(a.schema)
    columns = list(schema.numerical) + list(schema.categorical) + [schema.target]
    private = read_table(a.data, columns)
    synthetic = read_table(a.synthetic, columns)
    cells = tuple(c.strip() for c in a.columns.split(",") if c.strip())
    out = a.out or os.path.dirname(os.path.abspath(a.synthetic))
    os.makedirs(out, exist_ok=True)

    feasibility = check_feasible(schema, n_rows=len(private), raise_on_fail=False)
    emit(kv_block([("feasibility: " + k, v) for k, v in feasibility.items()
                   if not isinstance(v, (list, dict))]))
    triple = correct(schema, private, synthetic, columns=cells, epsilon=a.epsilon, n_min=a.n_min)
    corrected = triple[0]
    before = float((synthetic[schema.target].astype(str) == str(schema.positive)).mean())
    after = float((corrected[schema.target].astype(str) == str(schema.positive)).mean())
    rec = show(triple, schema=schema.name, n_rows=len(synthetic),
               positive_rate_before=before, positive_rate_after=after)
    corrected.to_csv(os.path.join(out, "corrected.csv"), index=False)
    files = save_record(rec, out, "correction", a.exports)

    files_e: dict = {}
    if a.holdout:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            emit(warn("evaluation skipped: scikit-learn is not installed"))
        else:
            from cortec import evaluate
            holdout = read_table(a.holdout, columns)
            rec_e = evaluate(schema, {"before": synthetic, "after": corrected},
                             train=private, holdout=holdout)
            rec_e.show()
            files_e = save_record(rec_e, out, "evaluation", a.exports)
    else:
        emit(note("no --holdout given: the before/after evaluation was skipped; worth_it alone does "
                  "not say whether downstream utility improved"))

    emit("")
    outdir = os.path.abspath(out)
    emit(header("Files written", outdir, tool="cortec-hybrid"))
    emit(note(f"{'corrected':12s} {'features unchanged, target relabelled':30s} corrected.csv"))
    for stage, stage_files in (("correction", files), ("evaluation", files_e)):
        for name, path in stage_files.items():
            if not str(path).startswith("not written"):
                label = name.replace("table_csv:", "table: ")
                emit(note(f"{stage:12s} {label:30s} {os.path.basename(path)}"))
    emit(note(f"all of the above are inside  {outdir}"))
    if a.exports == "json":
        emit(note("each JSON record renders to Markdown, CSV and a figure: --exports all, or "
                  "RunRecord.from_json(path).save(folder)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
