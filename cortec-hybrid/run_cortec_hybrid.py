"""Run cortec-hybrid on your own tables: release a DP conditional target table from the private
data and relabel ONLY the target column of synthetic data you already generated (with AIM, MST,
PrivBayes, or cortec) so each cell matches the table. Prints the correction in the standard
layout, writes corrected.csv and the record, and, when a holdout is given, scores the synthetic
table before and after beside real references.

  python3 run_cortec_hybrid.py --data private.csv --synthetic synthetic.csv \\
      --schema my_schema.py --columns age,sex --epsilon 0.5
  python3 run_cortec_hybrid.py --data private.csv --synthetic synthetic.csv --holdout holdout.csv \\
      --schema my_schema.py --columns age,sex,education --n-min 100

`my_schema.py` defines `SCHEMA = Schema(...)` from cortec (its README, step 4). This tool does not
generate data. `worth_it` in the output is a calibration criterion only; read the evaluation table
beside it before adopting the correction. A refusal prints as a boxed block and exits with 2.
"""
import argparse
import importlib.util
import os
import sys

import pandas as pd

from cortec import install_guard, show
from cortec.report import emit, header, kv_block, note, warn
from cortec_hybrid import check_feasible, correct

install_guard()

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True, help="the private table (CSV), one row per person")
ap.add_argument("--synthetic", required=True, help="the synthetic table to correct (CSV), same columns")
ap.add_argument("--schema", required=True, help="a Python file that defines SCHEMA = Schema(...)")
ap.add_argument("--columns", required=True, help="the cell columns, comma-separated (e.g. age,sex)")
ap.add_argument("--epsilon", type=float, default=0.5, help="the budget for the table (default 0.5)")
ap.add_argument("--n-min", type=int, default=150, help="cells below this size are not released; floor 50")
ap.add_argument("--max-rows-per-person", type=int, default=1, help="the privacy unit (default 1)")
ap.add_argument("--holdout", help="real rows not in --data (CSV); enables the before/after evaluation")
ap.add_argument("--out", default=None, help="output folder; default: the synthetic table's folder")
a = ap.parse_args()


def load_schema(path):
    spec = importlib.util.spec_from_file_location("user_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "SCHEMA"):
        sys.exit(f"{path} must define SCHEMA = Schema(...); see cortec's README, step 4")
    return module.SCHEMA


def read_table(path, columns):
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        sys.exit(f"{path} lacks the schema's column(s) {missing}; it has {list(df.columns)}")
    return df[columns]


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
files = rec.save(out, "correction")

files_e = {}
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
        files_e = rec_e.save(out, "evaluation")
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
