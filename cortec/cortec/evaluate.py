"""cortec.evaluate: score a synthetic table the way the paper does.

Never against a bare threshold. Every synthetic table is scored beside two references built from
real data of the same size: a real sample, the ceiling a synthetic method can reach at that size,
and the same sample with its target column permuted, the floor of data that carries no usable
information. Fidelity and utility are reported together in one table, because a method can score
well on one and poorly on the other.

Measures, as in the paper's Table 6:

    1-way TV   mean over columns of the total-variation distance between the table's binned
               marginals and the holdout's, on the schema's public bins (lower is better)
    TSTR-LR, TSTR-RF, TSTR-GBM
               train on the table, test on the holdout: AUC of logistic regression, a random
               forest and gradient boosting (higher is better)
    absent categories
               declared categorical values that never appear in the table: a representativeness
               failure that no aggregate measure reports

scikit-learn is needed for the three students; it is the `dev` extra (`pip install './cortec[dev]'`).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .report import ResultTable, RunRecord
from .schema import Schema

STUDENTS = ("LR", "RF", "GBM")


def _students():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as e:      # pragma: no cover
        raise ImportError("cortec.evaluate needs scikit-learn: pip install './cortec[dev]'") from e
    return (ColumnTransformer, HistGradientBoostingClassifier, RandomForestClassifier,
            LogisticRegression, roc_auc_score, make_pipeline, OneHotEncoder, StandardScaler)


def _binned(schema: Schema, df: pd.DataFrame, col: str) -> pd.Series:
    if col in schema.numerical:
        return pd.cut(pd.to_numeric(df[col], errors="coerce"), schema.bin_edges(col),
                      include_lowest=True).astype(str)
    return df[col].astype(str).str.strip()


def marginal_tv(schema: Schema, df: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Mean 1-way total-variation distance between `df` and `reference` over every column."""
    tvs = []
    for c in schema.columns:      # features and the target
        p = _binned(schema, df, c).value_counts(normalize=True)
        q = _binned(schema, reference, c).value_counts(normalize=True)
        idx = p.index.union(q.index)
        tvs.append(0.5 * float((p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()))
    return float(np.mean(tvs))


def absent_categories(schema: Schema, df: pd.DataFrame) -> list[str]:
    """Declared categorical values that never appear in `df`, as `column=value`."""
    out = []
    for col, allowed in schema.categorical.items():
        seen = set(df[col].astype(str).str.strip())
        out += [f"{col}={v}" for v in allowed if str(v) not in seen]
    return out


def tstr(schema: Schema, df: pd.DataFrame, holdout: pd.DataFrame, *, seed: int = 0) -> dict:
    """Train on `df`, test on `holdout`: AUC per student. NaN when `df` has one class only."""
    (ColumnTransformer, HGB, RF, LR, roc_auc_score, make_pipeline, OneHot, Scaler) = _students()
    num, cat, t = schema.numerical_cols, schema.categorical_cols, schema.target
    y = (df[t].astype(str).str.strip() == schema.positive).astype(int)
    yt = (holdout[t].astype(str).str.strip() == schema.positive).astype(int)
    if y.nunique() < 2:
        return {s: float("nan") for s in STUDENTS}
    x = df[num + cat].copy()
    xt = holdout[num + cat].copy()
    for c in num:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        xt[c] = pd.to_numeric(xt[c], errors="coerce")
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, clf in (("LR", LR(max_iter=2000)),
                          ("RF", RF(n_estimators=300, random_state=seed, n_jobs=-1)),
                          ("GBM", HGB(random_state=seed))):
            pre = ColumnTransformer([("n", Scaler(), num), ("c", OneHot(handle_unknown="ignore"), cat)])
            m = make_pipeline(pre, clf).fit(x, y)
            out[name] = float(roc_auc_score(yt, m.predict_proba(xt)[:, 1]))
    return out


def evaluate(schema: Schema, tables: dict[str, pd.DataFrame], *, train: pd.DataFrame,
             holdout: pd.DataFrame, n: int | None = None, seed: int = 1000) -> RunRecord:
    """Score every table in `tables` (name -> DataFrame) beside a real sample of `train` and its
    permuted-target floor, both of size `n` (default: the first table's size). `holdout` is real
    data that was not used to build the release. Returns the standard record; its one table is
    the paper's fidelity-and-utility layout with the two reference rows marked."""
    if not tables:
        raise ValueError("evaluate() needs at least one table to score")
    cols = schema.columns          # features and the target
    names = list(tables)
    n = int(n or len(tables[names[0]]))
    rng = np.random.default_rng(seed)
    real = train.sample(min(n, len(train)), random_state=seed).reset_index(drop=True)
    perm = real.copy()
    perm[schema.target] = rng.permutation(perm[schema.target].values)
    rows, absent = [], {}
    scored = [(nm, tables[nm][cols]) for nm in names] + \
             [(f"real sample, n={len(real)} (ceiling)", real[cols]),
              (f"permuted target, n={len(perm)} (floor)", perm[cols])]
    for label, df in scored:
        u = tstr(schema, df, holdout, seed=0)
        miss = absent_categories(schema, df)
        absent[label] = miss
        rows.append([label, len(df), marginal_tv(schema, df, holdout), u["LR"], u["RF"], u["GBM"], len(miss)])
    table = ResultTable("fidelity and utility",
                        ["table", "rows", "1-way TV", "TSTR-LR", "TSTR-RF", "TSTR-GBM", "absent categories"],
                        rows, reference_rows=[len(names), len(names) + 1],
                        note="lower TV is better; higher AUC is better; read every row against the "
                             "two reference rows, not against a threshold")
    values = [("tables scored", len(names)), ("reference size n", n),
              ("holdout rows", len(holdout)),
              ("holdout positive rate", float((holdout[schema.target].astype(str).str.strip() == schema.positive).mean()))]
    warnings_ = [f"{label}: absent categories {', '.join(miss)}" for label, miss in absent.items()
                 if miss and label in names]
    rec = RunRecord(tool="cortec", stage="Evaluation", title="Fidelity and utility beside real references",
                    schema=schema.name, values=values, tables=[table], warnings=warnings_,
                    notes=["A single run at one size is a signal, not a claim; replicate before "
                           "treating a difference as stable."])
    return rec
