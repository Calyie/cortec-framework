"""cortec.evaluate: score a synthetic table the way the paper does.

Never against a bare threshold. Every synthetic table is scored beside two references built from
real data of the same size: a real sample, the ceiling a synthetic method can reach at that size,
and the same sample with its target column permuted, the floor of data that carries no usable
information. Fidelity and utility are reported together in one table, because a method can score
well on one and poorly on the other.

Measures, as in the paper's head-to-head tables:

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


MEASURES = ("1-way TV", "TSTR-LR", "TSTR-RF", "TSTR-GBM")


def evaluate(schema: Schema, tables: dict[str, pd.DataFrame], *, train: pd.DataFrame,
             holdout: pd.DataFrame, n: int | None = None, seed: int = 1000,
             reference_draws: int = 5) -> RunRecord:
    """Score every table in `tables` (name -> DataFrame) beside a real sample of `train` and its
    permuted-target floor, both of size `n` (default: the first table's size). `holdout` is real
    data that was not used to build the release. Returns the standard record; its first table is
    the paper's fidelity-and-utility layout with the two reference rows marked.

    `reference_draws` real samples are drawn (seeds `seed`, `seed + 1`, ...), each with its
    permuted copy, and the two reference rows carry their means. With more than one draw the
    record gains a second table, `reference spread`: for each measure the mean, the lowest and
    the highest value over the real samples and over their permuted copies. That range is what
    real data of this size spans on this holdout, and the results panel reads every synthetic
    number against it instead of against a threshold. `reference_draws=1` is the single sample
    the earlier releases scored."""
    if not tables:
        raise ValueError("evaluate() needs at least one table to score")
    if reference_draws < 1:
        raise ValueError("reference_draws must be at least 1")
    cols = schema.columns          # features and the target
    names = list(tables)
    n = int(n or len(tables[names[0]]))
    from .report import emit_progress_done, ticker

    def score(label, df):
        # three students per table take seconds each; the terminal shows which is being scored
        with ticker(lambda t, label=label: f"Evaluation · training the students on {label} · {t:.0f}s"):
            u = tstr(schema, df, holdout, seed=0)
        miss = absent_categories(schema, df)
        return [len(df), marginal_tv(schema, df, holdout), u["LR"], u["RF"], u["GBM"], len(miss)], miss

    rows, absent = [], {}
    for nm in names:
        r, miss = score(nm, tables[nm][cols])
        absent[nm] = miss
        rows.append([nm] + r)
    ref_real, ref_perm = [], []
    n_ref = min(n, len(train))
    for i in range(reference_draws):
        s = seed + i
        real = train.sample(n_ref, random_state=s).reset_index(drop=True)
        perm = real.copy()
        perm[schema.target] = np.random.default_rng(s).permutation(perm[schema.target].values)
        tag = f" {i + 1}/{reference_draws}" if reference_draws > 1 else ""
        ref_real.append(score(f"real sample{tag}", real[cols])[0])
        ref_perm.append(score(f"permuted target{tag}", perm[cols])[0])
    emit_progress_done()

    def mean_row(label, rs):
        a = np.array([r[:5] for r in rs], dtype=float)
        m = a.mean(axis=0)
        return [label, int(round(m[0])), float(m[1]), float(m[2]), float(m[3]), float(m[4]),
                int(round(float(np.mean([r[5] for r in rs]))))]

    rows.append(mean_row(f"real sample, n={n_ref} (ceiling)", ref_real))
    rows.append(mean_row(f"permuted target, n={n_ref} (floor)", ref_perm))
    means = f" The two reference rows are means over {reference_draws} draws." if reference_draws > 1 else ""
    table = ResultTable("fidelity and utility",
                        ["table", "rows", "1-way TV", "TSTR-LR", "TSTR-RF", "TSTR-GBM", "absent categories"],
                        rows, reference_rows=[len(names), len(names) + 1],
                        subtitle="Fidelity (1-way TV, lower is better) and utility (train on the table, test on "
                                 "the holdout: AUC, higher is better) in one table, as in the paper." + means,
                        note="read every row against the two reference rows, not against a threshold: the "
                             "real sample is the ceiling at this size, the permuted target is the floor of "
                             "no information")
    tables_out = [table]
    if reference_draws > 1:
        spread = []
        for j, meas in zip((1, 2, 3, 4), MEASURES):
            a = np.array([r[j] for r in ref_real], dtype=float)
            b = np.array([r[j] for r in ref_perm], dtype=float)
            spread.append([meas, float(np.nanmean(a)), float(np.nanmin(a)), float(np.nanmax(a)),
                           float(np.nanmean(b)), float(np.nanmin(b)), float(np.nanmax(b))])
        tables_out.append(ResultTable(
            "reference spread",
            ["measure", "real sample mean", "real min", "real max", "permuted mean", "permuted min", "permuted max"],
            spread,
            subtitle=f"What real data of this size spans on this holdout: {reference_draws} real samples of "
                     f"{n_ref} rows and their permuted copies.",
            note="a synthetic number inside the real-sample range is indistinguishable from a real sample of "
                 "the same size on that measure, at this number of draws"))
    values = [("tables scored", len(names)), ("reference size n", n_ref),
              ("reference draws", reference_draws), ("holdout rows", len(holdout)),
              ("holdout positive rate", float((holdout[schema.target].astype(str).str.strip() == schema.positive).mean()))]
    warnings_ = [f"{label}: absent categories {', '.join(miss)}" for label, miss in absent.items()
                 if miss and label in names]
    rec = RunRecord(tool="cortec", stage="Evaluation", title="Fidelity and utility beside real references",
                    schema=schema.name, values=values, tables=tables_out, warnings=warnings_,
                    notes=["A single run at one size is a signal, not a claim; replicate before "
                           "treating a difference as stable."])
    return rec
