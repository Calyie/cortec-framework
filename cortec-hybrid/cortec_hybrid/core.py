"""
cortec-hybrid — conditional-signal correction for a marginal DP synthesiser, with no LLM.

**What this tool claims, precisely.** Marginal-based DP synthesisers (AIM, MST, PrivBayes) are
excellent at the statistic they optimise — low-order marginals — and are evaluated on it in their
own papers. Neither AIM's nor MST's paper reports a downstream predictive-utility experiment, and
in our measurements that omission is consequential: at matched budget and sample size on UCI Adult,
models trained on their output reached 0.675 to 0.728 AUC where a real sample of the same size
reached 0.834 to 0.872 (technical report, section 7.2).

This tool spends a *small, separately accounted* slice of the privacy budget on one thing those
mechanisms do not target: a conditional target table, P(y | cell) over a disjoint partition. It
then relabels the synthesiser's output to match that table, using rank-preserving assignment. The
relabelling is pure post-processing of two already-private artifacts, so it adds no privacy cost
beyond the table itself.

**What this tool does NOT claim.** It is not "CoRTeC without the LLM", and it is not a general
improvement over its inputs. In our own runs a coarse 12-cell table, the obvious implementation,
fixed calibration and added little utility (+0.025 AUC on AIM, +0.037 on MST). The gain requires a
*rich* table (technical report, section 8.2), and it is worth having in one specific situation: your marginal synthesiser's output has
weak conditional target structure and you need a calibrated rate. Measure before and after with
`estimate_gain()`; if the gain is small, use the marginal synthesiser alone and keep the budget.

**A correction to our own earlier belief, found while testing this package.** We previously held
that rank-preserving assignment was necessary for the correction to pay for itself. That is true
only when the input synthetic data already carries within-cell structure. When its target is close
to noise — precisely the case this tool targets — a ranking model fitted on that target learns the
noise, and ordering by it injects spurious structure that a downstream model mistakes for signal.
That guard did not work, and the way it failed is the useful part. `_rank_scores` gated on whether
the synthetic target is PREDICTABLE from the synthetic features (cross-validated AUC against
`RANK_INFORMATIVENESS_FLOOR`). But a synthesiser generates its target AS a function of those
features, so that score is near-perfect almost always -- 0.981 on one dataset whose real downstream
AUC is 0.631. **It measures self-consistency, not validity, and it is highest exactly when the
synthesiser has confidently learned a wrong relationship, so it selected FOR the failure mode it
was built to prevent.** Replicated over three datasets and three seeds each, plain i.i.d.
assignment led rank preservation in 9 of 9 runs. `preserve_ranking=False` is therefore the default
and rate matching alone is the mechanism; the gate survives only for callers who opt in
explicitly.

**Why it exists separately from `cortec`.** An organisation that cannot run an LLM at all, or does
not want to, can still correct the one axis marginal methods leave weakest. The dependency is
scikit-learn, not a model server.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Reuse the audited accounting rather than reimplementing composition rules.
try:
    from cortec.accounting import (VACUOUS_EPSILON_PER_PERSON, PrivacyLedger,
                                   laplace_noise)
    from cortec.schema import Schema
except ImportError as e:                                  # pragma: no cover
    raise ImportError(
        "cortec-hybrid reuses cortec's audited privacy accounting. Install the `cortec` "
        "package (pip install cortec) — it is a pure-python dependency and pulls in no LLM client."
    ) from e


# Cross-validated AUC the input synthetic data must reach before its own ordering is treated as
# signal worth preserving. Below this, ranking injects noise instead of preserving structure.
RANK_INFORMATIVENESS_FLOOR = 0.55


class DomainTooLargeError(RuntimeError):
    """The schema's domain is large enough that the marginal synthesiser is unlikely to finish."""


@dataclass
class ConditionalTable:
    """A DP table of P(target = positive | cell) over a disjoint partition."""
    columns: tuple[str, ...]
    cells: dict[str, float]
    support: dict[str, int]
    epsilon: float
    audit: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cells)


# ── up-front feasibility check ──────────────────────────────────────────────────────

def estimate_domain_size(schema: Schema) -> int:
    """Product of per-column domain sizes after public binning — what drives inference cost."""
    total = 1
    for c in schema.numerical:
        total *= max(len(schema.bin_edges(c)) - 1, 1)
    for c, vals in schema.categorical.items():
        total *= max(len(vals), 1)
    total *= 2   # binary target
    return total


# Every AIM fit we ran, at eps=2.0 under one harness. This table is the whole basis for the gate
# below, so it is kept here rather than in prose: a user who disagrees with the rule can read the
# observations and decide for themselves.
#
#   dataset              cols   rows    domain     result
#   titanic                 9   1,047   small      FIT      56 s
#   adult                  15  26,048   4.4e13     FIT   2,626 s      <- LARGEST domain, and it fits
#   diabetes (10-col)      10  26,048   6.8e6      FIT   1,579 s
#   diabetes (10-col)      10  81,410   6.8e6      TIMEOUT  > 3,600 s <- only n changed
#   diabetes (full)        19  26,048   3.5e13     TIMEOUT  > 3,600 s <- only width changed
#   credit (full)          15  24,000   1.5e12     TIMEOUT  > 2,400 s
#   credit  -BILL_AMT1-3   12  24,000   2.9e9      FIT   1,598 s
#   credit  -PAY_AMT1-3    12  24,000   4.3e9      FIT   1,173 s      <- LARGER domain than above,
#   credit  -3 demographics 12 24,000   —          TIMEOUT  > 2,400 s    and 425 s FASTER
#   credit  -all 6 amounts  9  24,000   8.4e6      FIT     414 s
AIM_OBSERVATIONS = (
    ("titanic", 9, 1047, "FIT 56 s"),
    ("adult", 15, 26048, "FIT 2,626 s"),
    ("diabetes (first 10 cols)", 10, 26048, "FIT 1,579 s"),
    ("diabetes (first 10 cols)", 10, 81410, "TIMEOUT > 3,600 s"),
    ("diabetes (full)", 19, 26048, "TIMEOUT > 3,600 s"),
    ("credit (full)", 15, 24000, "TIMEOUT > 2,400 s"),
    ("credit less BILL_AMT1-3", 12, 24000, "FIT 1,598 s"),
    ("credit less PAY_AMT1-3", 12, 24000, "FIT 1,173 s"),
    ("credit less 3 demographics", 12, 24000, "TIMEOUT > 2,400 s"),
    ("credit less all 6 amount cols", 9, 24000, "FIT 414 s"),
)

WIDE_NUMERIC_MIN_BINS = 8      # a heavy-tailed amount column, as declared in the schema
MAX_COLUMNS = 18               # 19 timed out at a row count where 15 fitted
MAX_ROWS_AT_WIDTH = 50_000     # 10 cols fitted at 26k and timed out at 81k
WIDTH_FOR_ROW_LIMIT = 10
MAX_WIDE_NUMERIC_AT_WIDTH = 5  # credit: 15 cols with 6+ amount columns timed out; 12 with 3 fitted
WIDTH_FOR_WIDE_LIMIT = 13


def _wide_numeric_columns(schema: Schema) -> list[str]:
    out = []
    for c in getattr(schema, "numerical_cols", []) or []:
        try:
            edges = schema.bin_edges(c)
        except Exception:
            continue
        if edges is not None and len(edges) - 1 >= WIDE_NUMERIC_MIN_BINS:
            out.append(c)
    return out


def check_feasible(schema: Schema, *, n_rows: int | None = None,
                   raise_on_fail: bool = True, max_domain: int | None = None) -> dict:
    """Warn or refuse BEFORE the user spends hours discovering the synthesiser will not converge.

    **This gate used to be domain size, and our own ablation refuted that.** Adult has the largest
    domain of every schema we measured -- 4.4e13, six million times larger than a diabetes prefix
    that times out -- and Adult is the one that fits. Worse, within one dataset the arm with the
    LARGER domain fit 425 s FASTER than the arm with the smaller one. Domain volume does not
    predict convergence and gating on it refuses the wrong schemas.

    What did predict it, each demonstrated by a controlled flip against AIM_OBSERVATIONS:

      * **attribute count** -- the full 19-column healthcare schema times out at the same row count
        where the 15-column census schema fits;
      * **row count** -- the same 10-column schema at the same domain moves from a 26-minute fit to
        a timeout purely by going from 26,048 rows to 81,410;
      * **high-cardinality correlated numeric columns** -- finance has Adult's width, FEWER rows and
        a 30x smaller domain and still does not converge. Dropping any three columns does not
        rescue it; dropping three *amount* columns does. A width-matched control (drop three
        demographics instead) still times out, which is the only way to separate the two causes.

    Scope, stated because it bounds the rule: ten fits, one dataset family per cause, one AIM
    implementation, one time budget. This is a heuristic from observations, not a theory of AIM.
    `max_domain` is accepted and ignored, for callers written against the old signature.
    """
    n_cols = len(schema.columns)
    wide = _wide_numeric_columns(schema)
    size = estimate_domain_size(schema)

    reasons: list[str] = []
    if n_cols > MAX_COLUMNS:
        reasons.append(
            f"{n_cols} attributes: a {MAX_COLUMNS + 1}-attribute schema timed out at a row count "
            f"where a 15-attribute one fitted")
    if n_rows is not None and n_rows > MAX_ROWS_AT_WIDTH and n_cols >= WIDTH_FOR_ROW_LIMIT:
        reasons.append(
            f"{n_rows:,} rows at {n_cols} attributes: the same schema at the same domain moved "
            f"from a 26-minute fit to a timeout between 26,048 and 81,410 rows")
    if len(wide) > MAX_WIDE_NUMERIC_AT_WIDTH and n_cols >= WIDTH_FOR_WIDE_LIMIT:
        reasons.append(
            f"{len(wide)} high-cardinality numeric columns ({', '.join(wide[:6])}"
            f"{', ...' if len(wide) > 6 else ''}): the one schema in our measurements with this "
            f"shape did not converge, and a width-matched control showed the cause was those "
            f"columns rather than the width")

    report = {
        "n_columns": n_cols,
        "n_rows": n_rows,
        "wide_numeric_columns": wide,
        "domain_size": size,          # reported because users expect it; NOT used to decide
        "domain_size_is_not_predictive": True,
        "feasible": not reasons,
        "reasons": reasons,
        "observations": [
            {"dataset": d, "cols": c, "rows": r, "result": res} for d, c, r, res in AIM_OBSERVATIONS
        ],
    }
    if n_rows is None:
        report.setdefault("notes", []).append(
            "n_rows was not supplied, so the row-count check was skipped. Row count is one of the "
            "three factors that predicted convergence; pass n_rows=len(df) for the full check.")
    if reasons:
        msg = (
            f"schema {schema.name!r} sits outside the region where AIM converged in our "
            f"measurements:\n"
            + "".join(f"  - {r}\n" for r in reasons)
            + f"  (estimated domain {size:,.0f} states -- reported for reference only; domain size "
              f"did NOT predict convergence in our measurements and is not used here.)\n"
              f"  Options: drop or coarsen the high-cardinality numeric columns first -- that is "
              f"what rescued the one failing schema we diagnosed -- reduce attributes, subsample "
              f"rows, or use MST, whose cost grew far more gently across the same schemas.\n"
              f"  This is a heuristic from ten fits under one implementation, not a theory. Pass "
              f"raise_on_fail=False to proceed and measure it yourself."
        )
        report["message"] = msg
        if raise_on_fail:
            raise DomainTooLargeError(msg)
    return report


# ── the DP conditional table ────────────────────────────────────────────────────────

def release_conditional_table(schema: Schema, df: pd.DataFrame, columns: tuple[str, ...], *,
                              epsilon: float, n_min: int = 150,
                              seed: int | None = None,
                              max_rows_per_person: int | None = None,
                              acknowledge_vacuous_privacy_unit: bool = False) -> ConditionalTable:
    """Release P(target | cell) under DP over a disjoint partition.

    The cells partition the data, so by parallel composition the whole table costs `epsilon`
    however many cells it contains — accuracy is set by cell *support*, not cell count. This is
    what makes a rich table affordable, and richness is what makes the correction work at all.
    """
    if n_min < 50:
        raise ValueError(
            f"n_min={n_min} is below the safe floor of 50: a rate over fewer records is dominated "
            f"by its own Laplace noise.")
    schema.validate(df, strict=True)     # before any budget is spent
    data = schema.coerce(df)
    rng = np.random.default_rng(seed)
    # epsilon protects one ROW; see cortec.accounting.PrivacyLedger for why this is
    # declared by the caller rather than inferred. This tool's whole purpose is correcting an
    # EXISTING synthesiser's output, so it is the one most likely to be pointed at whatever table
    # the institution already has -- including encounter-level clinical data -- which is exactly
    # where a row-level epsilon gets read as a per-person one. Same gate as `cortec`.
    if max_rows_per_person is not None:
        _eps_person = float(epsilon) * int(max_rows_per_person)
        if _eps_person > VACUOUS_EPSILON_PER_PERSON and not acknowledge_vacuous_privacy_unit:
            raise ValueError(
                f"REFUSED: max_rows_per_person={max_rows_per_person} at epsilon={epsilon} gives "
                f"epsilon_per_person={_eps_person:.1f}, past the "
                f"{VACUOUS_EPSILON_PER_PERSON:.0f} beyond which a per-person guarantee carries no "
                f"meaning. The privacy unit here is one row, so unaggregated longitudinal data "
                f"degrades the guarantee by the maximum contribution count. Either aggregate to "
                f"one row per person before releasing, or lower epsilon, or pass "
                f"acknowledge_vacuous_privacy_unit=True to proceed with a ROW-level guarantee and "
                f"have the acknowledgement recorded.")
    led = PrivacyLedger(epsilon, max_rows_per_person=max_rows_per_person)

    keys = _cell_keys(data, columns, schema)
    cells: dict[str, float] = {}
    support: dict[str, int] = {}
    # `support` is PUBLISHED on the returned ConditionalTable. It was the exact count of private
    # records per cell — unnoised and never charged — so the audit reported a clean epsilon while
    # the release carried the true cell sizes. Same defect as cortec's release.py. Cells partition
    # the data, so the counts are parallel composition: ONE epsilon for the whole group however
    # many cells there are. They are a separate GROUP from the rates because a record contributes
    # to both its rate and its count, so the two compose sequentially.
    eps_counts = 0.05 * epsilon
    eps_rates = epsilon - eps_counts
    for cell, part in data.groupby(keys):
        if len(part) < n_min:
            continue
        # Same correction as cortec's release.py (technical report, section 4.3): a bounded-mean rate noised at
        # scale 1/(|cell|·eps) has a data-dependent scale under add/remove-one adjacency and is
        # not pure eps-DP. Release the positive COUNT (sensitivity exactly 1) and divide by the
        # separately charged noised support, floored at the public n_min. The published rate is
        # post-processing of two DP quantities; the budget partition is unchanged.
        c_scale = led.spend(f"count[{cell}]", kind="count", epsilon=eps_counts,
                            sensitivity=1.0, composition="parallel",
                            partition=str(cell), group="published_counts")
        # clamped at n_min, as cortec's cohort sizes and cell supports are: the cell is in the
        # table only because it holds >= n_min records, so a lower published support is
        # impossible under the table's own rule (post-processing of the noised count)
        support[str(cell)] = int(max(n_min, round(len(part) + laplace_noise(rng, c_scale))))
        pos_true = float((part[schema.target].astype(str).str.strip() == schema.positive).sum())
        scale = led.spend(f"cond[{cell}]", kind="count", epsilon=eps_rates,
                          sensitivity=1.0, composition="parallel",
                          partition=str(cell), group="conditional")
        pos_noisy = max(0.0, pos_true + laplace_noise(rng, scale))
        cells[str(cell)] = float(np.clip(pos_noisy / float(max(support[str(cell)], n_min)),
                                         0.0, 1.0))

    if not cells:
        raise ValueError(
            f"no cell reached n_min={n_min}; the table is empty and would correct nothing. "
            f"Use fewer conditioning columns or lower n_min.")
    led.assert_within_budget()
    led.seal()
    return ConditionalTable(tuple(columns), cells, support, epsilon, led.audit_report())


def _cell_keys(df: pd.DataFrame, cols: tuple[str, ...], schema: Schema) -> pd.Series:
    parts = []
    for c in cols:
        if c in schema.numerical:
            edges = schema.bin_edges(c)
            v = pd.to_numeric(df[c], errors="coerce")
            idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
            parts.append(pd.Series([f"{c}[{edges[i]:g},{edges[i+1]:g})" for i in idx],
                                   index=df.index))
        else:
            v = df[c].astype(str).str.strip()
            # Honour a declared coarsening, exactly as `cortec.release._cell_keys` does. Without
            # this the SAME Schema produces different cells in the two tools, and a caller moving
            # a schema between them would silently get a different conditional table.
            cmap = (getattr(schema, "coarsen", None) or {}).get(c)
            if cmap:
                v = v.map(lambda x: cmap.get(x, x))
            parts.append(c + "=" + v)
    return pd.Series([" & ".join(t) for t in zip(*parts)], index=df.index)


# ── the correction ──────────────────────────────────────────────────────────────────

def relabel(schema: Schema, synthetic: pd.DataFrame, table: ConditionalTable, *,
            preserve_ranking: bool = False, seed: int | None = None) -> pd.DataFrame:
    """Relabel `synthetic`'s target column so each cell matches the DP table's rate.

    Pure post-processing of two already-DP artifacts: it costs no additional privacy budget.

    Within a cell, labels are assigned i.i.d. by default. `preserve_ranking=True` instead orders
    records by a model fitted on the synthetic data's own target.

    **Rank preservation used to be the default and it was measured to be WRONG.** Three datasets,
    three MST seeds each, downstream AUC against the real held-out split:

        renal_registry   MST 0.631 -> rank 0.546 (-0.085) -> i.i.d. 0.614 (-0.017)
        diabetes         MST 0.581 -> rank 0.528 (-0.053) -> i.i.d. 0.549 (-0.032)
        adult            MST 0.705 -> rank 0.788 (+0.083) -> i.i.d. 0.821 (+0.116)

    i.i.d. beat ranking in **9 of 9 runs**. An earlier version of this module claimed that i.i.d.
    assignment "bought essentially nothing (+0.002 AUC)"; that claim did not survive replication.

    Why the guard did not catch it: `_rank_scores` gated on whether the synthetic target is
    PREDICTABLE from the synthetic features (CV AUC >= RANK_INFORMATIVENESS_FLOOR). A synthesiser
    generates its target AS a function of those features, so that score is near-perfect almost
    always — renal_registry scores 0.981 while its real downstream AUC is 0.631. The gate measures
    SELF-CONSISTENCY, not validity, and it is highest exactly when the synthesiser has confidently
    learned a wrong relationship. It therefore selected FOR the failure mode.

    Rate matching alone is the mechanism. Preserving the ordering requires evidence that the
    synthesiser's within-cell ordering is real, and nothing available to this tool provides it.
    """
    out = synthetic.copy()
    rng = np.random.default_rng(seed)
    keys = _cell_keys(out, table.columns, schema)

    scores = _rank_scores(schema, out, seed=seed) if preserve_ranking else None
    # scores is None when the input's own target carries no learnable structure; assignment then
    # falls back to i.i.d. within the cell, which is strictly better than ordering by noise.

    for cell, idx in out.groupby(keys).groups.items():
        rate = table.cells.get(str(cell))
        if rate is None:
            continue                      # no released rate for this cell: leave it alone
        idx = list(idx)
        # STOCHASTIC rounding, matching cortec's `positives_for_cell`. Plain round() is
        # systematically biased in exactly the small cells this correction exists to fix: at a
        # released rate of 0.30 it yields a realised 0.000 for a 1-row cell (every positive
        # vanishes), 0.500 for 2 rows and 0.400 for 5, against 0.300 throughout for stochastic.
        # A rate-matching tool must not bias the rates it matches.
        _exact = float(rate) * len(idx)
        _base = int(np.floor(_exact))
        k = _base + (1 if rng.random() < (_exact - _base) else 0)
        k = max(0, min(len(idx), k))
        if preserve_ranking and scores is not None:
            order = sorted(idx, key=lambda i: -scores.loc[i])
        else:
            order = list(rng.permutation(idx))
        pos = set(order[:k])
        out.loc[idx, schema.target] = [schema.positive if i in pos else schema.negative
                                       for i in idx]
    return out


def _rank_scores(schema: Schema, df: pd.DataFrame,
                 seed: int | None = None) -> pd.Series | None:
    """Score each synthetic record by its propensity for the positive class, or return None if
    ranking would do more harm than good.

    Fitted on the SYNTHETIC data only — it never sees a private record, which is what keeps this
    step post-processing.

    The None case is the important one and it was found by testing rather than assumed. Ranking
    only helps when the input synthetic data already carries within-cell structure worth
    preserving. When its target is close to noise — which is exactly the case for a synthesiser
    whose conditional structure is weak, i.e. the case this tool is built for — a model fitted on
    that target learns the noise, and ordering by it *injects* spurious within-cell structure that
    a downstream model then mistakes for signal. In our own fixture that cost 0.09 AUC against
    plain i.i.d. assignment. So the informativeness of the ranking is measured by cross-validation
    first, and ranking is used only if it clears chance by a clear margin.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import OrdinalEncoder

    feats = [c for c in schema.numerical_cols + schema.categorical_cols if c in df.columns]
    X = df[feats].copy()
    cat = [c for c in schema.categorical_cols if c in X.columns]
    if cat:
        X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1).fit_transform(X[cat].astype(str))
    for c in schema.numerical_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    y = (df[schema.target].astype(str).str.strip() == schema.positive).astype(int)
    if y.nunique() < 2 or len(df) < 50:
        return None

    m = RandomForestClassifier(n_estimators=200, random_state=seed or 0, n_jobs=-1)
    try:
        auc = float(np.mean(cross_val_score(m, X, y, cv=3, scoring="roc_auc")))
    except Exception:
        return None
    if auc < RANK_INFORMATIVENESS_FLOOR:
        # the input's own target is ~unpredictable from its features: nothing to preserve
        return None
    m.fit(X, y)
    return pd.Series(m.predict_proba(X)[:, 1], index=df.index)


# ── did it help? ────────────────────────────────────────────────────────────────────

def estimate_gain(schema: Schema, before: pd.DataFrame, after: pd.DataFrame,
                  table: ConditionalTable) -> dict:
    """Measure what the correction actually changed, using only released quantities.

    Reports conditional error against the DP table before and after, and the change in marginal
    fidelity between the two synthetic sets — because a correction that fixes conditional
    structure while wrecking marginals is not a win, and the user should see both.
    """
    def cond_err(df):
        keys = _cell_keys(df, table.columns, schema)
        errs = []
        for cell, part in df.groupby(keys):
            rate = table.cells.get(str(cell))
            if rate is None or len(part) == 0:
                continue
            got = float((part[schema.target].astype(str).str.strip() == schema.positive).mean())
            errs.append(abs(got - rate))
        return float(np.mean(errs)) if errs else float("nan")

    def marg_shift():
        shifts = []
        for c in schema.categorical_cols:
            if c not in before.columns or c not in after.columns:
                continue
            a = before[c].astype(str).value_counts(normalize=True)
            b = after[c].astype(str).value_counts(normalize=True)
            keys = set(a.index) | set(b.index)
            shifts.append(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys))
        return float(np.mean(shifts)) if shifts else 0.0

    e0, e1 = cond_err(before), cond_err(after)
    out = {"conditional_error_before": e0, "conditional_error_after": e1,
           "conditional_error_reduction": e0 - e1,
           "feature_marginal_shift": marg_shift(),
           "n_cells_corrected": len(table)}
    out["worth_it"] = bool(e0 - e1 > 0.02)
    out["verdict"] = (
        "the correction materially improved conditional calibration"
        if out["worth_it"] else
        "the correction changed little; the marginal synthesiser's conditional structure was "
        "already close to the released table, so consider keeping the budget instead")
    # `worth_it` is read as "should I use this?" and it does not answer that question. It is a
    # CALIBRATION criterion, and we measured the two coming apart: across 11 datasets the
    # correction improved calibration in essentially every case while downstream AUC moved in both
    # directions -- of 9 runs flagged worth_it, downstream AUC FELL on 6 in one run and 3 of 10 in
    # a repeat. Naming what a verdict does not cover belongs beside the verdict, not only in a paper.
    out["worth_it_measures"] = "conditional calibration against the released table, nothing else"
    out["worth_it_does_not_predict"] = (
        "downstream utility. Across 11 datasets this correction improved calibration in "
        "essentially every case while downstream AUC moved in BOTH directions; of 9 runs flagged "
        "worth_it, downstream AUC fell on 6 in one run and on 3 of 10 in a repeat, and the "
        "direction was not stable between runs. Three datasets were reliably negative (a renal "
        "registry, hospital readmission, cervical cancer): there the correction reliably improved "
        "calibration AND reliably reduced downstream AUC. Validate on your own held-out data "
        "before adopting this; do not read worth_it as a recommendation.")
    return out


def correct(schema: Schema, private: pd.DataFrame, synthetic: pd.DataFrame, *,
            columns: tuple[str, ...], epsilon: float, n_min: int = 150,
            seed: int | None = None) -> tuple[pd.DataFrame, ConditionalTable, dict]:
    """Release a table, relabel, and report whether it helped. The one-call entry point."""
    table = release_conditional_table(schema, private, columns, epsilon=epsilon,
                                      n_min=n_min, seed=seed)
    # i.i.d. within the cell: measured better than rank preservation in 9 of 9 runs across three
    # datasets and three synthesiser seeds. See `relabel`'s docstring for the numbers and for why
    # the informativeness gate could not detect the problem.
    out = relabel(schema, synthetic, table, preserve_ranking=False, seed=seed)
    return out, table, estimate_gain(schema, synthetic, out, table)
