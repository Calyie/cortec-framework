"""select.py — return a generated pool to the released marginals before handing records over.

The generator's marginal error is decoding error, not privacy noise: a frozen model asked for a
batch of rows reproduces a released histogram only approximately, and its output then carries that
approximation plus the multinomial noise of any sample of n rows. The release is public, so nothing
stops us from asking for more rows than we need and choosing, among them, the n whose marginals
match the release. Every quantity used here is in the release (cohort sizes, class balance, one
histogram per cohort, class and column), so the choice is post-processing and costs no privacy
budget. It is the same reason a marginal synthesiser's output is smoother than a real sample: rows
are allocated to released cells by quota rather than drawn independently.

Method. Each pooled row is assigned to its cohort (the public stratification rule), its class, and
one bin per column (public edges / declared categories). Iterative proportional fitting finds
inclusion weights in [0, 1] whose sums over every released cell match the release's expected counts
at size n. Rows are then drawn by systematic sampling on the cumulative weights, grouped by cell, so
each cell's count lands within one row of its expected value.

Two details matter and are tested. A cohort whose release carries class-conditional blocks is
constrained per (class, column); a cohort with only a pooled block is constrained per column at the
cohort level, because imposing a pooled histogram on each class separately would force both classes
to the same marginals and erase the feature-target structure the release paid for. And a category
the release suppressed (below n_min) receives zero mass, so a row the generator invented for it is
never selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .release import Release, _cohort_keys
from .schema import Schema


def _cells(schema: Schema, df: pd.DataFrame, release: Release) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # the cohort rule the release was built under (autoconfig derives one the caller's schema
    # does not carry); without it every row mapped to one cohort and the selection matched nothing
    out["_cohort"] = _cohort_keys(release.effective_schema(schema), df).astype(str).values
    out["_pos"] = (df[schema.target].astype(str).str.strip() == str(schema.positive)).values
    first = release.cohorts[0]
    for c in schema.numerical:
        edges = np.asarray(first["numerical"][c]["bin_edges"], float)
        v = pd.to_numeric(df[c], errors="coerce").fillna(edges[0]).values
        out[c] = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    for c in schema.categorical:
        out[c] = df[c].astype(str).str.strip().values
    return out


def _hist_constraints(cons: list, cells: pd.DataFrame, schema: Schema, mask, block: dict, scale: float) -> None:
    for c in schema.numerical:
        h = np.asarray(block["numerical"][c]["proportions"], float); h = h / max(h.sum(), 1e-12)
        for b, p in enumerate(h):
            cons.append((mask & (cells[c].values == b), scale * float(p)))
    for c in schema.categorical:
        pr = block["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
        for k, p in pr.items():
            cons.append((mask & (cells[c].values == str(k)), scale * float(p) / tot))
        cons.append((mask & ~cells[c].isin([str(k) for k in pr]).values, 0.0))   # suppressed → zero


MIN_EXPECTED_ROWS = 1.0   # a released cell expecting fewer rows than this cannot be met by selection


def inclusion_weights(schema: Schema, release: Release, pool: pd.DataFrame, n_rows: int, *,
                      rounds: int = 40, min_expected_rows: float = MIN_EXPECTED_ROWS) -> np.ndarray:
    """Inclusion weights in [0, 1] summing to n_rows whose cell sums match the release.

    Constraints expecting fewer than `min_expected_rows` rows are dropped (zero-mass constraints
    for suppressed categories are kept): at integer resolution they cannot be satisfied, and fitting
    to them measurably worsened held-out conditional error and utility on Adult while buying nothing
    on the marginals."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    cells = _cells(schema, pool, release)
    total = sum(float(c["cohort_size"]) for c in release.cohorts) or 1.0
    w = np.full(len(pool), min(1.0, n_rows / max(len(pool), 1)))
    cons: list = []
    for coh in release.cohorts:
        m_c = cells["_cohort"].values == str(coh["cohort_name"])
        mass = n_rows * float(coh["cohort_size"]) / total
        cons.append((m_c, mass))
        cb = coh.get("class_balance") or {}
        by_class = coh.get("by_class") or {}
        for is_pos, cls in ((True, str(schema.positive)), (False, str(schema.negative))):
            share = float(cb.get(cls, 0.5))
            m_cy = m_c & (cells["_pos"].values == is_pos)
            cons.append((m_cy, mass * share))
            if cls in by_class:
                _hist_constraints(cons, cells, schema, m_cy, by_class[cls], mass * share)
        if not by_class:
            _hist_constraints(cons, cells, schema, m_c, coh, mass)
    cons = [(m, t) for m, t in cons if t == 0.0 or t >= min_expected_rows]
    for _ in range(rounds):
        for mask, t in cons:
            s = w[mask].sum()
            if s > 0:
                w[mask] *= (t / s) if t > 0 else 0.0
        np.clip(w, 0.0, 1.0, out=w)
        tot = w.sum()
        if tot > 0:
            w *= n_rows / tot
            np.clip(w, 0.0, 1.0, out=w)
    return w


def _release_cell_matrix(schema: Schema, release: Release, pool: pd.DataFrame, n_rows: int,
                         w_pooled: float = 3.0):
    """Indicator matrix (rows x released cells), each cell's target count at size n_rows, and its
    weight. Two families: the per-(cohort, class, column-bin) counts the release states, and the
    pooled marginals they imply, which is what 1-way total variation measures; the pooled family
    is weighted above 1, which is what moves marginal error to the release's own floor."""
    cells = _cells(schema, pool, release)
    total = sum(float(c["cohort_size"]) for c in release.cohorts) or 1.0
    cols, targ, wts = [], [], []

    def add(mask, t, w=1.0):
        cols.append(mask); targ.append(t); wts.append(w)

    coh_of, pos_of = cells["_cohort"].values, cells["_pos"].values
    pos, neg = str(schema.positive), str(schema.negative)
    for coh in release.cohorts:
        m_c = coh_of == str(coh["cohort_name"]); mass = n_rows * float(coh["cohort_size"]) / total
        add(m_c, mass)
        cb = coh.get("class_balance") or {}; by = coh.get("by_class") or {}
        for is_pos, cls in ((True, pos), (False, neg)):
            share = float(cb.get(cls, 0.5)); m_cy = m_c & (pos_of == is_pos)
            add(m_cy, mass * share)
            blk = by.get(cls)
            if blk is None:
                continue
            for c in schema.numerical:
                h = np.asarray(blk["numerical"][c]["proportions"], float); h = h / max(h.sum(), 1e-12)
                for b, p in enumerate(h):
                    add(m_cy & (cells[c].values == b), mass * share * float(p))
            for c in schema.categorical:
                pr = blk["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
                for k, p in pr.items():
                    add(m_cy & (cells[c].values == str(k)), mass * share * float(p) / tot)
        if not by:
            for c in schema.numerical:
                h = np.asarray(coh["numerical"][c]["proportions"], float); h = h / max(h.sum(), 1e-12)
                for b, p in enumerate(h):
                    add(m_c & (cells[c].values == b), mass * float(p))
            for c in schema.categorical:
                pr = coh["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
                for k, p in pr.items():
                    add(m_c & (cells[c].values == str(k)), mass * float(p) / tot)
    first = release.cohorts[0]
    for c in schema.numerical:
        e = np.asarray(first["numerical"][c]["bin_edges"], float); p = np.zeros(len(e) - 1)
        for coh in release.cohorts:
            h = np.asarray(coh["numerical"][c]["proportions"], float)
            p += float(coh["cohort_size"]) / total * h / max(h.sum(), 1e-12)
        for b, pp in enumerate(p):
            add(cells[c].values == b, n_rows * float(pp), w_pooled)
    for c in schema.categorical:
        pm: dict = {}
        for coh in release.cohorts:
            pr = coh["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
            for k, v in pr.items():
                pm[str(k)] = pm.get(str(k), 0.0) + float(coh["cohort_size"]) / total * float(v) / tot
        for k, pp in pm.items():
            add(cells[c].values == str(k), n_rows * float(pp), w_pooled)
        # a category the release suppressed has zero mass: any row carrying it costs the objective
        add(~cells[c].isin(list(pm)).values, 0.0, 10.0 * w_pooled)
    return np.array(cols, dtype=np.float32).T, np.array(targ, float), np.array(wts, float)


def refine_selection(schema: Schema, release: Release, pool: pd.DataFrame, chosen: np.ndarray, *,
                     seed: int | None = None, max_rounds: int = 40, w_pooled: float = 3.0) -> np.ndarray:
    """Swap rows in and out of `chosen` while the weighted L1 distance to the released cell counts
    falls. The rake's fractional weights can only be approximated at integer resolution; this
    closes the remainder. Reads only the release and the pool."""
    A, t, w = _release_cell_matrix(schema, release, pool, int(chosen.sum()), w_pooled)
    cur = A[chosen].sum(0); L = float((w * np.abs(cur - t)).sum())
    rng = np.random.default_rng(seed)
    for _ in range(max_rounds):
        moved = 0
        ins = np.where(~chosen)[0]; rng.shuffle(ins)
        for i in ins:
            outs = np.where(chosen)[0]
            cand = cur + A[i] - A[outs]
            losses = (w * np.abs(cand - t)).sum(1)
            j = int(np.argmin(losses))
            if losses[j] < L - 1e-9:
                chosen[i] = True; chosen[outs[j]] = False
                cur = cand[j]; L = float(losses[j]); moved += 1
        if not moved:
            break
    return chosen


def release_subbin_values(schema: Schema, release: Release, df: pd.DataFrame, *,
                          seed: int | None = None) -> pd.DataFrame:
    """Make the output's structure below bin resolution the release's own wherever the release
    states the class-conditional shape.

    A released histogram fixes how many rows fall in each declared bin and nothing finer, so a
    value's position inside its bin is never released information. In a cohort that carries
    class-conditional blocks the class shape is the release's at bin resolution and whatever the
    generator does below it is prior (on NHANES it separated the classes about twice as far as the
    data inside each bin and cost the linear student 0.044 AUC; paper, Appendix H.17): every
    numeric value there is redrawn uniformly inside its declared bin, intersected with the row's
    stratification band so cohort membership never moves. A cohort with only a pooled block keeps
    the generator's values: there they are the only carrier of the class signal. Post-processing.
    Integer-valued columns draw integers inside the bin, so no bin count changes."""
    rng = np.random.default_rng(seed); out = df.copy()
    eff = release.effective_schema(schema)
    coh = _cohort_keys(eff, df).astype(str).values
    scoped = np.isin(coh, [str(c["cohort_name"]) for c in release.cohorts if c.get("by_class")])
    bands = {col: np.asarray([b.lo for b in bs] + [bs[-1].hi], float)
             for col, bs in (eff.stratify or []) if col in schema.numerical and bs}
    for c in schema.numerical:
        edges = np.asarray(schema.bin_edges(c), float)
        v = pd.to_numeric(out[c], errors="coerce").values.astype(float); fin = np.isfinite(v)
        if not (scoped & fin).any():
            continue
        is_int = bool(np.all(np.mod(v[fin], 1) == 0))
        b = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
        lo = edges[b].copy(); hi = edges[b + 1].copy()
        if c in bands:
            be = np.sort(bands[c]); k = np.clip(np.digitize(v, be[1:-1]), 0, len(be) - 2)
            lo = np.maximum(lo, be[k]); hi = np.minimum(hi, be[k + 1])
        dlo, dhi = schema.numerical[c]
        lo = np.maximum(lo, dlo); hi = np.minimum(hi, dhi + (1.0 if is_int else 0.0))
        m = scoped & fin & (hi > lo)
        new = v.copy()
        if is_int:
            ilo = np.ceil(lo[m]); ihi = np.maximum(np.ceil(hi[m]) - 1, ilo)
            new[m] = np.floor(rng.uniform(ilo, ihi + 1))
        else:
            new[m] = rng.uniform(lo[m], hi[m])
        new = np.clip(new, dlo, dhi)
        out[c] = new.astype(int) if is_int and np.isfinite(new).all() else new
    return out


MIN_POOL_COVER = 1.1   # each cohort's pool rows must reach this multiple of the rows it owes


def pool_coverage(schema: Schema, release: Release, pool: pd.DataFrame, n_rows: int) -> dict:
    """Rows the pool holds per released cohort against the rows that cohort owes at size n_rows.
    A generation run that stops early (spend cap, network) leaves its later cohorts unfilled, and
    no selection can repair a cohort share from rows that do not exist; this is checked before
    selecting so the shortfall is named rather than silently produced."""
    cells = _cells(schema, pool, release)
    total = sum(float(c["cohort_size"]) for c in release.cohorts) or 1.0
    out = {}
    for coh in release.cohorts:
        owed = n_rows * float(coh["cohort_size"]) / total
        have = int((cells["_cohort"].values == str(coh["cohort_name"])).sum())
        out[str(coh["cohort_name"])] = {"owed": round(owed, 1), "have": have,
                                        "ratio": (have / owed) if owed > 0 else float("inf")}
    return out


def select_to_release(schema: Schema, release: Release, pool: pd.DataFrame, n_rows: int, *,
                      seed: int | None = None, rounds: int = 40, refine: bool = True,
                      w_pooled: float = 3.0, min_cover: float = MIN_POOL_COVER,
                      allow_short_pool: bool = False, sub_bin: str = "release") -> pd.DataFrame:
    """Choose n_rows rows of `pool` whose marginals match `release`. Post-processing: reads only the
    release and the pool. Refuses a pool that does not cover every released cohort (see
    `pool_coverage`) unless `allow_short_pool=True`, in which case the shortfall is a warning."""
    if len(pool) < n_rows:
        raise ValueError(f"pool holds {len(pool)} rows, fewer than the {n_rows} requested")
    cov = pool_coverage(schema, release, pool, n_rows)
    short = {k: v for k, v in cov.items() if v["owed"] >= 1 and v["ratio"] < min_cover}
    if short:
        msg = ("the pool does not cover every released cohort: " +
               "; ".join(f"{k} has {v['have']} rows for {v['owed']:.0f} owed" for k, v in short.items()) +
               f". Selection can only choose among rows that exist, so these cohorts would come out "
               f"short. Generate the full pool (a run that stopped at its spend cap leaves its later "
               f"cohorts unfilled), or pass allow_short_pool=True to proceed anyway.")
        if not allow_short_pool:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)
    w = inclusion_weights(schema, release, pool, n_rows, rounds=rounds)
    cells = _cells(schema, pool, release)
    order_cols = ["_cohort", "_pos"] + list(schema.categorical) + list(schema.numerical)
    idx = np.lexsort([cells[c].astype(str).values for c in order_cols[::-1]])
    ww = w[idx]; tot = ww.sum()
    if tot <= 0:
        return pool.iloc[:0].copy()
    ww = ww * n_rows / tot
    cum = np.cumsum(ww); u = np.random.default_rng(seed).random()
    picks = np.floor(cum - u).astype(int)
    take = np.r_[picks[0] >= 0, np.diff(picks) > 0]
    keep = np.sort(idx[take][:n_rows])
    if refine and len(keep) == n_rows and len(pool) > n_rows:
        chosen = np.zeros(len(pool), bool); chosen[keep] = True
        keep = np.where(refine_selection(schema, release, pool, chosen, seed=seed, w_pooled=w_pooled))[0]
    out = pool.iloc[keep].reset_index(drop=True)
    if sub_bin == "release":
        out = release_subbin_values(schema, release, out, seed=seed)
    elif sub_bin != "generator":
        raise ValueError("sub_bin must be 'release' (default) or 'generator'")
    return out
