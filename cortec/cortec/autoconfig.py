"""autoconfig.py — derive a conditional hierarchy from a schema, privately. On by default.

A CoRTeC deployment has to decide which columns the conditional table conditions on and how fine
that table should be. Choosing those by inspecting the data is what a research study does; it is
not something a deploying institution can legitimately do, and in our own study getting it wrong
cost real utility — a hierarchy conditioned on two demographic attributes carrying 0.009 and 0.004
of mutual information while omitting two payment attributes carrying 0.057 and 0.042.

So the tool derives it. `Schema.conditional` left empty means "derive"; supplying it means "I know
what I want, use mine". Autoconfig is the default because the failure mode of hand-tuning is silent:
nothing in the output says the hierarchy was the wrong one.

The derivation spends a small, declared share of the budget on noised (column x target) contingency
tables — L1 sensitivity 1 each — and everything after the noised counts is post-processing. Wide
columns are coarsened by their noisy positive rate, which is post-processing of tables already
released. Richness comes from public cardinalities and the requested record count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SELECTION_FRACTION_MIN = 0.05
SELECTION_FRACTION_MAX = 0.30
MAX_LEVELS_PER_COLUMN = 4
NOISE_FLOOR_MULTIPLE = 1.0
COHORT_GROUPS = 3
MIN_RECORDS_PER_CELL = 8
TARGET_RECORDS_PER_CELL = 20


def _derive_stratification(schema, private, col: str, coarsen: dict) -> list:
    """Split the top-ranked column into COHORT_GROUPS cohorts, using PUBLIC information only.

    Numeric columns are split on their DECLARED bin edges (public); categorical columns are grouped
    by their declared levels. Nothing here reads a private value: the edges and the level list come
    from the schema. Mirrors `src/autoconfig._coarse_band`.
    """
    from .schema import Band
    if col in schema.numerical:
        edges = list(schema.bin_edges(col))
        if len(edges) < 3:
            return []
        # pick COHORT_GROUPS-1 interior cut points, evenly spaced over the declared edges
        inner = edges[1:-1]
        if not inner:
            return []
        step = max(1, len(inner) // max(1, COHORT_GROUPS - 1))
        cuts = inner[::step][: COHORT_GROUPS - 1]
        pts = [edges[0]] + list(cuts) + [edges[-1]]
        bands = [Band(f"{col}_c{i}", float(pts[i]), float(pts[i + 1]))
                 for i in range(len(pts) - 1)]
        return [(col, bands)] if len(bands) >= 2 else []
    # Categorical stratification would have to group declared levels into COHORT_GROUPS buckets,
    # and `Schema.stratify` has no representation for that — `_cohort_keys` groups a categorical
    # column by RAW value, which gave 14 cohorts on Adult's `education` against the 3 the research
    # pipeline derives. Rather than add a half-supported representation, stratify on the
    # highest-ranked NUMERIC column, where declared bin edges give exactly the banding needed.
    return []


def _first_numeric(schema, ranked: list[str]) -> str | None:
    """Highest-ranked column that `Schema.stratify` can actually band."""
    for c in ranked:
        if c in schema.numerical:
            return c
    return None


@dataclass
class DerivedConfig:
    columns: tuple[str, ...]
    levels: list[tuple[str, ...]]
    coarsen: dict[str, dict[str, str]]
    epsilon_selection: float
    cells_at_finest: int
    notes: list[str]
    # Cohort stratification, in Schema.stratify form. This was MISSING: the tool derived a
    # conditional hierarchy but never a stratification, so a caller using autoconfig as documented
    # got ONE global cohort while the research pipeline derives COHORT_GROUPS of them. Per-cohort
    # marginals are a core element of the method, and they were collapsing to global marginals —
    # the shipped default was structurally weaker than the configuration our results validate.
    stratify: list = field(default_factory=list)

    def describe(self) -> str:
        head = (f"derived hierarchy: {list(self.columns)}  "
                f"({self.cells_at_finest} cells at the finest level, "
                f"selection spent eps={self.epsilon_selection:.3f})")
        return "\n".join([head] + [f"  note: {n}" for n in self.notes])


def _band(schema, df: pd.DataFrame, col: str) -> pd.Series:
    if col in schema.numerical:
        edges = np.asarray(schema.bin_edges(col), dtype=float)
        return pd.cut(pd.to_numeric(df[col], errors="coerce"),
                      bins=edges, include_lowest=True).astype(str)
    return df[col].astype(str).str.strip()


def _mi(tab: np.ndarray) -> float:
    tab = np.clip(tab, 0.0, None)
    t = tab.sum()
    if t <= 0:
        return 0.0
    p = tab / t
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(p * np.log(np.where(p > 0, p / np.maximum(px * py, 1e-12), 1.0))))


def derive(schema, private: pd.DataFrame, *, epsilon_total: float, n_records: int,
           n_min: int = 150,
           selection_fraction: float | None = None, seed: int | None = None,
           ledger=None) -> DerivedConfig:
    """Derive a conditional hierarchy. Consumes `selection_fraction` of `epsilon_total`.

    `ledger`, when supplied, is charged for the selection query HERE and supplies the noise scale,
    which is the package's central invariant: `PrivacyLedger.spend()` is meant to be the only
    function that returns a noise scale, so that nothing can be released without appearing in the
    audit trail. This function used to compute `m / eps_sel` itself and be charged separately by
    the caller afterwards. The two agreed, but only by convention: a change to the epsilon in one
    place and not the other would have added noise at one budget while charging another, silently,
    which is precisely the class of defect the ledger exists to prevent. Callers that pass no
    ledger keep the old behaviour and must charge the selection themselves.
    """
    cols = [c for c in list(schema.numerical) + list(schema.categorical)]
    m = max(1, len(cols))
    lv = [len(set(_band(schema, private, c))) for c in cols] or [2]
    ny = max(2, private[schema.target].nunique())
    notes: list[str] = []

    if selection_fraction is None:
        need = NOISE_FLOOR_MULTIPLE * m * float(np.median(lv)) * ny / max(1, len(private))
        selection_fraction = float(np.clip(need, SELECTION_FRACTION_MIN, SELECTION_FRACTION_MAX))
        notes.append(f"selection budget set from schema size and n: {selection_fraction:.0%}")
    eps_sel = epsilon_total * selection_fraction
    if ledger is not None:
        # sensitivity m: one record moves m contingency tables by one cell each
        scale = ledger.spend("autoconfig: conditional-column selection", kind="histogram",
                             epsilon=eps_sel, sensitivity=float(m),
                             composition="sequential", partition="__all__", group="autoconfig")
    else:
        scale = m / eps_sel
    # Seeding makes the SELECTION deterministic, and the selection spends eps_selection out of the
    # release budget on a query over the private data. A deterministic mechanism has a point-mass
    # output distribution, so that budget buys nothing. Default (None) is OS entropy and correct.
    if seed is not None:
        from . import report
        report.emit("\n" + report.warn(f"AUTOCONFIG WAS SEEDED (seed={seed}). Column selection is reproducible, so "
                                       f"the eps_selection charged for it does NOT provide its guarantee. Use seed=None "
                                       f"outside tests.") + "\n")
    rng = np.random.default_rng(seed)
    y = private[schema.target].astype(str).str.strip()
    ylv = sorted(y.unique().tolist())

    scored, coarsen = [], {}
    for c in cols:
        x = _band(schema, private, c)
        levels = sorted(x.unique().tolist())
        if len(private) / max(1, len(levels) * ny) < NOISE_FLOOR_MULTIPLE * scale:
            continue                       # cells smaller than their own noise: not rankable
        tab = pd.crosstab(x, y).reindex(index=levels, columns=ylv,
                                        fill_value=0).to_numpy(float)
        tab = tab + rng.laplace(0.0, scale, tab.shape)
        scored.append((c, _mi(tab)))
        # Coarsening groups DECLARED categorical levels. A numeric column is banded by its public
        # bin edges and is never coarsened: `Schema.coarsen` rejects a numeric key, so deriving one
        # crashed `release_statistics(autoconfig=True)` on any schema whose numeric column has more
        # bins than MAX_LEVELS_PER_COLUMN (defect: autoconfig coarsened a numeric column).
        if len(levels) > MAX_LEVELS_PER_COLUMN and c in schema.categorical:
            rate = np.divide(np.clip(tab, 0, None)[:, -1],
                             np.maximum(np.clip(tab, 0, None).sum(1), 1e-9))
            per = max(1, int(np.ceil(len(levels) / MAX_LEVELS_PER_COLUMN)))
            coarsen[c] = {levels[i]: f"g{r // per}" for r, i in enumerate(np.argsort(rate))}
    scored.sort(key=lambda kv: -kv[1])

    if not scored:
        notes.append("no column's cells survived the selection noise at this epsilon and sample "
                     "size; conditioning on nothing, which is the safe fallback")
        return DerivedConfig((), [()], {}, eps_sel, 1, notes)

    effective_n = max(1.0, n_records / float(COHORT_GROUPS))
    best, cells_at, cells = (1, 1, float("inf")), 1, 1
    for k, (c, _) in enumerate(scored, start=1):
        nlv = len(set(coarsen[c].values())) if c in coarsen else len(set(_band(schema, private, c)))
        cells *= max(1, nlv)
        if effective_n / cells < MIN_RECORDS_PER_CELL:
            break
        gap = abs(effective_n / cells - TARGET_RECORDS_PER_CELL)
        if gap < best[2]:
            best, cells_at = (k, cells, gap), cells
    k = best[0]
    chosen = tuple(c for c, _ in scored[:k])
    # Stratify ONLY when the data can support it. Splitting into COHORT_GROUPS cohorts that each
    # fall below n_min releases nothing at all — a 240-row frame split three ways has ~80 per
    # cohort against n_min=150, and the whole release fails. The research pipeline reaches the
    # same place from the other direction: its small-dataset specs declare no stratification.
    # This uses only len(private), which the CALLER supplied; cohort sizes remain private and are
    # charged separately by the suppression query.
    _can_stratify = len(private) / float(COHORT_GROUPS) >= n_min
    _strat_col = _first_numeric(schema, [c for c, _ in scored])
    _strat = (_derive_stratification(schema, private, _strat_col, coarsen)
              if _strat_col and _can_stratify else [])
    if chosen and not _can_stratify:
        notes.append(f"not stratifying: {len(private)} records over {COHORT_GROUPS} cohorts "
                     f"would give ~{len(private)//COHORT_GROUPS} each, below n_min={n_min}")
    return DerivedConfig(chosen, [chosen[:i] for i in range(len(chosen) + 1)],
                         {c: v for c, v in coarsen.items() if c in chosen},
                         eps_sel, cells_at, notes, _strat)
