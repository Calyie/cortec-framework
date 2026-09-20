"""Stage C: the UTILITY TRANSMISSION BOUND.

A budgeted, auditable bound on how far the synthetic data's conditional structure can differ from
the private data's, over the cells a release names. It is computed under differential privacy, it
spends its own budget, and it bounds UTILITY: it says nothing about re-identification risk and must
never be presented as a privacy audit.

The claim it produces. For each released cell `c` the private data has a true positive rate `p_c`
and the synthetic data a rate `q_c`. `q_c` is a function of the synthetic output alone, so it is
public and free. `p_c` is private: this module releases a Laplace estimate `p_hat_c` of it and turns
the noise into a one-sided confidence bound,

    with probability at least 1 - alpha, simultaneously over all k released cells,
        |p_c - q_c|  <=  |p_hat_c - q_c| + b * ln(k / alpha),      b = (1 / n_min) / epsilon_cert.

Three design points carry the guarantee and are enforced here rather than documented.

  * The Laplace scale uses the PUBLIC suppression floor `n_min` as the sensitivity bound, never the
    private cell size. A scale set from the true cell size is data-dependent under add/remove
    adjacency and is not pure epsilon-DP. The bound is conservative by n_c / n_min on each cell,
    which is the price of that.
  * Cells partition the private data, so the k rate queries compose in PARALLEL: each may spend the
    full `epsilon_cert`, and the deployment has spent `epsilon_release + epsilon_cert` in total. The
    spend goes through a `PrivacyLedger`, so it appears in the audit trail like every other query.
  * The noise is drawn from a cryptographically secure, unseedable source. A seeded draw would let
    anyone holding the seed subtract the noise and recover `p_c` exactly, which voids the spend.

Floor and ceiling. `bound_with_controls()` scores three conditions against ONE noisy release of the
private rates: the synthetic data, a real hold-out sample (the ceiling, which should clear the
tolerance) and the same sample with its target permuted (the floor, which must not). If the ceiling
fails or the floor clears, the test did not discriminate and NO verdict is issued. Scoring all three
against one draw spends `epsilon_cert` once, not three times.

Cells the synthetic data does not cover are scored at the trivial bound of 1.0, so a bound cannot be
obtained by covering a convenient subset of the cells; cells with fewer than
`MIN_SYNTH_ROWS_PER_CELL` synthetic rows are reported as thin rather than silently trusted.
"""
from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .accounting import PrivacyLedger, VACUOUS_EPSILON_PER_PERSON
from .release import Release, _cell_keys
from .schema import Schema

__all__ = ["transmission_bound", "bound_with_controls", "BoundResult", "BoundReport",
           "CellBound", "BoundError", "MIN_SYNTH_ROWS_PER_CELL"]

# Below this many synthetic rows in a cell, q_c is too coarse to mean much (with one row it can
# only be 0 or 1). Such cells are reported, not silently trusted.
MIN_SYNTH_ROWS_PER_CELL = 20


class BoundError(RuntimeError):
    """The bound could not be computed as asked, and nothing was spent."""


# ── the report's fixed text: what it is, what it is not, where the guarantee stops ──────
WHAT_THIS_IS = {
    "artifact": "utility transmission bound",
    "bounds": "how far the synthetic conditional structure can differ from the private one, over "
              "the released cells, at a stated confidence and a stated epsilon",
    "is_not": "a privacy audit, a re-identification risk assessment, or any form of privacy "
              "guarantee. It does not attest to privacy and must not be presented as doing so.",
    "privacy_evidence_lives_elsewhere": "the DP release itself (its accounting record) and the "
                                        "measured re-identification evidence, not this bound",
}

DP_CLAIM = {
    "variant": "pure epsilon-DP (central / trusted-curator model)",
    "delta": 0.0,
    "neighbouring_relation": "add or remove one record (unbounded DP)",
    "privacy_unit": "one dataset row",
    "composition": "parallel across the disjoint released cells",
    "mechanism": "Laplace",
    "sensitivity": "1/n_min per released cell rate: the public size floor, not the private cell "
                   "size. The bound is conservative by n_c/n_min on each cell.",
}

DP_CAVEATS = [
    "The privacy unit is one ROW. Under group privacy a person contributing k rows receives "
    "k*epsilon; the per-person figure in this report is the declared maximum rows per person times "
    "the row-level total. Reduce k before Stage A if a per-person guarantee is needed.",
    "Noise is sampled with a floating-point Laplace generator, which is vulnerable to the Mironov "
    "(2012) least-significant-bit attack. A deployment handling real PHI should use a discrete or "
    "snapping sampler; this implementation does not, and that is a known gap.",
    "The guarantee covers the released statistics and this bound's own query. It says nothing "
    "about the generator's pretraining data, which is outside the DP boundary entirely.",
    "This is a UTILITY TRANSMISSION BOUND computed under DP. It bounds how far the synthetic "
    "conditional structure can differ from the private one; it does NOT attest to privacy, says "
    "nothing about re-identification risk, and must never be presented as a privacy audit.",
    "Which cells enter the bound is decided by the release (a cell was released only if it held "
    "at least n_min records). The release's own accounting record says whether that suppression "
    "decision was charged.",
    "Cells for which the synthetic data supplies no rows are scored at the trivial bound of 1.0, "
    "so a bound cannot be obtained by covering a convenient subset of the cells.",
]

STANDARDS_ALIGNED = {
    "NIST SP 800-226": "parameter reporting for the DP claim block, and documenting where the "
                       "guarantee does not hold",
    "NIST SP 800-188": "governance and documentation of the release package. This is the "
                       "documented utility claim attached to a release, fitness for use. It is NOT "
                       "a disclosure review: disclosure risk is assessed by the aggregation floor "
                       "and the measured re-identification evidence, not by this bound.",
}

STANDARDS_NOT_CLAIMED = {
    "HIPAA Expert Determination":
        "NOT claimed. Expert Determination requires a statistical assessment of re-identification "
        "risk. This report bounds a utility quantity, and mapping it to that route would be a "
        "category error with compliance consequences. That route is carried by the DP release "
        "itself (with epsilon stated per person) together with measured re-identification "
        "evidence, not by this bound.",
    "ISO/IEC 27559 / ISO/IEC 20889":
        "NOT claimed. These frame re-identification risk and de-identification technique. The "
        "corresponding evidence is the aggregation floor (cells of at least n_min) and the "
        "measured membership-inference results, neither of which is this report.",
    "GDPR Art. 25 / Recital 26":
        "NOT claimed. Data protection by design is carried by the architecture, in which private "
        "data never leaves the regulated zone and the prompt carries only the DP release, not by "
        "a utility bound computed downstream of it.",
}


# ── results ───────────────────────────────────────────────────────────────────────────
@dataclass
class CellBound:
    cell: str
    n_private: int
    p_hat: float
    q_synth: float | None
    noise_halfwidth: float
    bound: float
    synth_rows: int

    def to_dict(self) -> dict:
        return {"cell": self.cell, "n": self.n_private, "p_hat": round(self.p_hat, 4),
                "q_synth": None if self.q_synth is None else round(self.q_synth, 4),
                "noise_halfwidth": round(self.noise_halfwidth, 4), "bound": round(self.bound, 4),
                "synth_rows": self.synth_rows}


@dataclass
class BoundResult:
    """One condition scored against the noisy private rates."""
    condition: str
    level: int
    columns: list[str]
    epsilon_cert: float
    alpha: float
    n_cells: int
    n_cells_covered: int
    n_cells_uncovered: int
    n_cells_thin: int
    worst_case_bound: float
    mean_bound: float
    cells: list[CellBound] = field(default_factory=list)
    within_bound: bool | None = None          # set once a tolerance is applied

    def to_dict(self) -> dict:
        d = {"condition": self.condition, "level": self.level, "columns": self.columns,
             "epsilon_spent": self.epsilon_cert, "alpha": self.alpha, "n_cells": self.n_cells,
             "n_cells_covered": self.n_cells_covered, "n_cells_uncovered": self.n_cells_uncovered,
             "n_cells_thin": self.n_cells_thin, "worst_case_bound": round(self.worst_case_bound, 4),
             "mean_bound": round(self.mean_bound, 4)}
        if self.within_bound is not None:
            d["within_bound"] = bool(self.within_bound)     # a real bool, never the string "False"
        d["cells"] = [c.to_dict() for c in self.cells]
        return d


@dataclass
class BoundReport:
    """The Stage C report: three conditions, a verdict or a refusal, and the accounting."""
    synthetic: BoundResult
    ceiling: BoundResult
    floor: BoundResult
    tolerance: float
    discriminating: bool
    verdict: str | None                        # "within bound", "outside tolerance", or None
    dp_claim: dict
    accounting: dict
    meta: dict

    def to_dict(self) -> dict:
        return {"_what_this_is": dict(WHAT_THIS_IS),
                "synthetic": self.synthetic.to_dict(),
                "real-sample [CEILING]": self.ceiling.to_dict(),
                "permuted-target [FLOOR]": self.floor.to_dict(),
                "_discriminating": bool(self.discriminating),
                "_tolerance": self.tolerance,
                "_verdict": self.verdict,
                "_dp_claim": dict(self.dp_claim),
                "_dp_caveats": list(DP_CAVEATS),
                "_standards": dict(STANDARDS_ALIGNED),
                "_standards_not_claimed": dict(STANDARDS_NOT_CLAIMED),
                "_accounting": self.accounting,
                "_meta": self.meta}

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def summary(self) -> str:
        c = self.dp_claim
        lines = ["=" * 88,
                 f"STAGE C: UTILITY TRANSMISSION BOUND | level {self.synthetic.level} "
                 f"({', '.join(self.synthetic.columns) or 'global'}) | eps_cert={c['epsilon_transmission_bound']} "
                 f"| alpha={self.synthetic.alpha} | tolerance={self.tolerance}",
                 "=" * 88,
                 f"{'condition':26s} {'cells':>6s} {'cov':>5s} {'uncov':>6s} {'thin':>5s} "
                 f"{'mean':>7s} {'worst':>7s} {'within':>7s}"]
        for lab, r in (("synthetic", self.synthetic), ("real-sample [CEILING]", self.ceiling),
                       ("permuted-target [FLOOR]", self.floor)):
            lines.append(f"{lab:26s} {r.n_cells:6d} {r.n_cells_covered:5d} {r.n_cells_uncovered:6d} "
                         f"{r.n_cells_thin:5d} {r.mean_bound:7.4f} {r.worst_case_bound:7.4f} "
                         f"{'YES' if r.within_bound else 'no':>7s}")
        if not self.discriminating:
            lines.append("\n  THIS TEST DID NOT DISCRIMINATE: a bound is meaningful only when the real-sample "
                         "ceiling clears the tolerance AND the permuted-target floor does not. Adjust the "
                         "tolerance or epsilon_cert; do not report the synthetic result from this run.")
        else:
            lines.append(f"\n  Test discriminates. VERDICT: synthetic data is {self.verdict} at tolerance "
                         f"{self.tolerance} with simultaneous confidence {1 - self.synthetic.alpha:.0%} "
                         f"over {self.synthetic.n_cells} released cells.")
        lines.append(f"\n  epsilon: release {c['epsilon_release']} + bound {c['epsilon_transmission_bound']} "
                     f"= {c['epsilon_total_per_row']} per row; per person {c['epsilon_per_person']} "
                     f"(max rows per person {c['max_rows_per_person']})")
        if c.get("epsilon_per_person_vacuous"):
            lines.append("  WARNING: the per-person epsilon is vacuous; no per-person privacy claim may be "
                         "made from this report.")
        lines.append("  This bounds UTILITY. It is not a privacy audit.")
        return "\n".join(lines)


# ── the mechanism ─────────────────────────────────────────────────────────────────────
def _secure_laplace(value: float, scale: float) -> float:
    """Laplace noise from a cryptographically secure source. Deliberately unseedable."""
    if scale <= 0:
        return float(value)
    try:
        from diffprivlib.mechanisms import Laplace as _L
        return float(_L(epsilon=1.0, sensitivity=float(scale)).randomise(float(value)))
    except Exception:
        u = secrets.SystemRandom().random() - 0.5
        return float(value) - scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def _effective_schema(schema: Schema, release: Release) -> Schema:
    eff = release.effective_schema(schema)
    if release.coarsen:
        eff = replace(eff, coarsen=release.coarsen)
    return eff


def _pick_level(release: Release, level: int | None) -> dict:
    if not release.conditional_levels:
        raise BoundError("the release carries no conditional level, so there are no cells to bound")
    if level is None:
        return release.conditional_levels[-1]           # the finest released level
    for lv in release.conditional_levels:
        if int(lv["level"]) == int(level):
            return lv
    raise BoundError(f"level {level} is not in the release; released levels are "
                     f"{[lv['level'] for lv in release.conditional_levels]}")


def _positives(schema: Schema, df: pd.DataFrame) -> pd.Series:
    return df[schema.target].astype(str).str.strip() == str(schema.positive)


def _noisy_private_rates(schema: Schema, release: Release, private_df: pd.DataFrame, *,
                         epsilon_cert: float, level: int | None, ledger: PrivacyLedger | None):
    """Spend `epsilon_cert` ONCE: one Laplace draw per released cell, charged through the ledger."""
    if epsilon_cert <= 0:
        raise BoundError("epsilon_cert must be positive")
    lv = _pick_level(release, level)
    cols = tuple(lv["columns"])
    cells = list(lv["cells"].keys())
    if not cells:
        raise BoundError("the chosen level released no cells")
    eff = _effective_schema(schema, release)
    keys = _cell_keys(private_df, cols, eff)
    pos = _positives(eff, private_df)
    n_min = int(release.n_min)
    own_ledger = ledger is None
    if own_ledger:
        ledger = PrivacyLedger(epsilon_cert)
    p_hat: dict[str, float] = {}
    n_priv: dict[str, int] = {}
    for cell in cells:
        m = (keys == cell).values
        n_c = int(m.sum())
        if n_c == 0:
            raise BoundError(f"released cell {cell!r} holds no private rows in the data passed; the "
                             f"private data does not match the release (wrong columns, bins or file)")
        p_true = float(pos[m].mean())
        # sensitivity 1/n_min (public), parallel across disjoint cells: each spends the full budget
        scale = ledger.spend(f"stage_c_rate[{cell}]", kind="rate", epsilon=epsilon_cert,
                             sensitivity=1.0 / n_min, composition="parallel", partition=cell,
                             group="stage_c_transmission_bound")
        p_hat[cell] = float(np.clip(_secure_laplace(p_true, scale), 0.0, 1.0))
        n_priv[cell] = n_c
    if own_ledger:
        ledger.seal()
    return lv, cols, cells, eff, p_hat, n_priv, ledger


def _score(condition: str, lv: dict, cols, cells, eff: Schema, p_hat: dict, n_priv: dict,
           df: pd.DataFrame, *, epsilon_cert: float, alpha: float, n_min: int) -> BoundResult:
    keys = _cell_keys(df, cols, eff)
    pos = _positives(eff, df)
    k = len(cells)
    halfwidth = (1.0 / n_min) / epsilon_cert * math.log(k / alpha)   # union bound over k cells
    out: list[CellBound] = []
    for cell in cells:
        m = (keys == cell).values
        rows = int(m.sum())
        if rows == 0:
            q, bound = None, 1.0                     # no evidence: the trivial bound
        else:
            q = float(pos[m].mean())
            bound = abs(p_hat[cell] - q) + halfwidth
        out.append(CellBound(cell=cell, n_private=n_priv[cell], p_hat=p_hat[cell], q_synth=q,
                             noise_halfwidth=halfwidth, bound=bound, synth_rows=rows))
    covered = [c for c in out if c.synth_rows > 0]
    thin = [c for c in out if 0 < c.synth_rows < MIN_SYNTH_ROWS_PER_CELL]
    return BoundResult(condition=condition, level=int(lv["level"]), columns=list(cols),
                       epsilon_cert=epsilon_cert, alpha=alpha, n_cells=k,
                       n_cells_covered=len(covered), n_cells_uncovered=k - len(covered),
                       n_cells_thin=len(thin), worst_case_bound=max(c.bound for c in out),
                       mean_bound=float(np.mean([c.bound for c in out])), cells=out)


def transmission_bound(schema: Schema, release: Release, private_df: pd.DataFrame,
                       synthetic_df: pd.DataFrame, *, epsilon_cert: float, alpha: float = 0.05,
                       level: int | None = None, ledger: PrivacyLedger | None = None) -> BoundResult:
    """Bound |p_c - q_c| simultaneously over the released cells, spending `epsilon_cert` once.

    Prefer `bound_with_controls`, which adds the ceiling and floor the verdict depends on. This
    function is the bare mechanism, for callers that bring their own controls or their own ledger.
    """
    lv, cols, cells, eff, p_hat, n_priv, _ = _noisy_private_rates(
        schema, release, private_df, epsilon_cert=epsilon_cert, level=level, ledger=ledger)
    return _score("synthetic", lv, cols, cells, eff, p_hat, n_priv, synthetic_df,
                  epsilon_cert=epsilon_cert, alpha=alpha, n_min=int(release.n_min))


def bound_with_controls(schema: Schema, release: Release, private_df: pd.DataFrame,
                        synthetic_df: pd.DataFrame, holdout_df: pd.DataFrame, *,
                        epsilon_cert: float, alpha: float = 0.05, tolerance: float = 0.15,
                        level: int | None = None, max_rows_per_person: int | None = None,
                        seed: int = 0) -> BoundReport:
    """Stage C with its floor and ceiling. Spends `epsilon_cert` once, on `private_df`.

    `holdout_df` must be real data that was NOT used to build the release, so that the ceiling is a
    genuine real sample rather than the training set compared against its own subset. `seed`
    controls only which hold-out rows form the ceiling and how the floor is permuted; it never
    touches the DP noise. `max_rows_per_person` is read from the release's accounting record when
    the release declared it; pass it here otherwise, because the report will not state a per-person
    figure it was not given.
    """
    lv, cols, cells, eff, p_hat, n_priv, ledger = _noisy_private_rates(
        schema, release, private_df, epsilon_cert=epsilon_cert, level=level, ledger=None)
    n_min = int(release.n_min)
    rng = np.random.RandomState(seed)
    n = min(len(synthetic_df), len(holdout_df))
    if n == 0:
        raise BoundError("the hold-out sample and the synthetic data must both be non-empty")
    ceiling_df = holdout_df.sample(n, random_state=seed)
    floor_df = ceiling_df.copy()
    floor_df[eff.target] = rng.permutation(floor_df[eff.target].values)

    kw = dict(epsilon_cert=epsilon_cert, alpha=alpha, n_min=n_min)
    synth = _score("synthetic", lv, cols, cells, eff, p_hat, n_priv, synthetic_df, **kw)
    ceil = _score("real-sample [CEILING]", lv, cols, cells, eff, p_hat, n_priv, ceiling_df, **kw)
    floor = _score("permuted-target [FLOOR]", lv, cols, cells, eff, p_hat, n_priv, floor_df, **kw)
    for r in (synth, ceil, floor):
        r.within_bound = bool(r.worst_case_bound <= tolerance)
    discriminating = bool(ceil.within_bound and not floor.within_bound)
    verdict = None if not discriminating else ("within bound" if synth.within_bound else "outside tolerance")

    # ── the DP claim block: the release's spend plus this one, per row and per person ──────
    audit = release.audit or {}
    k_person = max_rows_per_person if max_rows_per_person is not None else audit.get("max_rows_per_person")
    if not isinstance(k_person, (int, np.integer)) or int(k_person) < 1:
        raise BoundError("max_rows_per_person is undeclared: the release did not record it and none "
                         "was passed. This report will not state a per-person guarantee it was not "
                         "given; pass max_rows_per_person=1 if the data is one row per person.")
    k_person = int(k_person)
    eps_release = float(audit.get("epsilon_accounted", release.epsilon_total))
    total = eps_release + float(epsilon_cert)
    eps_person = total * k_person
    dp_claim = dict(DP_CLAIM)
    dp_claim.update({"epsilon_release": eps_release, "epsilon_transmission_bound": float(epsilon_cert),
                     "epsilon_total_per_row": total, "max_rows_per_person": k_person,
                     "epsilon_per_person": eps_person, "privacy_unit_declared": True,
                     "epsilon_per_person_vacuous": bool(eps_person > VACUOUS_EPSILON_PER_PERSON),
                     "per_person_claim_permitted": bool(eps_person <= VACUOUS_EPSILON_PER_PERSON),
                     "vacuous_threshold": VACUOUS_EPSILON_PER_PERSON})
    meta = {"level": int(lv["level"]), "columns": list(cols), "n_min": n_min, "alpha": alpha,
            "tolerance": tolerance, "seed_for_controls_only": seed,
            "n_synthetic_rows": int(len(synthetic_df)), "n_holdout_rows": int(len(holdout_df)),
            "noise_source": "cryptographically secure, unseeded: this bound is ONE draw and "
                            "re-running will not reproduce it",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return BoundReport(synthetic=synth, ceiling=ceil, floor=floor, tolerance=tolerance,
                       discriminating=discriminating, verdict=verdict, dp_claim=dp_claim,
                       accounting=ledger.audit_report(), meta=meta)
