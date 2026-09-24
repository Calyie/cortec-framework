"""
Stage A — the differentially private release. This is the only code in the package that reads
private data.

The budget allocation is the substance of the method, so it is worth stating plainly what it does
and why each choice is cheaper than the obvious alternative:

  * **Cohorts come from a public rule, so forming them is free.** The alternative — clustering the
    private data to find cohorts — spends budget and, in our measurements, produced degenerate
    cohorts of 3 and 8 records.
  * **Histograms, not moments.** A histogram has L1 sensitivity 1 whatever its bin count, so a
    whole distribution costs what a single count costs and *half* what a mean-and-standard-
    deviation pair costs. Under the moment formulation the standard-deviation query has
    sensitivity (hi-lo)/(2*sqrt(n)), which at realistic budgets produced released standard
    deviations larger than the attribute's entire range at every cohort size tested. Moments are
    recovered from the released histogram afterwards, for free.
  * **Cohorts are disjoint, so they compose in parallel.** The cost across cohorts is the maximum,
    not the sum — independent of how many cohorts there are.
  * **Each conditional level is a partition, so it also composes in parallel.** A level costs one
    query's worth of epsilon however many cells it contains; accuracy is governed by cell
    *support*, not cell count. This is why conditional structure is cheap to release, which is the
    observation the whole method rests on.

Levels compose sequentially with each other, because two granularities describe the same people.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import math

import numpy as np
import pandas as pd

from .accounting import (VACUOUS_EPSILON_PER_PERSON, PrivacyLedger,
                         histogram_sensitivity, laplace_noise)
from .schema import Band, Schema
from . import report

DEFAULT_N_MIN = 150


class ReleaseError(RuntimeError):
    """Stage A could not release as asked; the message names the setting or the data to change."""


@dataclass
class Release:
    """The complete set of released statistics. Contains no private data.

    Everything downstream — prompt construction, generation, evaluation — reads only this object,
    which is what makes generation post-processing.
    """
    schema_name: str
    epsilon_total: float
    n_min: int
    cohorts: list[dict] = field(default_factory=list)
    conditional_levels: list[dict] = field(default_factory=list)
    # The coarsening actually applied to the cell keys. Autoconfig derives it inside
    # `release_statistics`, which applies it to a LOCAL copy of the schema -- so the caller's
    # Schema never learns about it, and anything downstream comparing a cell key against the
    # schema's raw categories is comparing two different alphabets. It travels on the release
    # because it is part of what the release encodes.
    coarsen: dict = field(default_factory=dict)
    # The cohort rule the release was built under, in serialisable form:
    # [[column, [[band name, lo, hi], ...]], ...]. Autoconfig derives the stratification inside
    # `release_statistics` and applies it to a LOCAL copy of the schema, so a caller's Schema does
    # not know which cohort a row belongs to; anything that must map rows back to released cohorts
    # (the selection step) reads it from here. Empty means the schema's own rule (or one cohort).
    stratify: list = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump({"schema_name": self.schema_name, "epsilon_total": self.epsilon_total,
                       "n_min": self.n_min, "cohorts": self.cohorts,
                       "conditional_levels": self.conditional_levels,
                       "coarsen": self.coarsen, "stratify": self.stratify,
                       "audit": self.audit, "created": self.created}, f, indent=2, default=str)
        return path

    def effective_schema(self, schema: Schema) -> Schema:
        """The caller's schema with the cohort rule this release was built under."""
        if not self.stratify:
            return schema
        from dataclasses import replace as _replace
        bands = [(col, [Band(str(n), float(lo), float(hi)) for n, lo, hi in bl]) for col, bl in self.stratify]
        return _replace(schema, stratify=bands)

    @classmethod
    def from_json(cls, path: str) -> "Release":
        d = json.load(open(path))
        return cls(**d)

    @property
    def n_cohorts(self) -> int:
        return len(self.cohorts)


def _cohort_keys(schema: Schema, df: pd.DataFrame) -> pd.Series:
    """Assign each record to a cohort using the PUBLIC stratification rule only.

    The assignment is a function of the record's own attributes and publicly declared band edges,
    so it reads nothing about any other record and costs no budget.
    """
    if not schema.stratify:
        return pd.Series(["all"] * len(df), index=df.index)
    parts = []
    for col, bands in schema.stratify:
        if col in schema.numerical:
            v = pd.to_numeric(df[col], errors="coerce")
            lab = pd.Series(["oob"] * len(df), index=df.index, dtype=object)
            for b in bands:
                lab[b.contains(v)] = b.name
            # bands are half-open [lo, hi); the top band is closed at the declared maximum, so a
            # record sitting exactly on the domain's upper bound is a member of the top band and
            # not of an "oob" cohort that no generator can produce rows for (Adult: 460 records
            # at education_num = 16 formed such a cohort, and selection then refused every pool)
            top = max(bands, key=lambda b: b.hi)
            lab[(v == top.hi).values] = top.name
        else:
            lab = df[col].astype(str).str.strip()
        parts.append(lab.astype(str))
    return pd.Series([" & ".join(t) for t in zip(*parts)], index=df.index)


def _cell_keys(df: pd.DataFrame, cols: tuple[str, ...], schema: Schema) -> pd.Series:
    """Cells of one conditional level. These partition the data, hence parallel composition."""
    # The coarsest level conditions on nothing — one cell holding every record. It is a legitimate
    # level (it releases the global rate) and `zip(*[])` yields nothing, so it needs its own case.
    if not cols:
        return pd.Series(["*"] * len(df), index=df.index)
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
            # Apply the declared coarsening. The map is over DECLARED levels, which are public,
            # so this reads no private value -- and NOT applying it was a real defect: autoconfig
            # counted cells as if a 26-level column had been grouped into 4, then keyed cells on
            # the raw level and built 104 where it had budgeted for 16.
            cmap = (schema.coarsen or {}).get(c)
            if cmap:
                v = v.map(lambda x: cmap.get(x, x))
            parts.append(c + "=" + v)
    return pd.Series([" & ".join(t) for t in zip(*parts)], index=df.index)


def _with_release_warnings(audit: dict, *, seed_warning: str | None = None,
                           person_warning: str | None = None) -> dict:
    """Attach release-time warnings to the audit trail so they survive into the saved release.

    Printing alone is not enough: the release is written to JSON and read later by someone who
    never saw the console. A guarantee that was voided at generation time -- or one that protects
    a row while the reader assumes it protects a person -- has to travel with the artefact that
    claims it.
    """
    if not (seed_warning or person_warning):
        return audit
    audit = dict(audit)
    if seed_warning:
        audit["guarantee_void"] = True
        audit.setdefault("warnings", []).append(seed_warning)
    if person_warning:
        audit.setdefault("warnings", []).append(person_warning)
    return audit


def _numeric_entry(props, edges, bounds) -> dict:
    """A released numerical histogram with its moments derived from it (free post-processing)."""
    props = np.asarray(props, dtype=float)
    centres = (edges[:-1] + edges[1:]) / 2.0
    mean = float((props * centres).sum())
    var = float((props * (centres - mean) ** 2).sum())
    return {"bin_edges": [float(e) for e in edges],
            "proportions": [round(float(p), 5) for p in props],
            # derived from the released histogram — free post-processing, not extra queries
            "mean": round(mean, 3), "std": round(float(np.sqrt(max(var, 0.0))), 3),
            "bounds": list(bounds)}


def release_statistics(schema: Schema, df: pd.DataFrame, *, epsilon_total: float = 2.0,
                       conditional_fraction: float = 0.2, n_min: int = DEFAULT_N_MIN,
                       charge_suppression: bool = True, seed: int | None = None,
                       max_rows_per_person: int | None = None,
                       acknowledge_vacuous_privacy_unit: bool = False,
                       autoconfig: bool = True, n_records: int = 1000,
                       class_conditional: bool = True,
                       ) -> Release:
    """Run Stage A and return a `Release`. This is the only function that touches `df`.

    `n_min` suppresses cohorts and cells with too little support. It defaults to a value that
    keeps released rates meaningful and is validated below rather than being silently accepted:
    a small n_min releases rates whose Laplace noise exceeds the signal, which produces a release
    that looks populated and carries noise.

    **The conditional hierarchy is derived by default.** Leave `schema.conditional` empty and the
    hierarchy is chosen for you (see `cortec.autoconfig`), spending a declared slice of
    `epsilon_total` on the selection and charging it to the same ledger. Declare `schema.conditional`
    yourself and that is used instead, untouched — hand-tuning stays available for anyone who knows
    their data, it simply is not what happens when you say nothing.

    Autoconfig is the default because the failure mode of a wrong hierarchy is silent. Nothing in
    the output announces that the table conditioned on the least informative columns available; it
    just quietly carries less structure, and in our own study that cost real downstream utility.

    **Class-conditional histograms are released by default** (`class_conditional=True`). Where both
    classes of a cohort clear `n_min`, the cohort's feature histograms are released once per
    (cohort, class) instead of pooled: the two class blocks are disjoint, so they compose in
    parallel at the SAME epsilon per query the pooled block spent, and the pooled histogram is their
    noisy mixture (post-processing, free). The generator then receives what positives and negatives
    each look like for every column, instead of inventing the feature->target structure the
    conditional hierarchy does not name -- on finance those invented columns measurably degraded
    the downstream model. Charged as one chain per record (class balance + one block) through
    nested ledger partitions (`cohort/class`).
    """
    if n_min < 50:
        raise ReleaseError(
            f"n_min={n_min} is below the safe floor of 50. A conditional rate over fewer than "
            f"~50 records is dominated by its own noise: at epsilon_level=1.0 the Laplace scale "
            f"is 1/(n*eps), so a 30-record cell carries roughly +/-0.033 of noise on a rate in "
            f"[0,1] before any sampling error. Raise n_min, or coarsen the conditional levels so "
            f"cells have support."
        )
    if not (0.0 < conditional_fraction < 1.0):
        raise ReleaseError("conditional_fraction must be strictly between 0 and 1")

    # THE PRIVACY UNIT. epsilon here protects one ROW. Where the caller has declared that a person
    # contributes k rows the per-person guarantee is known before any budget is spent, so a vacuous
    # one is refused rather than warned about. Undeclared is a different case: k cannot be computed
    # without spending budget, so it is a banner and a recorded warning, not a refusal -- defaulting
    # it to 1 would be the silent per-person claim this gate exists to prevent.
    _person_warning = None
    if max_rows_per_person is None:
        _person_warning = (
            "PRIVACY UNIT UNDECLARED. The epsilon this release reports protects ONE ROW. If any "
            "person contributes k rows, their guarantee is k x that epsilon -- on encounter-level "
            "clinical data k is routinely 10-40, which makes the per-person guarantee vacuous. "
            "Pass max_rows_per_person=... (1 if one row per person) to have it computed, checked "
            "and recorded.")
        report.emit("\n" + report.warn(_person_warning) + "\n")
    else:
        _eps_person = float(epsilon_total) * int(max_rows_per_person)
        if _eps_person > VACUOUS_EPSILON_PER_PERSON and not acknowledge_vacuous_privacy_unit:
            raise ReleaseError(
                f"REFUSED: max_rows_per_person={max_rows_per_person} at epsilon_total="
                f"{epsilon_total} gives epsilon_per_person={_eps_person:.1f}, past the "
                f"{VACUOUS_EPSILON_PER_PERSON:.0f} beyond which a per-person guarantee carries no "
                f"meaning. CoRTeC's privacy unit is one row, so unaggregated longitudinal data "
                f"degrades the guarantee by the maximum contribution count. Either aggregate to "
                f"one row per person before Stage A (which makes k=1 and the guarantee exact), or "
                f"lower epsilon_total, or -- if you intend to publish the row-level number with "
                f"its group-privacy factor stated, as this paper's Diabetes 130 study does -- pass "
                f"acknowledge_vacuous_privacy_unit=True, which records the acknowledgement in the "
                f"audit trail.")
        if _eps_person > VACUOUS_EPSILON_PER_PERSON:
            _person_warning = (
                f"VACUOUS PER-PERSON GUARANTEE ACKNOWLEDGED: epsilon_per_person="
                f"{_eps_person:.1f} ({max_rows_per_person} rows/person x epsilon_total="
                f"{epsilon_total}). This release carries a ROW-level guarantee only. Any claim "
                f"made from it must state the group-privacy factor.")
            report.emit("\n" + report.warn(_person_warning) + "\n")

    schema.validate(df, strict=True)          # before a single unit of budget is spent
    data = schema.coerce(df)

    rng = np.random.default_rng(seed)
    # A SEEDED release is byte-reproducible, which means its noise is predictable and the epsilon
    # it reports is not a guarantee: anyone holding the release and the seed can subtract the noise
    # and recover the private counts exactly. The default (seed=None) draws from OS entropy and is
    # correct; seeding is offered for tests only. It used to pass silently -- the docstring on the
    # noise helper even invited callers to "seed it for reproducibility" -- so the one path that
    # voids the guarantee was also the most inviting one. Say so, loudly and in the audit trail.
    _seed_warning = None
    if seed is not None:
        _seed_warning = (
            f"RELEASE WAS SEEDED (seed={seed}). The noise is reproducible, so this release does "
            f"NOT carry the epsilon={epsilon_total} guarantee it reports: anyone with this file "
            f"and the seed can invert the noise and recover the private values. Use seed=None for "
            f"anything but a test.")
        report.emit("\n" + report.warn(_seed_warning) + "\n")
    ledger = PrivacyLedger(epsilon_total, max_rows_per_person=max_rows_per_person)
    ledger.charge_suppression = bool(charge_suppression)

    derived = None
    if autoconfig and not schema.conditional:
        from .autoconfig import derive
        # Pass the ledger so the selection is charged AND scaled in one place. It used to be
        # scaled inside derive() and charged here, which agreed only by convention.
        derived = derive(schema, data, epsilon_total=epsilon_total, n_records=n_records,
                         n_min=n_min, seed=seed, ledger=ledger)
        # (The selection query is charged inside derive(), which also supplies its noise scale.)
        # Apply BOTH the derived conditional hierarchy and the derived stratification. Applying
        # only the hierarchy left every autoconfigured release with a single global cohort.
        # Carry the coarsening onto the schema so the cells BUILT match the cells COUNTED.
        schema = replace(schema, conditional=derived.levels,
                         stratify=derived.stratify or schema.stratify,
                         coarsen=derived.coarsen or schema.coarsen)
        for line in derived.describe().splitlines():
            report.emit(report.note(line))
    elif schema.conditional:
        report.emit(report.note("using the conditional hierarchy declared in the schema (autoconfig skipped)"))

    # ── budget split ────────────────────────────────────────────────────────────────
    # Whatever autoconfig spent selecting the hierarchy is gone: the release itself must live on
    # what remains, or the ledger (correctly) refuses the last query and nothing is released.
    eps_available = epsilon_total - (derived.epsilon_selection if derived else 0.0)
    eps_cond = conditional_fraction * eps_available
    eps_marg = eps_available - eps_cond
    # Cohort sizes and per-cell supports are PUBLISHED in the release and used to allocate rows.
    # They were being emitted as exact counts of private records: unnoised, and never entered into
    # the ledger, so `verify_privacy_accounting` could not see them either — it sums what it is
    # told about. Exact counts break the epsilon guarantee outright. Charge and noise them.
    # Cohorts partition the data and so do the cells within one level, so this is parallel
    # composition: one epsilon for the whole group regardless of how many counts are published.
    # Split across two GROUPS, not one. Cohort-size queries and cell-support queries each
    # partition the data internally (parallel within a group), but they overlap EACH OTHER — a
    # record sits in one cohort and one cell — so the two groups compose sequentially. Filing both
    # in one group would let the parallel max() absorb one of them and under-charge, which is the
    # same mistake the suppression note above records.
    eps_counts = 0.05 * eps_available
    eps_counts_cohorts = 0.5 * eps_counts
    eps_counts_cells = eps_counts - eps_counts_cohorts
    eps_marg -= eps_counts

    if charge_suppression:
        # Charge the suppression decision out of the marginal share rather than silently
        # treating private counts as public.
        eps_supp = min(0.05 * eps_available, 0.25 * eps_marg)
        eps_marg -= eps_supp
        # Its own group, NOT "marginals". The suppression counts read every record, so this query
        # is not disjoint from any cohort's queries and must compose sequentially with them.
        # Filing it under "marginals" put it in the parallel max() alongside the per-cohort
        # partitions, where the larger cohort cost absorbed it and the release under-charged by
        # exactly eps_supp — an under-charge is a false privacy claim, so the grouping matters.
        ledger.spend_suppression_counts(epsilon=eps_supp, group="suppression")

    keys = _cohort_keys(schema, data)
    n_queries_per_cohort = len(schema.numerical) + len(schema.categorical) + 1  # +1 class balance
    eps_q = eps_marg / n_queries_per_cohort

    # ── per-cohort marginals ────────────────────────────────────────────────────────
    cohorts: list[dict] = []
    for cid, (name, part) in enumerate(sorted(data.groupby(keys), key=lambda kv: str(kv[0]))):
        if len(part) < n_min:
            continue
        _cs_scale = ledger.spend(f"count[cohort][{name}]", kind="count",
                                 epsilon=eps_counts_cohorts, sensitivity=1.0,
                                 composition="parallel", partition=str(name),
                                 group="published_counts_cohorts")
        # clamped at n_min: the cohort exists in the release only because it holds >= n_min records,
        # so a lower published size is impossible under the release's own rule (post-processing);
        # unclamped, a tight budget zeroed a 469-record cohort and the generator allocated it 1 row
        _cs = int(max(n_min, round(len(part) + laplace_noise(rng, _cs_scale))))
        entry = {"cohort_id": cid, "cohort_name": str(name), "cohort_size": _cs,
                 "numerical": {}, "categorical": {}, "class_balance": {}}

        def _hist_block(rows: pd.DataFrame, partition: str) -> tuple[dict, dict]:
            """One noised histogram per column over `rows`, charged to `partition`."""
            num_b, cat_b = {}, {}
            for col in schema.numerical:
                edges = schema.bin_edges(col)
                counts, _ = np.histogram(pd.to_numeric(rows[col], errors="coerce").dropna(), bins=edges)
                scale = ledger.spend(f"hist[{partition}][{col}]", kind="histogram", epsilon=eps_q,
                                     sensitivity=histogram_sensitivity(len(counts)),
                                     composition="parallel", partition=partition, group="marginals")
                noisy = np.maximum(counts + laplace_noise(rng, scale, size=len(counts)), 0.0)
                total = noisy.sum()
                props = (noisy / total) if total > 0 else np.full(len(noisy), 1.0 / len(noisy))
                num_b[col] = _numeric_entry(props, edges, schema.numerical[col])
            for col, allowed in schema.categorical.items():
                counts = rows[col].astype(str).str.strip().value_counts()
                vec = np.array([float(counts.get(str(v), 0)) for v in allowed])
                scale = ledger.spend(f"hist[{partition}][{col}]", kind="histogram", epsilon=eps_q,
                                     sensitivity=histogram_sensitivity(len(vec)),
                                     composition="parallel", partition=partition, group="marginals")
                noisy = np.maximum(vec + laplace_noise(rng, scale, size=len(vec)), 0.0)
                tot = noisy.sum()
                props = (noisy / tot) if tot > 0 else np.full(len(noisy), 1.0 / len(noisy))
                # every released category is kept: truncating this to the top few accounted for
                # essentially all of the marginal-fidelity gap in our measurements, at no privacy saving
                cat_b[col] = {str(v): round(float(p), 5) for v, p in zip(allowed, props)}
            return num_b, cat_b

        tgt = part[schema.target].astype(str).str.strip()
        is_pos = (tgt == schema.positive).values
        vec = np.array([float(is_pos.sum()), float((~is_pos).sum())])
        scale = ledger.spend(f"class_balance[{name}]", kind="histogram", epsilon=eps_q,
                             sensitivity=histogram_sensitivity(2), composition="parallel",
                             partition=str(name), group="marginals")
        noisy = np.maximum(vec + laplace_noise(rng, scale, size=2), 0.0)
        tot = noisy.sum() or 1.0
        entry["class_balance"] = {schema.positive: round(float(noisy[0] / tot), 5),
                                  schema.negative: round(float(noisy[1] / tot), 5)}

        if class_conditional and is_pos.sum() >= n_min and (~is_pos).sum() >= n_min:
            # CLASS-CONDITIONAL HISTOGRAMS: one block per (cohort, class), disjoint partitions
            # beneath the cohort, so a record is charged the class balance plus ONE block --
            # exactly what the pooled block cost. The pooled histogram is derived as the
            # class-balance-weighted mixture: post-processing, no query.
            p_num, p_cat = _hist_block(part[is_pos], f"{name}/{schema.positive}")
            q_num, q_cat = _hist_block(part[~is_pos], f"{name}/{schema.negative}")
            w = float(noisy[0] / tot)
            entry["by_class"] = {schema.positive: {"numerical": p_num, "categorical": p_cat},
                                 schema.negative: {"numerical": q_num, "categorical": q_cat}}
            for col in schema.numerical:
                h = w * np.asarray(p_num[col]["proportions"]) + (1.0 - w) * np.asarray(q_num[col]["proportions"])
                entry["numerical"][col] = _numeric_entry(h / h.sum() if h.sum() > 0 else h,
                                                         schema.bin_edges(col), schema.numerical[col])
            for col, allowed in schema.categorical.items():
                entry["categorical"][col] = {str(v): round(w * p_cat[col][str(v)] + (1.0 - w) * q_cat[col][str(v)], 5)
                                             for v in allowed}
        else:
            entry["numerical"], entry["categorical"] = _hist_block(part, str(name))
        cohorts.append(entry)

    if not cohorts:
        raise ReleaseError(
            f"no cohort reached n_min={n_min} (dataset has {len(data)} rows across "
            f"{keys.nunique()} cohorts). Nothing was released and the budget was not spent on "
            f"anything usable. Coarsen the stratification or lower n_min."
        )

    # ── conditional target table ────────────────────────────────────────────────────
    levels: list[dict] = []
    if schema.conditional:
        # Split eps_cond only across levels that will ACTUALLY release a cell. Splitting across all
        # declared levels wasted the share of every level whose cells all fall below n_min — 20-40%
        # of the conditional budget on 6 of 11 datasets (shortfalls 0.199-0.350), making those
        # releases needlessly noisy, worst on small data.
        #
        # Privacy argument: which cells clear n_min is exactly the data-dependent decision the
        # SUPPRESSION query above already charges for (`spend_suppression_counts`, "the counts used
        # to decide which cohorts/cells appear at all"). Allocating budget from that same decision
        # reveals nothing further, so this is post-processing of an already-paid query.
        _viable = []
        for _li, _cols in enumerate(schema.conditional):
            _ck = _cell_keys(data, _cols, schema)
            if any(len(_p) >= n_min for _, _p in data.groupby(_ck)):
                _viable.append(_li)
        _n_viable = max(len(_viable), 1)
        eps_level = eps_cond / _n_viable                  # levels overlap -> sequential
        for li, cols in enumerate(schema.conditional):
            cells = {}
            ck = _cell_keys(data, cols, schema)
            for cell, part in data.groupby(ck):
                if len(part) < n_min:
                    continue
                # The rate is NOT noised as a bounded mean. That mechanism's scale,
                # 1/(|cell|·eps), depends on |cell| -- which under add/remove-one adjacency is
                # itself private and differs between neighbouring datasets. Two Laplace densities
                # with different scales have an unbounded ratio in one tail, so it is not pure
                # eps-DP (technical report, section 4.3). Release the POSITIVE COUNT instead: a counting query of
                # sensitivity exactly 1 at a scale that depends on nothing private. The cell
                # size is the already-charged noised support below. The published rate is their
                # ratio -- post-processing of two DP-released quantities -- with the denominator
                # floored at the PUBLIC n_min so a small noised count cannot blow the ratio up.
                _sup_scale = ledger.spend(
                    f"count[cell][L{li}][{cell}]", kind="count",
                    epsilon=eps_counts_cells / _n_viable,
                    sensitivity=1.0, composition="parallel", partition=str(cell),
                    group=f"published_counts_cells_L{li}")
                # clamped at n_min, as the cohort size is: the cell is in the release only because
                # it holds >= n_min records, so a lower published support is impossible under the
                # release's own rule (post-processing). Unclamped, a noised support of 0 reached a
                # stored NHANES release, and `generate_by_cell` allocates rows in proportion to
                # support, so that cell would have received no rows at all.
                _sup = int(max(n_min, round(len(part) + laplace_noise(rng, _sup_scale))))
                pos_true = float((part[schema.target].astype(str).str.strip()
                                  == schema.positive).sum())
                scale = ledger.spend(f"cond[L{li}][{cell}]", kind="count",
                                     epsilon=eps_level, sensitivity=1.0,
                                     composition="parallel", partition=str(cell),
                                     group=f"conditional_L{li}")
                pos_noisy = max(0.0, pos_true + laplace_noise(rng, scale))
                noisy = float(np.clip(pos_noisy / float(max(_sup, n_min)), 0.0, 1.0))
                cells[str(cell)] = {"rate": round(noisy, 5), "support": _sup}
            if cells:
                levels.append({"level": li, "columns": list(cols), "cells": cells})

    # A level whose cells all fall below n_min releases NOTHING, and its share of eps_cond is then
    # never spent. With autoconfig deriving k columns the budget is split k+1 ways up front, so on
    # a small dataset most of it evaporates: heart_cleveland declared 2.0 and accounted 1.63,
    # german_credit 1.71 — a noisier release than the caller paid for, worst exactly where noise
    # hurts most. The tool's exact-spend test never saw this because it uses a DECLARED schema,
    # which skips autoconfig entirely. Surface it rather than let the audit be read as fine print.
    _unspent = eps_cond - sum(
        v["group_total"] for k, v in ledger.audit_report()["groups"].items()
        if k.startswith("conditional_L"))
    if _unspent > 0.02 * epsilon_total:
        report.emit(report.warn(f"{_unspent:.3f} of the {eps_cond:.3f} conditional budget was NOT spent: "
              f"{len(schema.conditional)} levels were budgeted for but only {len(levels)} released "
              f"cells at n_min={n_min}. The release is noisier than epsilon={epsilon_total} allows. "
              f"Use fewer conditional levels, lower n_min, or supply more records."))

    # Is the conditional table SIGNAL or NOISE? The Laplace sd on a cell's rate is
    # sqrt(2) * (1/n_cell) / eps_level. If that is comparable to the spread of the released rates
    # themselves, the table carries more noise than structure and no generator can transmit real
    # conditional signal from it — measured ratios: adult 0.04, nhanes 0.14, heart_cleveland 0.27,
    # cervical_cancer 1.40, german_credit 1.78. Computed from RELEASED quantities only, so this
    # costs nothing and is available at release time rather than after a generation run.
    for _lv in levels:
        _cells = _lv["cells"]
        if len(_cells) < 2:
            continue
        _rates = [c["rate"] for c in _cells.values()]
        _sup = [max(c["support"], 1) for c in _cells.values()]
        _sd = math.sqrt(2.0) * (1.0 / float(np.median(_sup))) / max(eps_level, 1e-12)
        _spread = max(_rates) - min(_rates)
        if _spread > 0 and _sd / _spread >= 1.0:
            report.emit(report.warn(f"level {_lv['level']}: noise on the released rates (sd~{_sd:.3f}) is at or "
                  f"above their entire spread ({_spread:.3f}). This table is noise-dominated and "
                  f"cannot carry conditional structure — raise epsilon, coarsen to fewer cells, or "
                  f"accept that conditioning will not help on this dataset."))

    if not levels:
        raise ReleaseError(
            "the conditional target table is empty: no cell reached n_min. This table is the "
            "channel that carries your data's conditional structure to the generator, so an empty "
            "one means the output would be unconditioned. Either let autoconfig derive the "
            "hierarchy (the default: leave `Schema.conditional` empty and do not pass "
            "autoconfig=False), coarsen the levels you declared, or lower n_min."
        )

    ledger.assert_within_budget()
    ledger.seal()          # the private data may not be queried again

    return Release(schema_name=schema.name, epsilon_total=epsilon_total, n_min=n_min,
                   coarsen=dict(schema.coarsen or {}),
                   stratify=[[col, [[b.name, float(b.lo), float(b.hi)] for b in bands]]
                             for col, bands in (schema.stratify or [])],
                   cohorts=cohorts, conditional_levels=levels,
                   audit=_with_release_warnings(ledger.audit_report(),
                                                seed_warning=_seed_warning,
                                                person_warning=_person_warning))
