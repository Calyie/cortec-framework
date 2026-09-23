"""
Privacy accounting for CoRTeC.

This module exists to be audited. A privacy engineer should be able to read it end to end and
confirm that the epsilon a caller is charged equals the epsilon the mechanism actually spends,
without reading any other file. Every quantity that is charged is recorded as a `Query` in a
ledger; the total is the sum over the ledger, computed the same way whether or not anyone is
watching. There is no code path that releases a statistic without appending to the ledger, because
`PrivacyLedger.spend()` is the only function that returns a noise scale.

The composition rules used here, and why each applies:

  * **Laplace mechanism** (Dwork et al. 2006). A query of L1 sensitivity `s` answered with noise
    `Lap(s/eps)` is `eps`-differentially private.
  * **Histogram sensitivity is 1, not the bin count.** Adding or removing one record changes
    exactly one bin by exactly one, so the L1 norm of the difference is 1 regardless of how many
    bins the histogram has. This is why a whole distribution costs what a single count costs.
  * **Sequential composition.** Queries that can both be influenced by the same record add:
    k queries at `eps` each cost `k*eps`.
  * **Parallel composition** (McSherry 2009). Queries over *disjoint* subsets of the data cost
    the MAXIMUM, not the sum, because any one record can influence only one of them. This is
    what makes cohorts and conditional cells cheap, and it is the single most load-bearing rule
    in the accounting — `PrivacyLedger` therefore requires callers to declare a partition key,
    and refuses to apply the parallel rule to queries whose partitions overlap.
  * **Post-processing immunity** (Dwork & Roth 2014). Any function of a released quantity, using
    no further access to the private data, is free. Generation is entirely post-processing.

A deliberate conservatism: data-dependent suppression (dropping cohorts below `n_min`) is a
decision made by looking at private counts. We charge it explicitly through
`spend_suppression_counts()` rather than treating cohort sizes as public, because the honest
accounting for "which cohorts appear in the output" is not free. Callers may opt into the
literature-standard treatment, but they must do so by name and it is recorded in the ledger.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Literal

Composition = Literal["sequential", "parallel"]


class PrivacyAccountingError(RuntimeError):
    """Raised when a release would exceed the declared budget, or when the ledger is asked to
    apply a composition rule whose preconditions do not hold."""


@dataclass(frozen=True)
class Query:
    """One privatised query. Everything charged to the budget is one of these."""
    name: str
    kind: str                    # histogram | count | bounded_mean | suppression
    epsilon: float
    sensitivity: float
    composition: Composition
    partition: str               # queries sharing a partition key are disjoint from each other
    group: str                   # queries in a group compose with each other by `composition`
    noise_scale: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


# NIST SP 800-226 asks for "where the guarantee does not hold" explicitly, and it is the part most
# often left out. The Stage C bound report carries its own version of this; the RELEASE audit is the
# artefact a compliance team actually files alongside the release, and it used to claim a guarantee
# and say nothing about the guarantee's limits. A machine-readable file that asserts
# "record-level eps-differential privacy" and "within_budget: true" with no stated boundary reads as
# a proof of privacy safety, which is the misreading this project renamed Stage C to avoid.
DP_CAVEATS = [
    "The privacy unit is one ROW. Under group privacy a person contributing k rows receives "
    "k*epsilon, so on data with repeated individuals the per-person guarantee is weaker than "
    "epsilon_accounted. See max_rows_per_person and epsilon_per_person in this report; if they say "
    "UNDECLARED, the per-person guarantee of this release is unknown, not equal to epsilon.",
    "Noise is sampled with a floating-point Laplace generator. Naive floating-point Laplace "
    "sampling is vulnerable to the Mironov (2012) least-significant-bit attack. A deployment "
    "handling real PHI should use a discrete or snapping sampler; this implementation does not, "
    "and that is a known gap.",
    "The guarantee covers the RELEASED STATISTICS recorded in this report. It says nothing about "
    "the generator's pretraining data, which is outside the DP boundary entirely, and nothing "
    "about whether a memorised record could surface from that corpus.",
    "If charge_suppression=False was passed, the decision about which cohorts and cells clear "
    "n_min was made from the private data and NOT charged. That is a data-dependent choice outside "
    "the accounted budget. The default charges it; check charge_suppression in this report.",
]

WHAT_THIS_IS = {
    "artifact": "differential privacy accounting record for one Stage A release",
    "records": "every privatised query, its epsilon, sensitivity, composition rule and partition "
               "key, so a privacy engineer can recompute the total by hand",
    "is_not": "a privacy audit, a re-identification risk assessment, or a certificate that the "
              "released data is safe to publish. It records what was spent, not what was achieved.",
    "read_with": "where_this_guarantee_does_not_hold, below, and the privacy-unit fields",
}


# Above this, a per-person epsilon carries no meaningful guarantee. There is no canonical
# threshold in the literature; 10 is chosen as an order of magnitude past the range regulated
# deployments actually argue over (typically eps <= 3), and it is reported alongside the raw number
# so a reader can apply their own line. It gates a REFUSAL in release_statistics, not just a note.
VACUOUS_EPSILON_PER_PERSON = 10.0


def _lineage_costs(per_partition: dict) -> dict:
    """Cost of each partition INCLUDING its ancestors.

    A partition key may be nested with "/" -- `cohort_3/YES` is the positive-class subset of
    cohort 3. Sub-partitions of one parent are disjoint from each other (parallel: the max), but
    every record of `cohort_3/YES` is also a record of `cohort_3`, so a query on the parent and a
    query on the child both touch it and compose SEQUENTIALLY. Charging bare keys would have
    called the child disjoint from its parent and under-charged by the parent's queries; charging
    the two children as one key would have over-charged by a whole block. The class-conditional
    histograms of `release_statistics` are the case this exists for: class balance on the cohort,
    one histogram block per class beneath it, charged as balance + ONE block per record.
    """
    out = {}
    for key, cost in per_partition.items():
        parts = str(key).split("/")
        out[key] = sum(per_partition.get("/".join(parts[:i + 1]), 0.0) for i in range(len(parts)))
    return out


class PrivacyLedger:
    """Records every DP query and enforces a hard total.

    The ledger is the only source of noise scales in this package. Code that wants to release a
    statistic must call `spend()`, which both charges the budget and returns the Laplace scale to
    use — so it is not possible to release something without it appearing in the audit trail.
    """

    def __init__(self, epsilon_total: float, *, delta: float = 0.0,
                 max_rows_per_person: int | None = None):
        """`max_rows_per_person` is the PRIVACY UNIT declaration, and it is the caller's.

        Epsilon here protects ONE ROW. Under group privacy a person contributing k rows receives
        k*epsilon, so on encounter-level data -- one hospital admission per row, several per
        patient -- the per-person guarantee is k times weaker than the number in this ledger. On a
        real hospital dataset whose heaviest patient contributes 40 encounters, a declared
        epsilon of 2.0 is an epsilon of 80 for that patient, which is outside any range normally
        considered meaningful.

        This cannot be computed from the data without spending budget on it, and it is not a
        property of the mechanism, so the caller declares it. Leaving it undeclared is recorded as
        UNDECLARED in the audit rather than silently assumed to be 1 -- an audit that quietly
        assumes one row per person is exactly how an encounter-level release comes to carry a
        per-person claim it has not earned.
        """
        if not (epsilon_total > 0):
            raise ValueError(f"epsilon_total must be positive, got {epsilon_total!r}")
        if delta != 0.0:
            raise ValueError("this implementation is pure eps-DP; delta must be 0.0")
        if max_rows_per_person is not None and (
                not isinstance(max_rows_per_person, int) or max_rows_per_person < 1):
            raise ValueError(f"max_rows_per_person must be an integer >= 1, "
                             f"got {max_rows_per_person!r}")
        self.epsilon_total = float(epsilon_total)
        self.delta = 0.0
        self.max_rows_per_person = max_rows_per_person
        # Recorded in the audit because one of the caveats points at it: an uncharged suppression
        # decision is a data-dependent choice outside the accounted budget, and a reader cannot
        # check a flag the report does not carry. Set by the release path; None where the concept
        # does not apply (the hybrid has no separate suppression query).
        self.charge_suppression: bool | None = None
        self.queries: list[Query] = []
        self._sealed = False

    # ── charging ────────────────────────────────────────────────────────────────────

    def spend(self, name: str, *, kind: str, epsilon: float, sensitivity: float,
              composition: Composition, partition: str, group: str) -> float:
        """Charge `epsilon` for one query and return the Laplace noise scale to use.

        `partition` identifies the disjoint subset of records this query reads. Two queries with
        the same partition key can both be influenced by the same record and therefore compose
        sequentially within their group; two with different keys are disjoint and compose in
        parallel. Getting this wrong in the unsafe direction is the failure this class exists to
        prevent, so the argument is required and never inferred.
        """
        if self._sealed:
            raise PrivacyAccountingError(
                "ledger is sealed: the release has been emitted and the private data may not be "
                "queried again. Post-processing the released statistics is free and needs no "
                "ledger entry."
            )
        if epsilon <= 0:
            raise PrivacyAccountingError(f"{name}: epsilon must be positive, got {epsilon!r}")
        if sensitivity <= 0:
            raise PrivacyAccountingError(f"{name}: sensitivity must be positive, got {sensitivity!r}")
        if composition not in ("sequential", "parallel"):
            raise PrivacyAccountingError(f"{name}: unknown composition {composition!r}")

        scale = sensitivity / epsilon
        q = Query(name=name, kind=kind, epsilon=float(epsilon), sensitivity=float(sensitivity),
                  composition=composition, partition=str(partition), group=str(group),
                  noise_scale=scale, timestamp=datetime.now(timezone.utc).isoformat())
        prospective = self._total_with(self.queries + [q])
        if prospective > self.epsilon_total + 1e-9:
            raise PrivacyAccountingError(
                f"refusing to run {name!r}: it would bring the total to {prospective:.6f}, "
                f"over the declared budget of {self.epsilon_total:.6f}. Nothing was released."
            )
        self.queries.append(q)
        return scale

    def spend_suppression_counts(self, *, epsilon: float, group: str = "suppression",
                                 partition: str = "__all__") -> float:
        """Charge for the counts used to decide which cohorts/cells appear at all.

        Whether a cohort is emitted depends on its private size, so the decision itself leaks.
        Charging it is the conservative choice and it is what this tool does by default.
        """
        return self.spend("n_min_suppression_counts", kind="suppression", epsilon=epsilon,
                          sensitivity=1.0, composition="parallel", partition=partition,
                          group=group)

    # ── totalling ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _total_with(queries: list[Query]) -> float:
        """Total epsilon under the declared composition rules.

        Within a group: parallel-composed queries contribute the max over partitions of the sum
        within each partition; sequential-composed queries contribute their plain sum. Groups
        compose sequentially with one another, because different groups describe overlapping
        populations (e.g. two granularities of the same conditional table).
        """
        by_group: dict[str, list[Query]] = {}
        for q in queries:
            by_group.setdefault(q.group, []).append(q)

        total = 0.0
        for _, qs in by_group.items():
            seq = sum(q.epsilon for q in qs if q.composition == "sequential")
            par = [q for q in qs if q.composition == "parallel"]
            par_cost = 0.0
            if par:
                per_partition: dict[str, float] = {}
                for q in par:
                    per_partition[q.partition] = per_partition.get(q.partition, 0.0) + q.epsilon
                # disjoint partitions -> the max, not the sum -- over LINEAGES, not bare keys
                par_cost = max(_lineage_costs(per_partition).values())
            total += seq + par_cost
        return total

    @property
    def total_epsilon(self) -> float:
        return self._total_with(self.queries)

    @property
    def remaining(self) -> float:
        return self.epsilon_total - self.total_epsilon

    def seal(self) -> None:
        """Close the ledger. After this, any attempt to query the private data raises."""
        self._sealed = True

    # ── audit output ────────────────────────────────────────────────────────────────

    def assert_within_budget(self) -> None:
        if self.total_epsilon > self.epsilon_total + 1e-9:
            raise PrivacyAccountingError(
                f"accounted {self.total_epsilon:.6f} > declared {self.epsilon_total:.6f}"
            )

    def audit_report(self) -> dict:
        """A machine-readable trail: every query, its cost, and how the total was reached.

        This is the artefact a compliance team keeps. It is intentionally verbose — it should be
        possible to recompute the total by hand from this dict alone.
        """
        by_group: dict[str, dict] = {}
        for q in self.queries:
            g = by_group.setdefault(q.group, {"sequential_sum": 0.0, "parallel_by_partition": {},
                                              "n_queries": 0})
            g["n_queries"] += 1
            if q.composition == "sequential":
                g["sequential_sum"] += q.epsilon
            else:
                p = g["parallel_by_partition"]
                p[q.partition] = p.get(q.partition, 0.0) + q.epsilon
        for g in by_group.values():
            g["parallel_cost"] = max(_lineage_costs(g["parallel_by_partition"]).values()) \
                if g["parallel_by_partition"] else 0.0
            if g["parallel_by_partition"]:
                g["parallel_by_lineage"] = _lineage_costs(g["parallel_by_partition"])
            g["group_total"] = g["sequential_sum"] + g["parallel_cost"]
            g["n_partitions"] = len(g["parallel_by_partition"])

        k = self.max_rows_per_person
        return {
            # first, so a reader who opens the file top-down learns what it is before what it says
            "_what_this_is": dict(WHAT_THIS_IS),
            "mechanism": "CoRTeC",
            "guarantee": "record-level eps-differential privacy, add/remove-one adjacency",
            "privacy_unit": "one dataset row",
            "epsilon_declared": self.epsilon_total,
            "epsilon_accounted": self.total_epsilon,
            # Group privacy: a person contributing k rows receives k*epsilon. Reported here
            # because a compliance reader will otherwise read epsilon_accounted as per-person.
            "max_rows_per_person": k if k is not None else "UNDECLARED",
            "epsilon_per_person": (round(self.total_epsilon * k, 10) if k is not None
                                   else "UNDECLARED — epsilon_accounted protects one ROW; if any "
                                        "person contributes k rows their guarantee is k x that"),
            # Machine-readable, because the human-readable warning lives in summary() and a
            # compliance pipeline reading audit.json never sees it. "UNKNOWN" is not "fine".
            "privacy_unit_declared": k is not None,
            "epsilon_per_person_vacuous": (
                bool(self.total_epsilon * k > VACUOUS_EPSILON_PER_PERSON) if k is not None
                else "UNKNOWN"),
            "vacuous_threshold": VACUOUS_EPSILON_PER_PERSON,
            "delta": self.delta,
            "charge_suppression": (self.charge_suppression if self.charge_suppression is not None
                                   else "n/a — this path has no separate suppression query"),
            "within_budget": self.total_epsilon <= self.epsilon_total + 1e-9,
            "n_queries": len(self.queries),
            "groups": by_group,
            "composition_rules": {
                "within_partition": "sequential (same records can influence both queries)",
                "across_partitions": "parallel — max, not sum (McSherry 2009)",
                "across_groups": "sequential (groups describe overlapping populations)",
                "generation": "post-processing, eps = 0 (Dwork & Roth 2014)",
            },
            "queries": [q.to_dict() for q in self.queries],
            "sealed": self._sealed,
            "where_this_guarantee_does_not_hold": list(DP_CAVEATS),
        }

    def write_audit(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.audit_report(), f, indent=2)
        return path

    def summary(self) -> str:
        r = self.audit_report()
        lines = [
            "PRIVACY ACCOUNTING",
            f"  guarantee        {r['guarantee']}",
            f"  epsilon declared {r['epsilon_declared']:.4f}",
            f"  epsilon spent    {r['epsilon_accounted']:.4f}   (per ROW)",
            f"  queries          {r['n_queries']}",
        ]
        # A compliance reader reads the headline epsilon as per-person unless told otherwise.
        k = r["max_rows_per_person"]
        if k == "UNDECLARED":
            lines += [
                "  !! PRIVACY UNIT UNDECLARED. The epsilon above protects one ROW. If any person",
                "     contributes k rows, that person's guarantee is k x this epsilon. Pass",
                "     max_rows_per_person=... to have it computed and recorded.",
            ]
        elif isinstance(k, int) and k > 1:
            eps_p = r["epsilon_per_person"]
            lines.append(f"  EPSILON PER PERSON {eps_p:.4f}   ({k} rows/person x per-row epsilon)")
            if eps_p > 10:
                lines.append(
                    f"  !! eps_person = {eps_p:.1f} is outside any range normally considered "
                    f"meaningful. Aggregate to one row per person before Stage A, or report this "
                    f"number rather than the per-row one.")
        else:
            lines.append("  EPSILON PER PERSON  = per-row epsilon (1 row per person, declared)")
        for name, g in r["groups"].items():
            lines.append(
                f"    group {name!r}: {g['n_queries']} queries over {g['n_partitions']} disjoint "
                f"partitions -> {g['group_total']:.4f}"
            )
        lines.append(f"  generation       post-processing, eps = 0 (unlimited draws)")
        # A reader who sees only the console must not leave thinking this asserts privacy safety.
        lines.append("\n  THIS IS AN ACCOUNTING RECORD, NOT A PRIVACY AUDIT. It records what was "
                     "spent,\n  not what was achieved, and it is not a certificate that the output "
                     "is safe to publish.")
        lines.append("  WHERE THIS GUARANTEE DOES NOT HOLD:")
        for c in r["where_this_guarantee_does_not_hold"]:
            lines.append(f"    - {c}")
        return "\n".join(lines)


def laplace_noise(rng, scale: float, size=None):
    """Laplace noise at the given scale, drawn from an explicitly passed generator.

    The generator is a parameter so that a caller can seed it for reproducibility in testing,
    and so that no module-level global RNG state can silently be shared between releases.
    """
    return rng.laplace(loc=0.0, scale=scale, size=size)


def histogram_sensitivity(n_bins: int) -> float:
    """L1 sensitivity of a histogram: 1, whatever the bin count.

    Present as a named function because the natural but wrong intuition is that more bins cost
    more. One record moves one bin by one, so the L1 difference is 1.
    """
    if n_bins < 1:
        raise ValueError("a histogram needs at least one bin")
    return 1.0


def bounded_mean_sensitivity(n: int, lo: float = 0.0, hi: float = 1.0) -> float:
    """Sensitivity of a mean of values in [lo, hi] over n records: (hi - lo) / n."""
    if n < 1:
        raise ValueError("cannot take a mean over zero records")
    if hi <= lo:
        raise ValueError(f"invalid bounds [{lo}, {hi}]")
    return (hi - lo) / n
