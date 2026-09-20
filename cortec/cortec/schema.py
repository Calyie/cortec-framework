"""
Schema declaration and input validation.

Two ideas carry the privacy argument and are enforced here rather than left to a convention:

1. **Everything in a `Schema` is public.** Column names, domain bounds, bin edges, the
   stratification rule and the conditional cell definitions are all assumed public knowledge, as
   in every marginal-based DP synthesiser. Nothing in this file may be derived from the private
   data, so `Schema.from_dataframe()` deliberately does NOT exist — a schema inferred from private
   values would leak through its own bounds. Bounds must be declared by the user from what they
   already know about their domain.

2. **Validation happens before any budget is spent.** A dataset that will fail halfway through a
   release wastes real privacy budget, which cannot be refunded. `validate()` is called before the
   first query and reports every problem at once rather than failing on the first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """The schema is not usable as declared."""


class DataValidationError(ValueError):
    """The data does not match the declared schema. Raised before any budget is spent."""


@dataclass(frozen=True)
class Band:
    """A named, public band over a numeric column, used to build cohorts."""
    name: str
    lo: float
    hi: float           # half-open [lo, hi)

    def contains(self, v):
        return (v >= self.lo) & (v < self.hi)


@dataclass
class Schema:
    """A public description of a tabular dataset.

    Parameters
    ----------
    name              short identifier used in output paths and the audit trail
    numerical         {column: (lo, hi)} public domain bounds
    categorical       {column: [allowed values]} public value sets
    target            name of the binary target column
    positive, negative  the two target label values
    bins              {column: [edges]} public histogram edges for numeric columns
    stratify          list of (column, [Band, ...]) forming the cohort grid; the cross product of
                      these bands is the partition. Cohorts are a function of public bounds only.
    conditional       list of column tuples defining conditional-table levels, coarse to fine
    coarsen           {column: {declared level: group label}} grouping a high-cardinality
                      categorical into a few classes for conditioning. The map is over DECLARED
                      levels, which are public, so applying it reads no private value. Autoconfig
                      derives one; a caller may also supply it.
    """
    name: str
    numerical: dict[str, tuple[float, float]]
    categorical: dict[str, list[str]]
    target: str
    positive: str
    negative: str
    bins: dict[str, list[float]] = field(default_factory=dict)
    stratify: list[tuple[str, list[Band]]] = field(default_factory=list)
    conditional: list[tuple[str, ...]] = field(default_factory=list)
    # Autoconfig USED to derive this, count cells with it, and then never apply it: cells were
    # keyed on the raw category. On a 26-level discharge column that reported 16 cells at the
    # finest level and produced 104 -- 3.2 records per cell against its own floor of 8, so the
    # depth chooser selected a depth it would have refused had it counted what it built.
    coarsen: dict[str, dict[str, str]] = field(default_factory=dict)

    # ── construction checks ─────────────────────────────────────────────────────────

    def __post_init__(self):
        if self.target in self.numerical or self.target in self.categorical:
            raise SchemaError(
                f"target {self.target!r} must not also be declared as a feature column"
            )
        if self.positive == self.negative:
            raise SchemaError("positive and negative class labels must differ")
        for c, (lo, hi) in self.numerical.items():
            if not (hi > lo):
                raise SchemaError(f"{c}: invalid bounds [{lo}, {hi}]")
        for c, vals in self.categorical.items():
            if not vals:
                raise SchemaError(f"{c}: categorical column needs at least one allowed value")
            if len(set(vals)) != len(vals):
                raise SchemaError(f"{c}: duplicate values in the allowed set")
        for c in self.numerical:
            edges = self.bins.get(c)
            if edges is None:
                continue
            if len(edges) < 2:
                raise SchemaError(f"{c}: need at least two bin edges")
            if list(edges) != sorted(edges):
                raise SchemaError(f"{c}: bin edges must be ascending")
        for col, bands in self.stratify:
            if col not in self.numerical and col not in self.categorical:
                raise SchemaError(f"stratify column {col!r} is not declared in the schema")
            if col in self.numerical and bands:
                # the bands must tile the declared domain: a gap would send records to an "oob"
                # cohort that is released like any other but that no generator can fill, and the
                # selection step would then refuse every pool for that cohort
                lo, hi = self.numerical[col]
                bs = sorted(bands, key=lambda b: b.lo)
                if bs[0].lo > lo or bs[-1].hi < hi or any(a.hi != b.lo for a, b in zip(bs, bs[1:])):
                    raise SchemaError(
                        f"stratify bands for {col!r} must tile its declared domain [{lo}, {hi}] with "
                        f"no gap; got {[(b.name, b.lo, b.hi) for b in bs]}")
        for level in self.conditional:
            for col in level:
                if col not in self.numerical and col not in self.categorical:
                    raise SchemaError(f"conditional column {col!r} is not declared in the schema")
        # A coarsening that names an undeclared column, or a level outside the declared set, would
        # not raise -- it would simply never match, and the column would be keyed on raw values
        # while everything downstream expected groups. Every other field here is validated; this
        # one was added later and was not.
        for col, cmap in (self.coarsen or {}).items():
            if col not in self.categorical:
                raise SchemaError(
                    f"coarsen column {col!r} is not a declared categorical column")
            if not cmap:
                raise SchemaError(f"coarsen[{col!r}] is empty")
            undeclared = sorted({str(k) for k in cmap} - {str(v) for v in self.categorical[col]})
            if undeclared:
                raise SchemaError(
                    f"coarsen[{col!r}] maps level(s) not in the declared value set: "
                    f"{undeclared[:5]}")

    # ── derived views ───────────────────────────────────────────────────────────────

    @property
    def numerical_cols(self) -> list[str]:
        return list(self.numerical)

    @property
    def categorical_cols(self) -> list[str]:
        return list(self.categorical)

    @property
    def columns(self) -> list[str]:
        return self.numerical_cols + self.categorical_cols + [self.target]

    def bin_edges(self, col: str) -> np.ndarray:
        """Public bin edges for a numeric column; a sane default if none were declared."""
        if col in self.bins:
            return np.asarray(self.bins[col], dtype=float)
        lo, hi = self.numerical[col]
        return np.linspace(lo, hi, 11)

    # ── validation ──────────────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame, *, strict: bool = True) -> dict:
        """Check a dataframe against this schema BEFORE any privacy budget is spent.

        Returns a report. With strict=True, raises if anything would make the release wrong
        rather than merely lossy. The distinction matters: values outside the declared bounds are
        clipped (a documented, safe operation), but a missing column or an unusable target cannot
        be recovered from and must stop the run.
        """
        problems: list[str] = []
        warnings: list[str] = []

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            problems.append(f"missing columns: {missing}")

        if self.target in df.columns:
            # NB: read the target as strings; a column of 0/1 read as ints will not match
            # declared string labels, and silently yielding an empty positive class is exactly
            # the kind of failure that looks like a modelling result.
            seen = set(df[self.target].astype(str).str.strip().unique())
            declared = {str(self.positive), str(self.negative)}
            unknown = seen - declared
            if unknown:
                problems.append(
                    f"target {self.target!r} contains values not declared as class labels: "
                    f"{sorted(unknown)[:5]} (declared: {sorted(declared)})"
                )
            if not (seen & declared):
                problems.append(f"target {self.target!r} contains none of the declared labels")
            elif len(seen & declared) == 1:
                problems.append(
                    f"target {self.target!r} has only one class present; nothing can be learned "
                    f"or generated from a constant target"
                )

        for c, (lo, hi) in self.numerical.items():
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            n_nan = int(v.isna().sum())
            if n_nan:
                warnings.append(f"{c}: {n_nan} non-numeric or missing value(s) will be dropped")
            out = int(((v < lo) | (v > hi)).sum())
            if out:
                warnings.append(
                    f"{c}: {out} value(s) outside the declared bounds [{lo}, {hi}] will be "
                    f"clipped. If that is a surprise, the declared bounds are wrong."
                )
            if v.notna().sum() == 0:
                problems.append(f"{c}: no usable numeric values at all")

        for c, allowed in self.categorical.items():
            if c not in df.columns:
                continue
            # keep_default_na is the caller's business, but 'None' as a CATEGORY is common in
            # clinical data (A1Cresult='None' means the test was not ordered) and must not be
            # confused with missingness.
            seen = set(df[c].astype(str).str.strip().unique())
            unknown = seen - set(map(str, allowed))
            if unknown:
                warnings.append(
                    f"{c}: {len(unknown)} value(s) not in the declared set will be recorded as "
                    f"out-of-domain: {sorted(unknown)[:5]}"
                )

        if len(df) == 0:
            problems.append("the dataset is empty")

        report = {"n_rows": len(df), "problems": problems, "warnings": warnings,
                  "ok": not problems}
        if strict and problems:
            raise DataValidationError(
                "the data does not match the declared schema; NO privacy budget was spent:\n  - "
                + "\n  - ".join(problems)
            )
        return report

    def coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the documented, safe normalisations: numeric coercion, clipping to declared
        bounds, and whitespace stripping on categoricals. Never invents or imputes values."""
        out = df.copy()
        for c, (lo, hi) in self.numerical.items():
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").clip(lo, hi)
        for c in list(self.categorical) + [self.target]:
            if c in out.columns:
                out[c] = out[c].astype(str).str.strip()
        num = [c for c in self.numerical if c in out.columns]
        return out.dropna(subset=num) if num else out
