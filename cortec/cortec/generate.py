"""
Stage B — generation. Reads only a `Release`; never touches private data, so it is post-processing
and costs no privacy budget. Any number of datasets may be drawn from one release.

Most of this file is guardrails rather than generation, and that is deliberate. Each check below
corresponds to a defect that, during the research behind this tool, produced a plausible but wrong
conclusion before being caught. They are cheap to check and expensive to rediscover:

  1. context truncation silently dropping the conditional table -> looked like "this model family
     ignores the DP statistics";
  2. a reasoning model spending its whole output budget on hidden chain-of-thought and returning
     empty content -> looked like "this model cannot follow the CSV schema";
  3. a reverse-proxy request timeout -> looked like "this backend is unstable";
  4. pandas reading the string 'None' as missing, deleting 83% of rows at a 100% call-success rate;
  5. rows accepted outside the schema's declared domain;
  6. uniform row allocation across cohorts of very different sizes, over-representing a 2% cohort
     by 12x;
  7. a low apparent yield measured over a denominator too small to act on.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import prompts
from .models import check_model, warnings_for
from .release import Release
from .schema import Schema
from . import report


class GenerationError(RuntimeError):
    """Stage B stopped: the message names the parse or yield fault, or the API condition."""


class ContextTruncationError(GenerationError):
    """The prompt did not fit in the model's context, so the conditioning was silently cut."""


class EmptyContentError(GenerationError):
    """The call succeeded but returned no content (typically a reasoning model out of budget)."""


# Which enterprise surface serves which vendor family. `public` is the vendor's developer API.
SURFACES = {"public": None, "bedrock": "anthropic", "azure": "openai", "vertex": "gemini"}


def _sdk(module: str, extra: str):
    """Import a vendor SDK, or say which extra installs it. An ImportError from deep inside the
    constructor named the missing module and nothing else; a deployment then had to guess which
    `pip install cortec[...]` it needed."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise GenerationError(
            f"backend needs the {module!r} package, which is not installed: "
            f"pip install cortec[{extra}]") from e


def _require_bedrock_deps() -> None:
    """AnthropicBedrock signs requests with botocore, which is the `cortec[bedrock]` extra. The
    SDK constructs the client without it and fails only at the first request; fail here instead."""
    try:
        import botocore  # noqa: F401
    except ImportError as e:
        raise GenerationError(
            "surface='bedrock' needs botocore for request signing: pip install 'cortec[bedrock]'"
        ) from e


@dataclass
class GenerationStats:
    calls: int = 0
    parse_ok: int = 0
    rows: int = 0
    rows_requested: int = 0
    rows_out_of_bounds: int = 0
    rows_undeclared_category: int = 0
    rate_limit_waits: int = 0
    empty_content: int = 0
    truncations: int = 0
    timeouts: int = 0
    spend_usd: float = 0.0
    # Set when the budget_usd cap stopped generation early. The output then holds the rows that
    # were completed within the budget, not the full request; the run does not fail.
    budget_capped: bool = False
    thinking_tokens: int = 0
    calls_without_reasoning: int = 0
    # A call whose reasoning could not be observed at all is tracked apart from one observed to
    # have none, so an absence of evidence can never be reported as evidence of absence.
    calls_reasoning_unmeasured: int = 0
    calls_with_reasoning_block: int = 0
    # Cell-wise generation ASKS for "exactly k positive of n" and does not force it: the emitted
    # count is measured per call and reported, never silently relabelled. A live Gemini smoke test
    # emitted 22 positives where 21 were requested in a 46-row cell -- small, but only visible if
    # it is counted.
    positives_requested: int = 0
    positives_emitted: int = 0
    positives_off_by_more_than_one: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def yield_rate(self) -> float:
        return self.rows / max(self.rows_requested, 1)


def _fmt_numeric(col: str, d: dict) -> str:
    edges, props = d["bin_edges"], d["proportions"]
    parts = [f"[{edges[i]:g},{edges[i+1]:g}): {props[i]:.1%}"
             for i in range(len(props)) if props[i] >= 0.005]
    lo, hi = d["bounds"]
    return (f"  {col}: range [{lo:g}, {hi:g}], mean {d['mean']:g}, sd {d['std']:g}\n"
            f"    distribution: {', '.join(parts)}")


def _fmt_categorical(col: str, d: dict) -> str:
    items = sorted(d.items(), key=lambda kv: -kv[1])
    return f"  {col}: " + ", ".join(f"{k} {v:.1%}" for k, v in items if v >= 0.001)


def _apportion(probs, n: int, rng) -> np.ndarray:
    """Integer counts summing to n from probabilities, by largest remainder with random ties."""
    p = np.asarray(probs, float); p = p / max(p.sum(), 1e-12)
    raw = p * n; base = np.floor(raw).astype(int); rem = raw - base
    short = int(n - base.sum())
    if short > 0:
        order = np.lexsort((rng.random(len(p)), -rem))
        base[order[:short]] += 1
    return base


def batch_quotas(schema: Schema, cohort: dict, n_rows: int, rng) -> dict:
    """Exact per-column counts for one batch, apportioned from the released histograms: the number
    of positives from the class balance and, for every column, rows per bin or per category.
    Where the release carries class blocks the positives' counts come from the positive block and
    the negatives' from the negative block; the batch total per column is their sum. Reads only the
    release, so handing the counts to the generator is post-processing. Measured on Claude Fable 5:
    a 25-row batch given these counts reproduced every one of 96 to 98 count lines exactly, where
    the plain prompt matched 59 to 72 of them."""
    cb = cohort.get("class_balance") or {}
    pos, neg = str(schema.positive), str(schema.negative)
    k = int(_apportion([float(cb.get(pos, 0.5)), float(cb.get(neg, 0.5))], n_rows, rng)[0])
    by_class = cohort.get("by_class") or {}
    parts = ([(k, by_class[pos]), (n_rows - k, by_class[neg])] if pos in by_class and neg in by_class
             else [(n_rows, cohort)])
    q = {"positives": k, "n": n_rows, "numerical": {}, "categorical": {}}
    for col in schema.numerical:
        edges = cohort["numerical"][col]["bin_edges"]
        counts = np.zeros(len(edges) - 1, int)
        for m, blk in parts:
            if m > 0:
                counts += _apportion(blk["numerical"][col]["proportions"], m, rng)
        q["numerical"][col] = [(f"{edges[i]:g}-{edges[i + 1]:g}", int(c)) for i, c in enumerate(counts) if c > 0]
    for col in schema.categorical:
        cats = list(cohort["categorical"][col].keys())
        counts = np.zeros(len(cats), int)
        for m, blk in parts:
            if m > 0:
                pr = blk["categorical"][col]
                counts += _apportion([float(pr.get(c, 0.0)) for c in cats], m, rng)
        q["categorical"][col] = [(c, int(v)) for c, v in zip(cats, counts) if v > 0]
    return q


def cohort_quota_targets(schema: Schema, cohort: dict, n_rows: int, rng) -> dict:
    """Integer targets for a whole cohort's n_rows (see batch_quotas); batches then ask for what
    the cohort still owes, so a batch returning more or fewer valid rows than asked cannot break
    the cohort's totals."""
    q = batch_quotas(schema, cohort, n_rows, rng)
    return {"n": n_rows, "positives": q["positives"],
            "numerical": {c: dict(v) for c, v in q["numerical"].items()},
            "categorical": {c: dict(v) for c, v in q["categorical"].items()}}


def emitted_counts(schema: Schema, cohort: dict, df: pd.DataFrame) -> dict:
    pos = (int((df[schema.target].astype(str).str.strip() == str(schema.positive)).sum())
           if schema.target in df.columns else 0)
    out = {"n": len(df), "positives": pos, "numerical": {}, "categorical": {}}
    for c in schema.numerical:
        if c not in df.columns:
            continue
        edges = cohort["numerical"][c]["bin_edges"]; e = np.asarray(edges, float)
        v = pd.to_numeric(df[c], errors="coerce").fillna(e[0]).values
        cnt = np.bincount(np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2), minlength=len(e) - 1)
        out["numerical"][c] = {f"{edges[i]:g}-{edges[i + 1]:g}": int(k) for i, k in enumerate(cnt) if k}
    for c in schema.categorical:
        if c not in df.columns:
            continue
        out["categorical"][c] = {str(k): int(v) for k, v in df[c].astype(str).str.strip().value_counts().items()}
    return out


def remaining_quotas(target: dict, done: dict, b: int, rng) -> dict:
    """Quotas for the next b rows, apportioned from what the cohort still owes."""
    def _scale(t: dict, d: dict) -> list:
        rem = {k: max(int(v) - int(d.get(k, 0)), 0) for k, v in t.items()}
        if sum(rem.values()) <= 0 or b <= 0:
            return []
        keys = list(rem); counts = _apportion([rem[k] for k in keys], b, rng)
        return [(k, int(c)) for k, c in zip(keys, counts) if c > 0]
    pos_rem = max(target["positives"] - done["positives"], 0); rows_rem = max(target["n"] - done["n"], 0)
    k = int(round(b * pos_rem / rows_rem)) if rows_rem > 0 else 0
    return {"n": b, "positives": min(max(k, 0), b),
            "numerical": {c: _scale(t, done["numerical"].get(c, {})) for c, t in target["numerical"].items()},
            "categorical": {c: _scale(t, done["categorical"].get(c, {})) for c, t in target["categorical"].items()}}


def _absorb(done: dict, e: dict) -> None:
    done["n"] += e["n"]; done["positives"] += e["positives"]
    for kind in ("numerical", "categorical"):
        for c, cnt in e[kind].items():
            d = done[kind].setdefault(c, {})
            for k, v in cnt.items():
                d[k] = d.get(k, 0) + v


def _normalise_categorical(col: pd.Series) -> pd.Series:
    """Categorical cells as declared strings: an integer-valued float (2.0) becomes "2", other
    values are stripped text, an empty cell becomes NaN (and fails the declared-set check)."""
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        t = str(v).strip()
        if re.fullmatch(r"-?\d+\.0+", t):
            return str(int(float(t)))
        return t if t else np.nan
    return col.map(one)


def _counts_block(schema: Schema, q: dict) -> str:
    lines = [f"EXACT COUNTS FOR THIS BATCH OF {q['n']} ROWS. These are requirements, not targets. Every "
             f"column's counts below sum to {q['n']}; plan the rows so that EVERY line is satisfied "
             f"exactly, then write them out.",
             f"  {schema.target}: exactly {q['positives']} rows \"{schema.positive}\" and "
             f"{q['n'] - q['positives']} rows \"{schema.negative}\""]
    for col, items in q["numerical"].items():
        lines.append(f"  {col}: " + ", ".join(f"{b}: {c} rows" for b, c in items))
    for col, items in q["categorical"].items():
        lines.append(f"  {col}: " + ", ".join(f"{c}: {v}" for c, v in items))
    return "\n".join(lines)


def build_prompt(schema: Schema, release: Release, cohort: dict, n_rows: int,
                 quotas: dict | None = None) -> tuple[str, object]:
    """Assemble the cohort prompt from released statistics only. With `quotas` (see
    `batch_quotas`) the prompt also carries exact per-column counts for the batch."""
    by_class = cohort.get("by_class") or {}
    if by_class:
        # The release carries one histogram block per outcome. Telling the model what positives
        # and negatives each look like lets it reproduce the feature->target structure for EVERY
        # column instead of inventing it from its prior for the columns the hierarchy does not
        # name -- on finance those invented columns measurably degraded the downstream model.
        head = ("  BY OUTCOME. Rows with different outcomes have DIFFERENT distributions. Decide "
                "each row's outcome first (class balance and the target table below), then draw "
                "its other columns from the distributions of THAT outcome.")
        num_parts, cat_parts = [head], [head]
        for cls, blk in by_class.items():
            num_parts.append(f"  === rows with {schema.target} = {cls} ===")
            num_parts += [_fmt_numeric(c, d) for c, d in blk["numerical"].items()]
            cat_parts.append(f"  === rows with {schema.target} = {cls} ===")
            cat_parts += [_fmt_categorical(c, d) for c, d in blk["categorical"].items()]
        numeric_block, categorical_block = "\n".join(num_parts), "\n".join(cat_parts)
    else:
        numeric_block = "\n".join(_fmt_numeric(c, d) for c, d in cohort["numerical"].items()) or "  (none)"
        categorical_block = "\n".join(_fmt_categorical(c, d)
                                      for c, d in cohort["categorical"].items()) or "  (none)"
    cb = cohort["class_balance"]
    class_balance_block = "  " + ", ".join(f"{k}: {v:.1%}" for k, v in cb.items())

    lines = []
    for lv in release.conditional_levels:
        lines.append(f"  by {', '.join(lv['columns'])}:")
        for cell, d in sorted(lv["cells"].items()):
            lines.append(f"    {cell}  ->  {d['rate']:.1%}")
    conditional_block = "\n".join(lines)

    return prompts.render_cohort_prompt(
        cohort_name=cohort["cohort_name"], n_rows=n_rows, numeric_block=numeric_block,
        categorical_block=categorical_block, class_balance_block=class_balance_block,
        conditional_block=conditional_block, columns=", ".join(schema.columns),
        target=schema.target, positive=schema.positive, negative=schema.negative,
        counts_block=_counts_block(schema, quotas) if quotas else "")


def _largest_remainder(shares, n_total: int) -> list[int]:
    """Integer allocation summing exactly to n_total, by largest remainder."""
    exact = np.asarray(shares, float) * int(n_total)
    base = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - base))[: int(n_total - base.sum())]:
        base[i] += 1
    return base.tolist()


def allocate_rows(release: Release, n_total: int) -> list[int]:
    """Rows per cohort, PROPORTIONAL to released cohort size.

    Allocating uniformly is the natural implementation and it is wrong: it over-represented a
    2%-of-population cohort by 12x, so every marginal measured afterwards was describing a
    deliberately incorrect mixture. Sizes here come from the release, not from private data.
    """
    sizes = np.array([c["cohort_size"] for c in release.cohorts], dtype=float)
    if sizes.sum() <= 0:
        raise GenerationError("released cohort sizes sum to zero")
    exact = n_total * sizes / sizes.sum()
    base = np.floor(exact).astype(int)
    # hand out the remainder to the largest fractional parts, so the total is exact
    for i in np.argsort(-(exact - base))[: int(n_total - base.sum())]:
        base[i] += 1
    return base.tolist()


# Share of the population that released cells must cover before cell-wise generation is safe.
# Measured head to head on a shared release: adult 99.9% coverage (cell-wise mildly better),
# a health survey 25.4% (cell-wise erased four of six racial groups). Two observations do not fix a
# threshold precisely; 0.90 sits in the empty gap between them and errs toward the path that cannot
# silently delete a subpopulation.
CELL_COVERAGE_FLOOR = 0.90
# Coverage is a ratio of floats, so a release sitting exactly ON the floor computes as
# 0.8999999... and would be refused. A release that exactly meets the bar must pass.
_FLOOR_EPS = CELL_COVERAGE_FLOOR - 1e-9

MIN_ROWS_PER_CELL = 8   # below this the per-cell prompt asks for 1-2 rows and parsing collapses


def positives_for_cell(rate: float, n_rows: int, rng) -> int:
    """How many of `n_rows` should carry the positive label, given a released `rate`.

    Uses **stochastic rounding**: floor(rate*n) plus one more with probability equal to the
    fractional part. Deterministic rounding would bias every small cell in the same direction —
    round-half-up turns a released 0.554 over 1 row into 1.000 every single time, which is exactly
    the saturation this path exists to remove. Stochastic rounding leaves the expected rate equal
    to the released rate, so the error is bounded by one row and averages out across draws rather
    than accumulating.
    """
    exact = float(rate) * int(n_rows)
    base = int(np.floor(exact))
    if rng.random() < (exact - base):
        base += 1
    return int(min(max(base, 0), n_rows))


def cell_description(cell_key: str, columns, coarsen: dict | None = None,
                     shares: dict | None = None) -> str:
    """Human-readable restatement of a released cell key, for the prompt.

    When a conditioning column has been COARSENED, the cell key carries an opaque group label
    (`disposition = g0`) that means nothing to the generator. Naming the group's members without
    their frequencies is barely better: it pins WHICH values are legal and says nothing about how
    often each occurs, and the model then spreads them roughly uniformly. Measured on a HIPAA
    discharge-disposition column coarsened into 7-value groups, the dominant category fell from a
    real 0.592 to 0.143 -- which is 1/7, the uniform-spreading signature -- and the column's total
    variation went 0.050 to 0.641.

    So the group is expanded into its members WITH their within-group shares. Those shares are a
    renormalisation of proportions already in the release, so supplying them is post-processing
    and costs no privacy budget. Validated on two vendors' models against one shared release
    (within-group TV 0.751 -> 0.022 and 0.435 -> 0.032).
    """
    coarsen = coarsen or {}
    shares = shares or {}
    parts = [p.strip() for p in str(cell_key).split("&")]
    out = []
    for p in parts:
        if "=" not in p:
            out.append(f"  {p}")
            continue
        col, val = (x.strip() for x in p.split("=", 1))
        cmap = coarsen.get(col)
        members = [lvl for lvl, g in (cmap or {}).items() if g == val]
        if not members:
            out.append(f"  {col} = {val}")
        elif len(members) == 1:
            out.append(f"  {col} = {members[0]}")
        else:
            dist = shares.get(col) or {}
            sub = [(m, float(dist.get(str(m), 0.0))) for m in members]
            tot = sum(x for _, x in sub)
            if tot > 0:
                sub = sorted(((m, x / tot) for m, x in sub), key=lambda kv: -kv[1])
                pretty = ", ".join(f"{m} {q:.0%}" for m, q in sub if q >= 0.005)
                out.append(f"  {col} is one of: {pretty} "
                           f"(these percentages are the shares WITHIN this group -- match them)")
            else:
                out.append(f"  {col} is one of: {', '.join(map(str, members))}")
    return "\n".join(out)


class ModelRefusalError(GenerationError):
    """The model declined to answer. Distinct from an exhausted output budget.

    Both surface as "no text content", and the remedies are opposite: a refusal needs a different
    prompt, a coarser cell, or a different generator, while an exhausted budget needs a larger one.
    Reporting a refusal as a budget problem sends the operator to re-run the same request at
    greater cost -- the same shape as the yield-guard defect, where a confident diagnosis pointed
    at the wrong subsystem entirely.
    """


class FatalAPIError(GenerationError):
    """An API refusal that will not resolve by retrying: credits, quota, auth, permissions.

    Separated from a normal call failure because the retry path treats the two very differently.
    A depleted account surfaced through the generic path looks like a PARSE problem -- the yield
    guard reports "0% of calls produced usable rows", which sends the operator to inspect the
    schema when the actual cause is billing. Worse, every retry spends wall-clock and, on a
    partially-funded account, real money, chasing a condition that cannot improve.
    """


# Vendors word these differently; the shared list keeps the abort discipline identical across all
# of them rather than depending on which backend happened to be configured.
_FATAL_MARKERS = (
    "credit", "quota", "billing", "authentication", "permission", "invalid_api_key",
    "invalid api key", "usage limit", "exceeded your current",
    "api key not valid", "unauthorized",
    # "resource exhausted" is NOT here: on Vertex AI it is how a transient rate limit is spelled
    # ("Resource exhausted. Please try again later."), and it is waited out (RateLimitedError)
    "payment", "account is not active",
    # Credential failures. Observed live on Vertex AI when a service account had been deleted:
    # every call failed with `invalid_grant: account not found` and the run counted each one as
    # a parse failure until the yield checkpoint stopped it. Nothing about retrying restores an
    # account; abort at the first one and say what to fix.
    "invalid_grant", "unauthenticated", "forbidden", "account not found",
    # A model that the surface does not serve under that name/location. Observed live on Vertex AI:
    # `gemini-3.5-flash` 404s in every regional location and is served from `global`; the run
    # retried the 404 twice before this marker existed. Retrying cannot make a model appear.
    "not_found", "does not exist",
)
# `insufficient_quota` was here and is deliberately absent: it contains "quota", so the shorter
# marker always matches first and the longer one is unreachable. A test that requires every marker
# to be individually load-bearing surfaced it. Keeping dead entries in a list like this is how it
# comes to be trusted without being exercised.


class RateLimitedError(RuntimeError):
    """The vendor asked us to wait (429 / resource exhausted / overloaded). Waited out with
    backoff by `Generator._call`; it never counts as a failed call unless it persists."""


_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "resource exhausted", "rate limit", "rate_limit",
                       "ratelimit", "try again later", "overloaded", "too many requests", "503")
_RATE_LIMIT_NOT = ("billing", "credit", "insufficient_quota", "spending cap", "api key", "unauthorized",
                   "permission", "authentication", "daily", "monthly")   # a daily/monthly cap is not transient


def _classify_api_error(vendor: str, e: Exception) -> Exception:
    """Return a FatalAPIError when retrying is pointless, a RateLimitedError when the vendor asked
    us to wait, otherwise the original exception."""
    text = f"{type(e).__name__}: {e}".lower()
    if any(m in text for m in _RATE_LIMIT_MARKERS) and not any(f in text for f in _RATE_LIMIT_NOT):
        return RateLimitedError(f"{vendor}: {str(e)[:300]}")
    for marker in _FATAL_MARKERS:
        if marker in text:
            return FatalAPIError(
                f"{vendor} refused the call for a reason that will not resolve by retrying "
                f"({marker!r}): {str(e)[:300]}\n"
                f"  Nothing further was attempted and no budget was spent retrying. Clear the "
                f"account condition and re-run; the DP release is unchanged, so resuming costs "
                f"no additional privacy budget."
            )
    return e


class Generator:
    """Frozen-LLM generator. The model is never fine-tuned and never sees a private record."""

    MIN_YIELD_RATE = 0.40
    MIN_REQUESTED_BEFORE_YIELD_ABORT = 60
    MIN_PARSE_RATE = 0.40
    CHECK_EVERY = 2

    # Measured list prices, $ per million tokens (input, output). Longest prefix wins, so a
    # cheap tier is never billed at its family's frontier rate — that mistake overcharged one
    # model 6.7x and tripped its budget cap at roughly a tenth of its true spend.
    PRICES: dict[str, tuple[float, float]] = {
        "claude-fable": (10.0, 50.0), "claude-opus": (10.0, 50.0),
        "claude-sonnet": (3.0, 15.0), "claude-haiku": (1.0, 5.0),
        "gpt-5": (1.25, 10.0), "gpt-4.1": (2.0, 8.0), "gpt-4o": (2.5, 10.0),
        "gemini-3-flash": (0.30, 2.50), "gemini-3.1-flash": (0.30, 2.50),
        "gemini-3.5-flash": (0.30, 2.50), "gemini-3.6-flash": (0.30, 2.50),
        "gemini-3.7-flash": (0.30, 2.50), "gemini-3": (2.0, 12.0),
        "gemini-2.5-pro": (1.25, 10.0), "gemini-2.5-flash": (0.30, 2.50),
    }

    def __init__(self, schema: Schema, *, backend: str = "anthropic",
                 model: str = "claude-fable-5", rows_per_call: int = 25,
                 ollama_url: str = "http://localhost:11434",
                 allow_unvalidated: bool = False, acknowledge_insufficient: bool = False,
                 budget_usd: float | None = None, request_timeout: float = 120.0,
                 max_retries: int = 3, seed: int | None = None,
                 reasoning: str = "on", surface: str = "public",
                 surface_model: str | None = None, quota: bool = True, quota_seed: int | None = None):
        self.schema = schema
        self.backend = backend
        self.model = model
        self.rows_per_call = rows_per_call
        # exact per-batch counts (build_prompt quotas): the model is given the number of rows per
        # bin and per category apportioned from the release for each batch, instead of shares
        self.quota = bool(quota)
        self._quota_rng = np.random.default_rng(quota_seed)
        self.max_retries = max_retries
        self.budget_usd = budget_usd
        self._seed = seed
        self.stats = GenerationStats()

        # Reasoning is a LOAD-BEARING configuration parameter, not a cost knob. Measured on one
        # model against itself, same release and same prompts, with only this flag changed:
        # conditional error 0.173 -> 0.045, a 3.8x improvement, moving the same model from BELOW
        # a no-information control to among the best generators measured. The suppressed setting
        # is 14.5x cheaper and still scores 95% of a real sample's downstream utility, so a user
        # optimising cost against the standard metric would choose it and ship a pipeline carrying
        # none of their data's conditional structure. The tool therefore defaults it ON and makes
        # suppressing it an explicit, argued choice.
        if reasoning not in ("on", "suppressed"):
            raise GenerationError(
                f"reasoning must be 'on' or 'suppressed', got {reasoning!r}")
        self.reasoning = reasoning
        if reasoning == "suppressed":
            self.stats.warnings.append(
                "reasoning=suppressed: measured 3.8x WORSE conditional fidelity, to below a "
                "no-information control, on the one model tested against itself. Aggregate "
                "utility barely moves, so this will not show up in a TSTR check. Verify "
                "conditional fidelity on your own data before relying on this output.")

        pin, pout = 10.0, 50.0            # conservative default for an unrecognised model
        for stem, (i, o) in sorted(self.PRICES.items(), key=lambda kv: -len(kv[0])):
            if str(model).startswith(stem):
                pin, pout = i, o
                break
        self._price_in, self._price_out = pin, pout

        # Gate capability BEFORE any call is made, so an unusable model costs nothing.
        self.profile = check_model(model, allow_unvalidated=allow_unvalidated,
                                   acknowledge_insufficient=acknowledge_insufficient)
        self.stats.warnings.extend(warnings_for(model))

        # Defect 2: a reasoning model's hidden chain of thought is billed against the same output
        # budget as the CSV. Budget for both, and size the context to hold prompt + both.
        self.max_output_tokens = 8192
        self.num_ctx = 32768

        # Which SURFACE serves the model. The vendors' public developer APIs are what the paper's
        # measurements used. For regulated data the recommended surface is the enterprise-hosted,
        # tenant-isolated one inside the institution's own account -- Claude on AWS Bedrock, GPT on
        # Azure OpenAI, Gemini on Vertex AI -- which carries the BAA, data residency, private
        # networking and no-training-on-inputs commitments a compliance review assesses. The
        # weights are the same; the contract is not. Request and response handling is identical on
        # both surfaces; what differs is the client class, its credential flow, and the identifier
        # the surface expects for the model:
        #   bedrock -> anthropic.AnthropicBedrock; AWS credentials from the standard chain (env,
        #              profile, instance role); `surface_model` is the Bedrock model id or
        #              inference-profile ARN
        #   azure   -> openai.AzureOpenAI; AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (or an Entra
        #              token via the SDK); `surface_model` is the DEPLOYMENT name
        #   vertex  -> google.genai.Client(vertexai=True); Application Default Credentials;
        #              GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION
        # `model` stays the validated profile name on every surface, so the capability gate and the
        # price table key on what the model IS, not on what a surface calls it. These adapters are
        # exercised against the SDKs' real client classes with the transport mocked (see tests);
        # they have NOT been run against a live enterprise account, and the README says so.
        if surface not in SURFACES:
            raise GenerationError(f"unknown surface {surface!r}; expected one of {sorted(SURFACES)}")
        if SURFACES[surface] and SURFACES[surface] != backend:
            raise GenerationError(
                f"surface={surface!r} serves the {SURFACES[surface]!r} backend, not {backend!r}. "
                f"Bedrock serves Claude, Azure OpenAI serves GPT, Vertex AI serves Gemini.")
        self.surface = surface
        self._request_model = surface_model or model
        if surface == "public" and backend in ("anthropic", "openai", "gemini"):
            self.stats.warnings.append(
                f"surface='public': {backend} is being reached through the vendor's PUBLIC developer "
                f"API. That is the surface the paper's measurements used and it is NOT the "
                f"recommended deployment for regulated data -- the weights are the same, the "
                f"contractual envelope (BAA, residency, private networking, no training on inputs) "
                f"is not. Use surface='bedrock' | 'azure' | 'vertex' inside your own account.")

        if backend == "anthropic":
            anthropic = _sdk("anthropic", "anthropic")
            if surface == "bedrock":
                region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
                if not region:
                    raise GenerationError(
                        "surface='bedrock' needs AWS_REGION (or AWS_DEFAULT_REGION) in the "
                        "environment; AWS credentials come from the standard chain (environment, "
                        "profile, instance role). Set surface_model to the Bedrock model id or "
                        "inference-profile ARN your account has access to.")
                _require_bedrock_deps()
                self._client = anthropic.AnthropicBedrock(aws_region=region, timeout=request_timeout)
            else:
                self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        elif backend == "openai":
            if surface == "azure":
                # configuration errors are reported before the SDK is required, so a missing
                # environment variable is named as such on a machine without the SDK installed
                endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
                if not endpoint:
                    raise GenerationError(
                        "surface='azure' needs AZURE_OPENAI_ENDPOINT "
                        "(https://<resource>.openai.azure.com) and AZURE_OPENAI_API_KEY in the "
                        "environment, and surface_model set to the deployment name.")
                if not surface_model:
                    raise GenerationError(
                        "surface='azure' needs surface_model=<deployment name>: Azure OpenAI "
                        "addresses a DEPLOYMENT you created, not a model id.")
                openai = _sdk("openai", "openai")
                self._client = openai.AzureOpenAI(
                    azure_endpoint=endpoint, api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
                    timeout=request_timeout)
            else:
                openai = _sdk("openai", "openai")
                self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                                             timeout=request_timeout)
        elif backend == "gemini":
            if surface == "vertex" and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
                raise GenerationError(
                    "surface='vertex' needs GOOGLE_CLOUD_PROJECT in the environment (and "
                    "GOOGLE_CLOUD_LOCATION, default 'global'); credentials come from Application "
                    "Default Credentials.")
            genai = _sdk("google.genai", "gemini")
            from google.genai import types as _gt
            # `HttpOptions.timeout` alone is NOT honoured by this SDK: the httpx client it builds
            # still carries Timeout(None), so a hung request blocks forever and the run stalls with
            # no error and no rows. `client_args` is forwarded to the httpx constructor and is the
            # one that binds. Keep both; HttpOptions is milliseconds, httpx is seconds.
            _http = _gt.HttpOptions(timeout=int(request_timeout * 1000),
                                    client_args={"timeout": float(request_timeout)})
            if surface == "vertex":
                project = os.environ.get("GOOGLE_CLOUD_PROJECT")
                if not project:
                    raise GenerationError(
                        "surface='vertex' needs GOOGLE_CLOUD_PROJECT (and GOOGLE_CLOUD_LOCATION, "
                        "default 'global') in the environment; credentials come from "
                        "Application Default Credentials.")
                # The current Gemini models are served on Vertex from the `global` location; a
                # regional default returned 404 for every region tried, so `global` is the default
                # and a region is an explicit choice.
                self._client = genai.Client(
                    vertexai=True, project=project,
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                    http_options=_http)
            else:
                self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                                            http_options=_http)
            bound = getattr(getattr(self._client, "_api_client", None), "_httpx_client", None)
            if bound is not None and getattr(bound, "timeout", None) is not None:
                if getattr(bound.timeout, "read", 1) is None:
                    raise GenerationError(
                        "the Gemini client did not accept a request timeout; a hung call would "
                        "block forever. Upgrade google-genai or pass request_timeout explicitly.")
        elif backend == "ollama":
            import httpx
            # Defect 3: a single long call behind a reverse proxy can exceed its request timeout,
            # which looks like backend instability. Keep calls short and time them out ourselves.
            self._http = httpx.Client(base_url=ollama_url, timeout=request_timeout)
            self._host_override = None
        elif backend != "mock":
            raise GenerationError(
                f"unknown backend {backend!r}; expected one of "
                f"'anthropic', 'openai', 'gemini', 'ollama', 'mock'")

    # ── prompt safety ───────────────────────────────────────────────────────────────

    def _assert_prompt_fits(self, prompt: str) -> None:
        """Defect 1: silent context truncation removes the conditional table.

        A conservative 3.5 characters-per-token estimate is used rather than a tokenizer, because
        the check must work for every backend. It errs toward raising, which is the safe direction:
        the failure it prevents is invisible in the output.
        """
        est = int(len(prompt) / 3.5) + self.max_output_tokens
        if self.backend == "ollama" and est > self.num_ctx:
            self.stats.truncations += 1
            raise ContextTruncationError(
                f"prompt (~{est - self.max_output_tokens} tokens) plus output budget "
                f"({self.max_output_tokens}) exceeds num_ctx={self.num_ctx}. Ollama silently "
                f"truncates, which cuts the conditional table off the end of the prompt and "
                f"produces unconditioned output that still looks valid. Raise num_ctx or lower "
                f"rows_per_call."
            )

    # ── backends ────────────────────────────────────────────────────────────────────

    # A rate limit is not a failed call. Two transient 429s in a row once aborted a run under the
    # yield rule, which counts a call that returned nothing as a parse failure. The vendor's
    # "try again later" is waited out with exponential backoff (15 s doubling to 4 min, six
    # waits, ~8 min in total) before the error is allowed to surface.
    RATE_LIMIT_RETRIES = 6
    RATE_LIMIT_FIRST_DELAY_S = 15.0

    def _call(self, prompt: str) -> str:
        delay = self.RATE_LIMIT_FIRST_DELAY_S
        for attempt in range(self.RATE_LIMIT_RETRIES + 1):
            try:
                return self._call_once(prompt)
            except RateLimitedError as rl:
                if attempt >= self.RATE_LIMIT_RETRIES:
                    raise
                self.stats.rate_limit_waits += 1
                time.sleep(delay)
                delay = min(delay * 2, 240.0)
        raise RateLimitedError("unreachable")

    def _call_once(self, prompt: str) -> str:
        self._assert_prompt_fits(prompt)
        if self.backend == "mock":
            return self._mock(prompt)
        if self.backend == "anthropic":
            # This vendor's reasoning is governed by `output_config.effort`, and it MUST be sent:
            # omitting it leaves the model on its default, which in our measurements fired
            # reasoning on some calls and not others. That intermittency was previously read as a
            # property of the model; it is a property of not asking. "low" is the setting measured
            # to reach ~0 reasoning tokens on this vendor, so it is what "suppressed" means here.
            kw: dict = {"output_config": {"effort": "high" if self.reasoning == "on" else "low"}}
            try:
                r = self._client.messages.create(
                    model=self._request_model, max_tokens=self.max_output_tokens,
                    system=prompts.system_prompt(),
                    messages=[{"role": "user", "content": prompt}], **kw)
            except Exception as e:
                raise _classify_api_error("Anthropic", e) from e
            usage = getattr(r, "usage", None)
            if usage:
                self._bill(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
                det = getattr(usage, "output_tokens_details", None)
                self._note_reasoning(
                    getattr(det, "thinking_tokens", 0) or 0 if det else None,
                    saw_block=any(getattr(b, "type", None) == "thinking" for b in r.content))
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            if not text.strip():
                self.stats.empty_content += 1
                stop = getattr(r, "stop_reason", "?")
                # A REFUSAL and an exhausted output budget both arrive as "no text", and the
                # remedies are opposite: one needs a different prompt or model, the other needs a
                # larger budget. Telling an operator to raise max_output_tokens after a refusal
                # sends them to rerun the same request at greater cost: a confident diagnosis
                # pointing at the wrong subsystem entirely.
                if stop == "refusal":
                    raise ModelRefusalError(
                        f"{self.model} declined to answer (stop_reason=refusal). This is not a "
                        f"token-budget problem and raising max_output_tokens will not fix it. It "
                        f"usually means a cell asked for very few rows, or the schema resembles "
                        f"personal records closely enough to trip a safety filter; the prompt "
                        f"carries only DP-released statistics, never a private record. Try a "
                        f"coarser conditional level so each cell asks for more rows, or a "
                        f"different generator."
                    )
                raise EmptyContentError(
                    f"{self.model} returned no text content (stop_reason={stop}). For a reasoning "
                    f"model this usually means the output budget went entirely to hidden "
                    f"reasoning; raise max_output_tokens."
                )
            return text

        if self.backend == "openai":
            kw: dict = {}
            # This vendor's "low" is NOT low — only "minimal" reaches zero reasoning tokens. We
            # therefore name the suppressed setting explicitly and leave the default alone when
            # reasoning is wanted, rather than passing a value that silently half-suppresses it.
            if self.reasoning == "suppressed":
                kw["reasoning_effort"] = "minimal"
            try:
                r = self._client.chat.completions.create(
                    model=self._request_model,
                    messages=[{"role": "system", "content": prompts.system_prompt()},
                              {"role": "user", "content": prompt}],
                    **kw)
            except Exception as e:
                raise _classify_api_error("OpenAI", e) from e
            u = getattr(r, "usage", None)
            if u:
                self._bill(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
                det = getattr(u, "completion_tokens_details", None)
                self._note_reasoning(getattr(det, "reasoning_tokens", 0) or 0 if det else None)
            text = (r.choices[0].message.content or "")
            if not text.strip():
                self.stats.empty_content += 1
                raise EmptyContentError(
                    f"{self.model} returned no text content (finish_reason="
                    f"{getattr(r.choices[0], 'finish_reason', '?')}). For a reasoning model this "
                    f"usually means the output budget went entirely to hidden reasoning.")
            return text

        if self.backend == "gemini":
            from google.genai import types as _gt
            cfg: dict = {"system_instruction": prompts.system_prompt()}
            if self.reasoning == "suppressed":
                # This vendor rejects a budget of 0 outright ("only works in thinking mode"), and
                # in our measurements did not honour a small budget either — so suppression here
                # is best-effort and is reported as such rather than assumed.
                cfg["thinking_config"] = _gt.ThinkingConfig(thinking_budget=128)
            try:
                r = self._client.models.generate_content(
                    model=self._request_model, contents=prompt,
                    config=_gt.GenerateContentConfig(**cfg))
            except Exception as e:
                raise _classify_api_error("Gemini", e) from e
            u = getattr(r, "usage_metadata", None)
            if u:
                # `thoughts_token_count` is reported SEPARATELY from `candidates_token_count` and
                # is billed at the output rate, so it must be added or spend undercounts ~2x.
                # (The other two vendors already fold reasoning into their output count.)
                thoughts = getattr(u, "thoughts_token_count", 0) or 0
                self._bill(getattr(u, "prompt_token_count", 0) or 0,
                           (getattr(u, "candidates_token_count", 0) or 0) + thoughts)
                self._note_reasoning(thoughts)
            text = r.text or ""
            if not text.strip():
                self.stats.empty_content += 1
                raise EmptyContentError(
                    f"{self.model} returned no text content. This is usually a safety block or a "
                    f"response whose budget went entirely to hidden reasoning.")
            return text

        # ollama
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": prompts.system_prompt()},
                                {"role": "user", "content": prompt}],
                   "options": {"num_ctx": self.num_ctx, "num_predict": self.max_output_tokens,
                               "temperature": 0.7}}
        headers = {"Host": self._host_override} if self._host_override else {}
        try:
            resp = self._http.post("/api/chat", json=payload, headers=headers)
        except Exception as e:
            self.stats.timeouts += 1
            raise GenerationError(
                f"ollama request failed ({type(e).__name__}). If this is a timeout behind a "
                f"reverse proxy, lower rows_per_call so a single call finishes sooner."
            ) from e
        if resp.status_code == 403 and self._host_override is None:
            # Ollama rejects a non-loopback Host unless bound to 0.0.0.0; retry once.
            self._host_override = "127.0.0.1"
            return self._call(prompt)
        resp.raise_for_status()
        d = resp.json()
        msg = d.get("message", {})
        content = (msg.get("content") or "").strip()
        if not content:
            self.stats.empty_content += 1
            thinking = len(msg.get("thinking") or "")
            raise EmptyContentError(
                f"{self.model} returned empty content (done_reason="
                f"{d.get('done_reason')!r}, {thinking} chars of hidden reasoning). The output "
                f"budget was consumed before any CSV was produced; raise max_output_tokens."
            )
        return content

    def _bill(self, tok_in: int, tok_out: int) -> None:
        self.stats.spend_usd += (tok_in / 1e6) * self._price_in
        self.stats.spend_usd += (tok_out / 1e6) * self._price_out

    def _note_reasoning(self, thinking_tokens: int | None, *, saw_block: bool = False) -> None:
        """Record whether reasoning actually fired, because the flag is not evidence that it did.

        A user who sets the flag and assumes it took effect can ship the suppressed configuration
        without knowing it -- the 14.5x-cheaper setting that produces output below a
        no-information control. So the setting is never trusted; only what comes back is.

        `thinking_tokens=None` means the vendor did not report a count on this call, which is NOT
        the same as reporting zero and must never be counted as one. We learned this the
        expensive way: a vendor SDK omits the token-details object entirely on its streaming
        path, and streaming is exactly what a large output budget forces -- i.e. the reporting
        gap opens precisely in the configuration a reasoning model needs. Recording those calls
        as "zero reasoning tokens" manufactured an apparent intermittency that was an artefact of
        the transport, not behaviour of the model.
        """
        if thinking_tokens is not None:
            self.stats.thinking_tokens += int(thinking_tokens)
        if saw_block:
            self.stats.calls_with_reasoning_block += 1
        if self.reasoning != "on":
            return
        if thinking_tokens is None and not saw_block:
            self.stats.calls_reasoning_unmeasured += 1
        elif not thinking_tokens and not saw_block:
            self.stats.calls_without_reasoning += 1

    def describe_surface(self) -> str:
        """One line for the audit trail: which surface the rows came through, and under what id."""
        return (f"backend={self.backend} surface={self.surface} model={self.model} "
                f"request_model={self._request_model}")

    def _mock(self, prompt: str) -> str:
        """An offline stand-in for a model. It honours the exact-count block of the cohort prompt
        (rows per bin, per category and per outcome, each column independently) when one is present,
        so the shipped generate -> select loop can be exercised end to end with no API; without one
        it draws uniformly, as before."""
        rng = np.random.default_rng(abs(hash(prompt)) % (2**32))
        m = re.search(r"exactly (\d+) synthetic", prompt)
        n = int(m.group(1)) if m else 5
        mk = re.search(r"EXACTLY (\d+) must have", prompt)     # the cell template's hard count
        k = int(mk.group(1)) if mk else None
        cols: dict[str, list[str]] = {}
        mq = re.search(r"EXACT COUNTS FOR THIS BATCH OF (\d+) ROWS", prompt)
        if mq:
            n = int(mq.group(1))
            block = prompt[mq.end():].split("\n\n", 1)[0].splitlines()[1:]
            for line in block:
                if ":" not in line:
                    continue
                col, rest = line.strip().split(":", 1); col = col.strip()
                vals: list[str] = []
                if col == self.schema.target:
                    mt = re.match(r'\s*exactly (\d+) rows "(.*?)" and (\d+) rows "(.*?)"', rest)
                    if mt:
                        vals = [mt.group(2)] * int(mt.group(1)) + [mt.group(4)] * int(mt.group(3))
                elif col in self.schema.numerical:
                    for item in rest.split(", "):
                        mb = re.match(r"\s*(-?[\d.]+)-(-?[\d.]+): (\d+) rows", item)
                        if mb:
                            lo, hi, c = float(mb.group(1)), float(mb.group(2)), int(mb.group(3))
                            vals += [str(int(rng.uniform(lo, max(lo, hi - 1e-9)))) for _ in range(c)]
                elif col in self.schema.categorical:
                    for item in rest.split(", "):
                        cat, _, c = item.rpartition(": ")
                        if c.strip().isdigit():
                            vals += [cat.strip()] * int(c)
                if vals:
                    vals = vals[:n] + [vals[-1]] * max(0, n - len(vals))
                    rng.shuffle(vals); cols[col] = vals
        rows = [",".join(self.schema.columns)]
        for i in range(n):
            vals = []
            for c in self.schema.numerical_cols:
                if c in cols:
                    vals.append(cols[c][i]); continue
                lo, hi = self.schema.numerical[c]
                vals.append(str(int(rng.uniform(lo, hi))))
            for c in self.schema.categorical_cols:
                vals.append(cols[c][i] if c in cols else str(rng.choice(self.schema.categorical[c])))
            if self.schema.target in cols:
                vals.append(cols[self.schema.target][i])
            elif k is None:
                vals.append(str(rng.choice([self.schema.positive, self.schema.negative])))
            else:
                vals.append(str(self.schema.positive if i < k else self.schema.negative))
            rows.append(",".join(vals))
        return "\n".join(rows)

    # ── parsing ─────────────────────────────────────────────────────────────────────

    def parse(self, text: str) -> pd.DataFrame | None:
        """Parse CSV output, rejecting anything that would silently corrupt the dataset."""
        if not text or not text.strip():
            return None
        text = re.sub(r"```[a-z]*\n?", "", text).strip()
        lines = text.splitlines()
        min_commas = max(2, len(self.schema.columns) - 5)
        start = next((i for i, l in enumerate(lines) if l.count(",") >= min_commas), None)
        if start is None:
            return None
        try:
            df = pd.read_csv(
                io.StringIO("\n".join(lines[start:])), skipinitialspace=True,
                on_bad_lines="skip",
                # Defect 4: pandas' default NA strings include 'None', 'NA' and 'nan', which are
                # LEGITIMATE CATEGORY VALUES in real data (a clinical result of 'None' means the
                # test was not ordered). With the default, those cells became NaN and the row was
                # dropped -- 83% of rows deleted at a 100% call-success rate.
                keep_default_na=False, na_values=[""])
        except Exception:
            return None

        df.columns = [str(c).strip() for c in df.columns]
        present = [c for c in self.schema.columns if c in df.columns]
        if len(present) < len(self.schema.columns) * 0.7:
            return None
        df = df[present]

        if self.schema.target in df.columns:
            t = df[self.schema.target].astype(str).str.strip()
            df = df[t.isin([self.schema.positive, self.schema.negative])]
            df[self.schema.target] = t

        num = [c for c in self.schema.numerical if c in df.columns]
        for c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=num) if num else df

        # Defect 6: categorical cells are normalised to the declared strings and checked against
        # the declared set. A batch that wrote an integer-coded category with a decimal point
        # ("2.0" for a column whose declared values are "1" and "2") typed that column float, and
        # concatenation re-typed every row of the pool, which the selection step and any consumer
        # then read as a category the data does not have. A value outside the declared set is a
        # malformed record, like an out-of-domain number, not a fidelity error to keep.
        cat = [c for c in self.schema.categorical if c in df.columns]
        if cat:
            keep = pd.Series(True, index=df.index)
            for c in cat:
                df[c] = _normalise_categorical(df[c])
                keep &= df[c].isin([str(v) for v in self.schema.categorical[c]])
            self.stats.rows_undeclared_category += int((~keep).sum())
            df = df[keep]

        # Defect 5: reject values outside the PUBLIC declared domain. These bounds are public, so
        # filtering on them costs nothing. A degenerate response once produced a credit limit of
        # 34 against a declared floor of 10,000 and an age of 0 against a floor of 21; without
        # this the rows entered the dataset silently.
        if num:
            keep = pd.Series(True, index=df.index)
            for c in num:
                lo, hi = self.schema.numerical[c]
                keep &= df[c].between(lo, hi)
            self.stats.rows_out_of_bounds += int((~keep).sum())
            df = df[keep]
        return df if len(df) else None

    # ── guards ──────────────────────────────────────────────────────────────────────

    def _checkpoint(self) -> None:
        s = self.stats
        # The spend cap is a limit the user chose, not a fault: the generation loop stops at it,
        # keeps the rows completed within the budget and sets stats.budget_capped. The two guards
        # below are faults: the run is spending money and producing nothing usable, so it stops
        # before spending more.
        parse_rate = s.parse_ok / max(s.calls, 1)
        if s.calls >= self.CHECK_EVERY and parse_rate < self.MIN_PARSE_RATE:
            raise GenerationError(
                f"only {parse_rate:.0%} of {s.calls} calls produced usable rows "
                f"(threshold {self.MIN_PARSE_RATE:.0%}). Fix the cause before spending more.")
        # Defect 7: a yield ratio needs a denominator worth acting on. Cohorts are generated
        # smallest-first on some schemas, so an early denominator can be a dozen rows, and one
        # unlucky response then reads as a systematic schema fault.
        if s.rows_requested >= self.MIN_REQUESTED_BEFORE_YIELD_ABORT \
                and s.yield_rate < self.MIN_YIELD_RATE:
            raise GenerationError(
                f"{s.rows} usable rows returned against {s.rows_requested} requested "
                f"({s.yield_rate:.0%} yield, threshold {self.MIN_YIELD_RATE:.0%}). Calls are "
                f"SUCCEEDING but rows are being dropped — check the schema before spending more.")

    # ── main entry point ────────────────────────────────────────────────────────────

    def _assert_reasoning_fired(self) -> None:
        """Warn when reasoning was asked for and the vendor reports none of it.

        This is the only check available: the setting is not evidence. A model that silently runs
        with reasoning off produces well-formed records whose conditional structure sits at a
        no-information floor, while aggregate utility stays near 95% of a real sample -- so nothing
        downstream will flag it either.
        """
        s = self.stats
        if s.positives_off_by_more_than_one:
            s.warnings.append(
                f"{s.positives_off_by_more_than_one} of {s.calls} cell-wise calls emitted a positive "
                f"count that missed the 'exactly k' requested by more than one row "
                f"({s.positives_emitted} emitted against {s.positives_requested} requested in total). "
                f"The model is ASKED for the count, not forced to it, so verify the per-cell rates of "
                f"this output against the release before relying on them; the tool never relabels "
                f"rows to hide the miss.")
        if self.reasoning != "on" or not s.calls:
            return
        if s.calls_without_reasoning >= s.calls:
            s.warnings.append(
                f"reasoning='on' was requested but the backend reported ZERO reasoning tokens on "
                f"all {s.calls} calls. Either this model has no reasoning mode, or it did not "
                f"engage. Measured effect of running without it: conditional error 3.8x worse, to "
                f"below a no-information control. Verify conditional fidelity before relying on "
                f"this output. (Some vendors do not report the count at all, in which case this "
                f"warning is not evidence of a problem -- check the vendor's usage fields.)")
        elif s.calls_without_reasoning:
            s.warnings.append(
                f"reasoning was intermittent: {s.calls_without_reasoning} of {s.calls} calls "
                f"showed no reasoning -- no reported tokens AND no reasoning block. Those calls "
                f"carry the suppressed configuration's fidelity, so check conditional fidelity "
                f"before relying on this output.")
        if s.calls_reasoning_unmeasured:
            s.warnings.append(
                f"reasoning could NOT be verified on {s.calls_reasoning_unmeasured} of {s.calls} "
                f"calls: this backend returned no reasoning-token count and no reasoning block, "
                f"so those calls are unmeasured rather than known-bad. Do not read this as "
                f"reasoning being off -- and do not read it as reasoning being on. If you need "
                f"the guarantee, verify conditional fidelity against a held-out sample directly.")

    def generate_selected(self, release: Release, n_rows: int, *, pool_factor: int = 3,
                          seed: int | None = None, verbose: bool = True,
                          sub_bin: str = "release") -> pd.DataFrame:
        """Generate `pool_factor * n_rows` records cohort-wise, then return the `n_rows` of them
        whose cell counts match the release (see `cortec.select`). The pool is post-processing of
        the release, so the selection costs no privacy budget; it costs `pool_factor` times the
        generation spend. Measured in the paper (section 7.12, Adult, three pools): a random 300
        of the pool carries 1-way marginal error 0.050 against real data; the selected 300 carry
        0.026, below a 300-record real sample's 0.043 and at MST's level, with the three
        downstream students unchanged or up by at most 0.017 AUC.

        `sub_bin="release"` (default) then makes every numeric value's position inside its bin
        the release's own in the cohorts that carry class-conditional blocks
        (`cortec.select.release_subbin_values`); `sub_bin="generator"` keeps the generator's."""
        from .select import select_to_release, release_subbin_values
        if pool_factor < 1:
            raise GenerationError("pool_factor must be at least 1")
        if sub_bin not in ("release", "generator"):
            raise GenerationError("sub_bin must be 'release' (default) or 'generator'")
        pool = self.generate(release, int(pool_factor * n_rows), verbose=verbose)
        # pool_factor 1 generates exactly n_rows with no pool to select from. A budget cap that
        # left fewer than n_rows rows is the partial result the user paid for (already flagged
        # budget-capped): return it as-is, since selection needs a pool at least as large as the
        # target. When the cap left n_rows or more, selection still runs, with allow_short_pool so
        # a cohort the cap left thin is handled by a warning rather than a refusal.
        if pool_factor == 1 or (self.stats.budget_capped and len(pool) < n_rows):
            return release_subbin_values(self.schema, release, pool, seed=seed) if sub_bin == "release" else pool
        return select_to_release(self.schema, release, pool, n_rows, seed=seed, sub_bin=sub_bin,
                                 allow_short_pool=self.stats.budget_capped)

    def generate(self, release: Release, n_rows: int, *, verbose: bool = True) -> pd.DataFrame:
        """Draw `n_rows` synthetic records from a release. Repeatable at no privacy cost."""
        if release.schema_name != self.schema.name:
            raise GenerationError(
                f"release is for schema {release.schema_name!r} but this generator was built for "
                f"{self.schema.name!r}")
        alloc = allocate_rows(release, n_rows)
        total = int(sum(alloc))
        frames = []
        for cohort, want in zip(release.cohorts, alloc):
            if want <= 0:
                continue
            got = 0
            attempts = 0
            max_attempts = (want // self.rows_per_call + 1) * self.max_retries
            _target = cohort_quota_targets(self.schema, cohort, int(want), self._quota_rng) if self.quota else None
            _done = {"n": 0, "positives": 0, "numerical": {}, "categorical": {}}
            while got < want and attempts < max_attempts:
                if self.budget_usd is not None and self.stats.spend_usd >= self.budget_usd:
                    self.stats.budget_capped = True
                    break
                attempts += 1
                ask = min(self.rows_per_call, want - got)
                prompt, _rec = build_prompt(
                    self.schema, release, cohort, ask,
                    quotas=remaining_quotas(_target, _done, ask, self._quota_rng) if self.quota else None)
                self.stats.rows_requested += ask
                self.stats.calls += 1
                try:
                    df = self.parse(self._call(prompt))
                except (EmptyContentError, ContextTruncationError, FatalAPIError):
                    raise                   # configuration or account faults: stop, do not retry
                except Exception as e:
                    if verbose:
                        report.emit(report.warn(f"[{cohort['cohort_name']}] call failed: {e}"))
                    df = None
                if df is not None and len(df):
                    if self.quota:
                        _absorb(_done, emitted_counts(self.schema, cohort, df))
                    self.stats.parse_ok += 1
                    self.stats.rows += len(df)
                    got += len(df)
                    df = df.copy()
                    df["_cohort"] = cohort["cohort_name"]
                    frames.append(df)
                if self.stats.calls % self.CHECK_EVERY == 0:
                    self._checkpoint()
                if verbose:
                    report.emit_progress(
                        f"Stage B · {self.backend} · call {self.stats.calls} · "
                        f"{self.stats.rows}/{total} rows · ${self.stats.spend_usd:.2f}")
            if verbose:
                report.emit_progress_done()     # wipe the stderr line before the stdout one
                report.emit(report.progress(f"{cohort['cohort_name'][:38]:40s} {got}/{want} rows"))
            if self.stats.budget_capped:
                break
        if verbose:
            report.emit_progress_done()
        if self.stats.budget_capped:
            self.stats.warnings.append(
                f"budget cap reached: ${self.stats.spend_usd:.2f} of ${self.budget_usd:.2f} after "
                f"{self.stats.calls} calls. This output holds the {self.stats.rows} rows completed "
                f"within the budget, not the {total} requested; raise budget_usd, or use a cheaper "
                f"validated model, for the full request.")
        if not frames:
            raise GenerationError(
                f"the budget of ${self.budget_usd:.2f} was spent (${self.stats.spend_usd:.2f} over "
                f"{self.stats.calls} calls) before any usable rows were produced; raise budget_usd"
                if self.stats.budget_capped else "no usable rows were generated")
        self._assert_reasoning_fired()
        return pd.concat(frames, ignore_index=True)

    # ── cell-wise generation ────────────────────────────────────────────────────────

    def generate_by_cell(self, release: Release, n_rows: int, *, level: int = 0,
                         allow_low_coverage: bool = False, verbose: bool = True) -> pd.DataFrame:
        """Generate one released conditional cell at a time, with an explicit positive COUNT.

        `generate()` asks for a mixed batch and leaves the model to allocate rows across cells.
        That caps how finely any single cell's rate can be expressed: on UCI Adult with 12 rows per
        call, `education = Masters` receives 0.65 rows per call and can only emit 0% or 100%, so a
        released rate of 0.554 rounds to 1.000 on every call. Measured end to end this produced
        MAE 0.253 with slope 1.64 — an apparent "rates above 0.24 saturate" threshold that was
        really "cells rarer than one row per call cannot express their rate".

        Generating per cell removes the coupling. Each call covers exactly one cell, the row count
        for that cell is known, and the prompt states the positive count as an integer rather than
        a probability, so the model has no discretion over the target column at all. Residual error
        is bounded by the stochastic rounding of one row.

        Costs more calls than `generate()` for the same output size; that is the trade.
        """
        if not release.conditional_levels:
            raise GenerationError("cell-wise generation needs a released conditional table")
        lvl = release.conditional_levels[min(level, len(release.conditional_levels) - 1)]
        cells = lvl["cells"]
        if not cells:
            raise GenerationError("the released conditional level has no cells")

        rng = np.random.default_rng(self._seed)
        support = np.array([c["support"] for c in cells.values()], float)

        # COVERAGE GUARD -- refuse per CONDITIONING COLUMN, not on aggregate cell mass.
        #
        # This path emits rows only for released cells, so any band of a conditioning column that
        # no released cell names is produced at essentially rate zero, and that is invisible
        # downstream. On a health survey whose release named a single blood-pressure band, 40.1% of
        # the population lay in unreleased bands and only 3.7% was emitted there; the output
        # contained no Asian, Mexican-American, other-Hispanic or multiracial records at all (real
        # shares 0.142, 0.127, 0.084, 0.052) while TSTR held at 0.729 against a cohort-wise 0.737.
        # Every utility number a practitioner checks stayed healthy while 40% of the population
        # was erased.
        #
        # The gate is per column because aggregate coverage does not predict the harm. Two releases
        # 34 points apart in aggregate coverage (25.4% and 59.7%) did identical damage, both having
        # released one blood-pressure band; and two releases an aggregate 90% rule refused (82.5%,
        # 87.0%) were in fact fine -- at 87.0% cell-wise beat cohort-wise on every aggregate metric.
        # Per-column coverage separates all five releases measured, in both directions.
        #
        # We refuse rather than silently switching paths: the caller asked for this path by name.
        # Everything is read from the release, so the check costs no privacy budget.
        _cols = tuple(lvl.get("columns") or ())
        _percol: dict[str, list[float]] = {}
        _pop = float(sum(c["cohort_size"] for c in release.cohorts)) or 1.0
        for _co in release.cohorts:
            _w = float(_co.get("cohort_size", 0)) / _pop
            for _col in _cols:
                _num = (_co.get("numerical") or {}).get(_col)
                if _num:
                    _edges = _num["bin_edges"]
                    _props = np.asarray(_num["proportions"], dtype=float)
                    _labels = [f"{_col}[{_edges[i]:g},{_edges[i+1]:g})" for i in range(len(_edges) - 1)]
                else:
                    _cat = (_co.get("categorical") or {}).get(_col)
                    if not _cat:
                        continue
                    # Cell keys carry the COARSENED label (`disposition=g0`) while the release's
                    # categorical block is keyed on raw levels, so the two must be brought onto the
                    # same alphabet before they are compared. Without this the guard matched nothing
                    # on every coarsened column, reported 0% coverage, and refused a release whose
                    # groups were in fact entirely covered.
                    _cmap = ((getattr(release, "coarsen", None) or {})
                             or (self.schema.coarsen or {})).get(_col)
                    if _cmap:
                        _agg: dict[str, float] = {}
                        for _v, _pr in _cat.items():
                            _g = _cmap.get(str(_v), str(_v))
                            _agg[_g] = _agg.get(_g, 0.0) + float(_pr)
                        _labels = [f"{_col}={g}" for g in _agg]
                        _props = np.asarray(list(_agg.values()), dtype=float)
                    else:
                        _labels = [f"{_col}={v}" for v in _cat]
                        _props = np.asarray(list(_cat.values()), dtype=float)
                _seen = {part.strip() for key in cells for part in str(key).split(" & ")}
                _props = _props / max(float(_props.sum()), 1e-9)
                _cov = float(sum(_props[i] for i, l in enumerate(_labels)
                                 if i < len(_props) and l in _seen))
                _acc = _percol.setdefault(_col, [0.0, 0.0])
                _acc[0] += _w * _cov
                _acc[1] += _w
        _percol_cov = {k: v[0] / max(v[1], 1e-9) for k, v in _percol.items()}
        _worst_col, _worst = (min(_percol_cov.items(), key=lambda kv: kv[1])
                              if _percol_cov else ("", 1.0))
        _short = ", ".join(f"{c} ({v:.0%} of its mass)" for c, v in
                           sorted(_percol_cov.items(), key=lambda kv: kv[1])
                           if v < CELL_COVERAGE_FLOOR)
        if _worst < _FLOOR_EPS and not allow_low_coverage:
            raise GenerationError(
                f"cell-wise generation is unsafe on this release: only {_short} lies in released "
                f"bands (floor {CELL_COVERAGE_FLOOR:.0%} per conditioning column). Rows outside "
                f"those bands are emitted at essentially rate zero, so that share of the "
                f"population would be missing from the output while aggregate fidelity and utility "
                f"still looked healthy. Either generate cohort-wise with generate(), or lower "
                f"n_min / use a coarser conditional level so more bands of '{_worst_col}' survive "
                f"the release. Pass allow_low_coverage=True only if you have checked that the "
                f"uncovered population does not matter for your use.")
        alloc = _largest_remainder(support / support.sum(), n_rows)
        total = int(sum(alloc))
        # Cell-wise generation needs n_rows >> len(cells). At 75 rows over 54 cells every cell gets
        # 1-2 rows, the prompt asks for a single record at a time, and parsing collapses — a 7B
        # model aborted at 0% parse in exactly this regime. Warn rather than let the run fail
        # halfway with an abort that looks like a model-capability result.
        _per = n_rows / max(len(cells), 1)
        if _per < MIN_ROWS_PER_CELL:
            report.emit(report.warn(f"{n_rows} rows across {len(cells)} cells is {_per:.1f} rows per cell; "
                  f"below ~{MIN_ROWS_PER_CELL} the per-cell prompt asks for one or two records and "
                  f"parsing degrades. Raise n_rows to >= {int(MIN_ROWS_PER_CELL*len(cells))} or "
                  f"generate at a coarser level."))

        # Population-level released marginals give the prompt its distributional context; the
        # cohort a cell spans is not identified here, so we use the size-weighted mixture.
        sizes = np.array([c["cohort_size"] for c in release.cohorts], float)
        w = sizes / sizes.sum()
        # the cell's conditioning columns are stated in the GROUP line; never restate them as
        # free distributions
        num_block, cat_block = self._mixture_blocks(release, w,
                                                    exclude=tuple(lvl.get("columns") or ()))
        # Size-weighted released category shares, used to expand a coarsened group label into its
        # members WITH their within-group frequencies. Renormalising released proportions is
        # post-processing, so this costs no budget.
        _mix_shares: dict[str, dict[str, float]] = {}
        for wi, coh in zip(w, release.cohorts):
            for col, dist in (coh.get("categorical") or {}).items():
                acc = _mix_shares.setdefault(col, {})
                for lvl_name, q in dist.items():
                    acc[str(lvl_name)] = acc.get(str(lvl_name), 0.0) + float(wi) * float(q)

        frames = []
        for (key, meta), want in zip(cells.items(), alloc):
            if want <= 0:
                continue
            got = 0
            attempts = 0
            while got < want and attempts < self.max_retries * (want // self.rows_per_call + 1):
                if self.budget_usd is not None and self.stats.spend_usd >= self.budget_usd:
                    self.stats.budget_capped = True
                    break
                attempts += 1
                ask = min(self.rows_per_call, want - got)
                npos = positives_for_cell(meta["rate"], ask, rng)
                prompt, _rec = prompts.render_cell_prompt(
                    cell_description=cell_description(key, self.schema.columns,
                                                      self.schema.coarsen, _mix_shares),
                    n_rows=ask, n_positive=npos, numeric_block=num_block,
                    categorical_block=cat_block, columns=", ".join(self.schema.columns),
                    target=self.schema.target, positive=self.schema.positive,
                    negative=self.schema.negative)
                self.stats.rows_requested += ask
                self.stats.calls += 1
                try:
                    df = self.parse(self._call(prompt))
                except (EmptyContentError, ContextTruncationError, FatalAPIError):
                    raise                   # configuration or account faults: stop, do not retry
                except Exception as e:
                    if verbose:
                        report.emit(report.warn(f"[{key}] call failed: {e}"))
                    df = None
                if df is not None and len(df):
                    self.stats.parse_ok += 1
                    self.stats.rows += len(df)
                    got += len(df)
                    df = df.copy()
                    # the count the prompt asked for versus the count the model emitted; the
                    # model is asked, not forced, and a miss must be visible to the caller
                    emitted = int((df[self.schema.target].astype(str).str.strip()
                                   == str(self.schema.positive)).sum())
                    self.stats.positives_requested += int(npos)
                    self.stats.positives_emitted += emitted
                    if abs(emitted - int(npos)) > 1:
                        self.stats.positives_off_by_more_than_one += 1
                    df["_cell"] = key
                    frames.append(df)
                if self.stats.calls % self.CHECK_EVERY == 0:
                    self._checkpoint()
                if verbose:
                    report.emit_progress(
                        f"Stage B · {self.backend} · call {self.stats.calls} · "
                        f"{self.stats.rows}/{total} rows · ${self.stats.spend_usd:.2f}")
            if verbose:
                report.emit_progress_done()     # wipe the stderr line before the stdout one
                report.emit(report.progress(f"{str(key)[:40]:42s} {got}/{want} rows "
                      f"(released rate {meta['rate']:.3f})"))
            if self.stats.budget_capped:
                break
        if verbose:
            report.emit_progress_done()
        if self.stats.budget_capped:
            self.stats.warnings.append(
                f"budget cap reached: ${self.stats.spend_usd:.2f} of ${self.budget_usd:.2f} after "
                f"{self.stats.calls} calls. This output holds the {self.stats.rows} rows completed "
                f"within the budget, not the {total} requested; raise budget_usd, or use a cheaper "
                f"validated model, for the full request.")
        if not frames:
            raise GenerationError(
                f"the budget of ${self.budget_usd:.2f} was spent (${self.stats.spend_usd:.2f} over "
                f"{self.stats.calls} calls) before any usable rows were produced; raise budget_usd"
                if self.stats.budget_capped else "no usable rows were generated")
        self._assert_reasoning_fired()
        return pd.concat(frames, ignore_index=True)

    def _mixture_blocks(self, release: Release, w, exclude: tuple[str, ...] = ()) -> tuple[str, str]:
        """Size-weighted mixture of the released per-cohort distributions, as prompt blocks.

        `exclude` drops columns that the caller pins elsewhere in the prompt. In cell-wise
        generation the cell's own conditioning columns are fixed by the GROUP line, and printing
        their population distribution as well contradicts it outright: a `grp = a` cell was being
        told "grp: a 60.1%, b 29.6%, c 10.3%" in the same prompt.
        """
        _skip = {c for c in exclude if c}
        first = release.cohorts[0]
        weighted = [(c, wi) for c, wi in zip(release.cohorts, w) if wi > 0]
        if weighted and all(c.get("by_class") for c, _ in weighted):
            head = ("  BY OUTCOME. Records with different outcomes have DIFFERENT distributions. "
                    "Draw each record's other columns from the distributions of ITS outcome.")
            num_lines, cat_lines = [head], [head]
            for cls in (self.schema.positive, self.schema.negative):
                blocks = [(c["by_class"][cls], wi) for c, wi in weighted]
                num_lines.append(f"  === records with {self.schema.target} = {cls} ===")
                cat_lines.append(f"  === records with {self.schema.target} = {cls} ===")
                n_l, c_l = self._mix_lines(blocks, first, _skip)
                num_lines += n_l; cat_lines += c_l
            return "\n".join(num_lines), "\n".join(cat_lines)
        n_l, c_l = self._mix_lines([(c, wi) for c, wi in weighted], first, _skip)
        return "\n".join(n_l) or "  (none)", "\n".join(c_l) or "  (none)"

    def _mix_lines(self, blocks, first, _skip) -> tuple[list[str], list[str]]:
        """Weighted mixture of histogram blocks (pooled cohorts or one class's blocks)."""
        num_lines, cat_lines = [], []
        for col in first["numerical"]:
            if col in _skip:
                continue
            edges = first["numerical"][col]["bin_edges"]
            acc = np.zeros(len(first["numerical"][col]["proportions"]))
            for c, wi in blocks:
                acc += wi * np.asarray(c["numerical"][col]["proportions"], float)
            lo, hi = first["numerical"][col]["bounds"]
            parts = [f"[{edges[i]:g},{edges[i+1]:g}): {acc[i]:.1%}"
                     for i in range(len(acc)) if acc[i] >= 0.005]
            num_lines.append(f"  {col}: range [{lo:g}, {hi:g}]\n    {', '.join(parts)}")
        for col in first["categorical"]:
            if col in _skip:
                continue
            acc: dict[str, float] = {}
            for c, wi in blocks:
                for k, v in c["categorical"][col].items():
                    acc[k] = acc.get(k, 0.0) + wi * v
            items = sorted(acc.items(), key=lambda kv: -kv[1])
            cat_lines.append(f"  {col}: " + ", ".join(f"{k} {v:.1%}" for k, v in items
                                                      if v >= 0.001))
        return num_lines, cat_lines
