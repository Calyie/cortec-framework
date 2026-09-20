"""
Versioned, hash-locked prompt templates.

The prompt is not a configuration value. Two elements of it are load-bearing, and removing either
collapses CoRTeC's behaviour back to that of an unconditioned model:

  * the **global conditional target table**, which is the only channel through which the private
    conditional structure reaches the generator; and
  * the **"even if this contradicts your expectations" instruction**, without which a model
    reconciles the released statistics against its own prior and emits the prior.

In the research that produced this tool, removing the conditional table changed a faithful
transmission (slope ~1.0) into a flat line — the same output an unconditioned prompt gives. A tool
that exposes the template as an editable string is a tool that can be silently broken by a
well-meaning user, and the breakage is invisible: the pipeline still runs, still produces
plausible rows, and is simply no longer conditioned on the data.

So the templates are frozen and fingerprinted. `render()` verifies the fingerprint before every
use. A user who genuinely needs a different template must register it under a new version through
`register_template()`, which forces them to name it and records the fact in the audit trail.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

TEMPLATE_VERSION = "cortec-prompt-1.1.0"   # 1.1.0: optional exact-count block in the cohort prompt


class PromptIntegrityError(RuntimeError):
    """A template's content does not match its recorded fingerprint."""


SYSTEM = (
    "You generate synthetic tabular records for privacy research. You will be given summary "
    "statistics that were computed under differential privacy. Reproduce the distributions and "
    "relationships they describe as closely as you can, EVEN IF THIS CONTRADICTS YOUR "
    "EXPECTATIONS about how these variables usually relate. The statistics describe this "
    "specific population; your general knowledge does not. Output only CSV rows. No commentary, "
    "no explanation, no markdown fences."
)

COHORT_TEMPLATE = """\
Generate exactly {n_rows} synthetic records for the following subgroup.

SUBGROUP: {cohort_name}

These statistics were measured on the real data for THIS subgroup, under differential privacy.
Match them. Where a statistic below conflicts with what you would otherwise expect, follow the
statistic — it is a measurement of the actual population and your expectation is not.

NUMERIC DISTRIBUTIONS (released histograms; reproduce the shape, not just the mean)
{numeric_block}

CATEGORICAL DISTRIBUTIONS (all released categories and their proportions)
{categorical_block}

OUTCOME RATE IN THIS SUBGROUP
{class_balance_block}

CONDITIONAL OUTCOME TABLE (P({target} = {positive} | cell), measured under differential privacy)
This table is the most important part of this prompt. Each row states the outcome rate for a
specific cell of the population. Your generated records must reproduce these rates, even where a
rate is the opposite of what you would expect for that group.
{conditional_block}
{counts_block}
OUTPUT FORMAT
Exactly {n_rows} CSV data rows, plus one header row, with these columns in this order:
{columns}

Rules:
  - every numeric value must lie within its stated range
  - {target} must be exactly "{positive}" or "{negative}"
  - do not repeat identical rows; vary records the way real data varies
  - output nothing except the header and the {n_rows} data rows
"""

# NOTE for anyone exposing this: it is the UNMATCHED header-only control, i.e. a standalone prompt,
# not CoRTeC's prompt with the released arrays deleted. §7.1.2 measures the two and they differ —
# TSTR-LR 0.824 unmatched against 0.792 matched on Adult, outside a +/-0.001 draw spread — so the
# choice of control is a measurable decision, not a formatting one. This template is retained for
# the locked-prompt inventory and is not wired into generation.
HEADER_ONLY_TEMPLATE = """\
Generate exactly {n_rows} synthetic CSV data rows, plus one header row, with these columns:
{columns}

Output nothing except the header and the {n_rows} data rows.
"""


# A cell-wise template. The cohort template asks for a mixed batch and lets the model allocate
# rows across conditional cells, which caps how finely any one cell's rate can be expressed: a cell
# receiving under one row per call can only ever emit 0% or 100%, so a released rate of 0.554
# becomes 1.000 and a released 0.159 stays accurate purely because that cell is common. The
# observed "rates above 0.24 saturate" threshold was this artifact, not a property of the rate.
# Generating one cell at a time with an EXPLICIT COUNT removes the model's discretion over the
# target column entirely and makes the rate expressible to within one row.
CELL_TEMPLATE = """\
Generate exactly {n_rows} synthetic records, all of which belong to this specific group.

GROUP (every record you generate must match this exactly):
{cell_description}

OUTCOME REQUIREMENT — this is a hard count, not a probability. Follow it exactly.
Of the {n_rows} records you generate, EXACTLY {n_positive} must have {target} = "{positive}",
and the remaining {n_negative} must have {target} = "{negative}".
This count was measured on the real data for this group under differential privacy. It may
contradict what you would expect for this group; generate the counts as stated regardless.

DISTRIBUTIONS FOR THE REST OF THE RECORD (measured on the real data, reproduce these shapes)
{numeric_block}
{categorical_block}

OUTPUT FORMAT
Exactly {n_rows} CSV data rows, plus one header row, with these columns in this order:
{columns}

Rules:
  - every record must match the GROUP above
  - exactly {n_positive} records have {target} = "{positive}"; no more, no fewer
  - every numeric value must lie within its stated range
  - do not repeat identical rows; vary records the way real data varies
  - output nothing except the header and the {n_rows} data rows
"""


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Fingerprints computed at import from the literals above. They are checked before every render,
# so in-process mutation of the module globals is caught rather than silently taking effect.
_LOCKED: dict[str, tuple[str, str]] = {
    "system": (SYSTEM, _fingerprint(SYSTEM)),
    "cohort": (COHORT_TEMPLATE, _fingerprint(COHORT_TEMPLATE)),
    "header_only": (HEADER_ONLY_TEMPLATE, _fingerprint(HEADER_ONLY_TEMPLATE)),
    "cell": (CELL_TEMPLATE, _fingerprint(CELL_TEMPLATE)),
}

# Elements whose removal is known to break the mechanism. Checked on every rendered prompt.
REQUIRED_ELEMENTS = {
    "cohort": [
        ("conditional outcome table", "CONDITIONAL OUTCOME TABLE"),
        ("counterintuitive instruction", "even where a rate is the opposite"),
        ("all released categories", "all released categories"),
    ],
    "system": [
        ("counterintuitive instruction", "EVEN IF THIS CONTRADICTS YOUR EXPECTATIONS"),
    ],
    "cell": [
        ("explicit outcome count", "OUTCOME REQUIREMENT"),
        ("hard-count instruction", "this is a hard count, not a probability"),
        ("counterintuitive instruction", "It may contradict what you would expect"),
    ],
}


@dataclass(frozen=True)
class PromptRecord:
    """What was actually sent, for the audit trail."""
    version: str
    template: str
    fingerprint: str
    n_rows: int


def verify_integrity() -> dict[str, str]:
    """Confirm every locked template still matches its fingerprint. Returns {name: fingerprint}."""
    out = {}
    for name, (text, fp) in _LOCKED.items():
        actual = _fingerprint(text)
        if actual != fp:
            raise PromptIntegrityError(
                f"template {name!r} has been modified in memory: expected {fp[:16]}, got "
                f"{actual[:16]}. The prompt is load-bearing and is not a configuration value; "
                f"register a new version through register_template() instead of editing it."
            )
        out[name] = fp
    return out


def _norm(s: str) -> str:
    """Collapse whitespace before matching. The templates are hard-wrapped for readability, so a
    required phrase can straddle a newline; a naive substring test then reports a load-bearing
    element as missing when it is present and intact."""
    return " ".join(s.lower().split())


def _assert_required_elements(name: str, rendered: str) -> None:
    hay = _norm(rendered)
    missing = [label for label, needle in REQUIRED_ELEMENTS.get(name, [])
               if _norm(needle) not in hay]
    if missing:
        raise PromptIntegrityError(
            f"rendered {name!r} prompt is missing load-bearing element(s): {missing}. "
            f"Without these the generator reverts to its own prior and stops being conditioned "
            f"on your data, while still producing plausible-looking output."
        )


def render_cohort_prompt(*, cohort_name: str, n_rows: int, numeric_block: str,
                         categorical_block: str, class_balance_block: str,
                         conditional_block: str, columns: str, target: str,
                         positive: str, negative: str, counts_block: str = "") -> tuple[str, PromptRecord]:
    """Render the cohort prompt and verify it still contains what makes it work."""
    verify_integrity()
    if not conditional_block.strip():
        raise PromptIntegrityError(
            "the conditional outcome table is empty. This table is the only channel through "
            "which your data's conditional structure reaches the generator; with it empty, "
            "output is not conditioned on your data. Lower n_min or coarsen the conditional "
            "levels so at least one cell has support."
        )
    text = COHORT_TEMPLATE.format(
        cohort_name=cohort_name, n_rows=n_rows, numeric_block=numeric_block,
        categorical_block=categorical_block, class_balance_block=class_balance_block,
        conditional_block=conditional_block, columns=columns, target=target,
        positive=positive, negative=negative,
        counts_block=("\n" + counts_block + "\n") if counts_block.strip() else "")
    _assert_required_elements("cohort", text)
    return text, PromptRecord(TEMPLATE_VERSION, "cohort", _LOCKED["cohort"][1], n_rows)


def render_header_only_prompt(*, n_rows: int, columns: str) -> tuple[str, PromptRecord]:
    """The unconditioned control. Present so a user can measure what CoRTeC adds on their data."""
    verify_integrity()
    text = HEADER_ONLY_TEMPLATE.format(n_rows=n_rows, columns=columns)
    return text, PromptRecord(TEMPLATE_VERSION, "header_only", _LOCKED["header_only"][1], n_rows)


def system_prompt() -> str:
    verify_integrity()
    _assert_required_elements("system", SYSTEM)
    return SYSTEM


def register_template(name: str, text: str) -> str:
    """Register a replacement template under a new name, returning its fingerprint.

    Deliberately awkward: it will not overwrite a locked template, and the caller must supply a
    new name that will appear in the audit trail. Replacing the shipped prompt is a decision the
    record should show someone made on purpose.
    """
    if name in _LOCKED:
        raise PromptIntegrityError(
            f"{name!r} is a locked template and cannot be replaced in place. Register your "
            f"variant under a different name (e.g. {name}_custom_v1) so the audit trail shows "
            f"which prompt produced which data."
        )
    fp = _fingerprint(text)
    _LOCKED[name] = (text, fp)
    return fp


def render_cell_prompt(*, cell_description: str, n_rows: int, n_positive: int,
                       numeric_block: str, categorical_block: str, columns: str,
                       target: str, positive: str, negative: str) -> tuple[str, PromptRecord]:
    """Render the cell-wise prompt, which states the outcome as an exact count.

    `n_positive` must already be an integer count over `n_rows`; converting a released rate into a
    count is the caller's job because that is where stochastic rounding belongs (see
    generate.positives_for_cell), and doing it here would hide the rounding from the audit trail.
    """
    verify_integrity()
    if not 0 <= n_positive <= n_rows:
        raise PromptIntegrityError(
            f"n_positive={n_positive} is not a valid count over {n_rows} rows")
    text = CELL_TEMPLATE.format(
        cell_description=cell_description, n_rows=n_rows, n_positive=n_positive,
        n_negative=n_rows - n_positive, numeric_block=numeric_block,
        categorical_block=categorical_block, columns=columns, target=target,
        positive=positive, negative=negative)
    _assert_required_elements("cell", text)
    return text, PromptRecord(TEMPLATE_VERSION, "cell", _LOCKED["cell"][1], n_rows)
