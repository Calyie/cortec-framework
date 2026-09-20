"""
Model capability gating.

CoRTeC degrades *silently*. A model too weak to track the released conditional structure does not
error — it produces well-formed, plausible rows that simply ignore the statistics, which is
indistinguishable from success unless you measure transmission directly. In the measurements
behind this tool, a 7B model scored a transmission slope of 0.21 (a faithful generator scores
~1.0), while emitting perfectly valid CSV throughout.

So capability is gated rather than documented. The thresholds below are measured, not guessed, and
the failure mode they prevent is a false privacy-preserving-data-generation claim: output that
looks fine and carries none of your data's structure.

Two facts shape the policy:
  * scale matters — within one family, magnitude error fell monotonically from 7B to 72B;
  * scale is not sufficient — at comparable size, models from different families differed by 2x,
    with a 24B model beating a 32B one.

Because family matters and this list cannot enumerate every model, an unknown model is not
assumed good. It is allowed to run only in `measure` mode, which requires the caller to verify
transmission on their own data before trusting the output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    """Capability tiers.

    A tier records what a model was measured to do ON FULL-DATASET GENERATION, which is what a
    caller actually runs. An earlier version of this table assigned tiers from the *transmission
    sweep* alone -- a narrow test of whether a model reproduces a released rate -- and two
    self-hosted 70B-class models scored inside the frontier band on it. Measured on full-dataset
    generation the same models closed 12% of the conditional-error gap to an UNCONDITIONED prompt
    and nothing at all on 2-way total variation. The sweep is a necessary condition, not a
    sufficient one, and calling a model "validated" on the strength of it vouched for a
    configuration that does not work.
    """
    VALIDATED = "validated"       # measured on full-dataset generation, accurate throughout
    ADEQUATE = "adequate"         # measured, direction right, magnitude approximate
    SWEEP_ONLY = "sweep-only"     # passes the transmission sweep; fails full-dataset generation
    INSUFFICIENT = "insufficient"  # measured, does not track the data
    UNKNOWN = "unknown"           # not measured by us


class ModelCapabilityError(RuntimeError):
    """The requested model is below the validated capability floor for this mechanism."""


@dataclass(frozen=True)
class ModelProfile:
    pattern: str          # regex matched against the model identifier, case-insensitively
    tier: Tier
    magnitude_error: float | None   # mean |generated rate - released rate|, lower is better
    slope: float | None             # transmission slope; 1.0 is faithful
    note: str


# Measured on a real hospital-readmission dataset by forcing a subgroup's outcome rate across
# 0% / 50% / 100% and regressing generated rate on the rate actually shown to the model.
PROFILES: list[ModelProfile] = [
    ModelProfile(r"claude.*(fable|opus|sonnet)", Tier.VALIDATED, 0.009, 0.995,
                 "frontier API model; most accurate measured"),
    # Measured 2026-09-10 on the SAME transmission sweep as every other row here (diabetes,
    # number_inpatient forced to 0/50/100%, 3 rates x 75 rows): shown 0.004/0.505/0.995 ->
    # generated 0.000/0.521/1.000, slope 1.009, MAE 0.008. Added only after running that sweep —
    # an earlier magnitude-error figure of 0.002 existed from the cell-wise registry test, which
    # is a DIFFERENT experiment and is not interchangeable with this metric.
    # FLASH ONLY. A pattern of `gemini.*(flash|pro)` would silently vouch for gemini-*-pro, which
    # has NOT been run through this sweep — asserting a tier for an unmeasured sibling is the exact
    # error this table exists to prevent. Measure pro before adding it.
    ModelProfile(r"gemini.*flash", Tier.VALIDATED, 0.008, 1.009,
                 "frontier API model; measured on the same sweep as the rest of this table"),
    # These two were deliberately absent while tiers were assigned from the transmission sweep:
    # they were never scored on it. That ordering is now inverted -- full-dataset generation is
    # the axis this tool runs and the axis the recommendation is built on -- so a model measured
    # THERE is validated, with magnitude_error left None because the sweep number does not exist.
    ModelProfile(r"gemini.*3\.1.*pro", Tier.VALIDATED, None, None,
                 "validated on full-dataset generation: conditional error 0.041, the lowest of "
                 "any generator measured. Not scored on the transmission sweep."),
    ModelProfile(r"gpt-5", Tier.VALIDATED, None, None,
                 "validated on full-dataset generation at DEFAULT reasoning: conditional error "
                 "0.045. At reasoning_effort='minimal' the same model scores 0.173, past the "
                 "no-information floor. Not scored on the transmission sweep."),
    # DEMOTED from VALIDATED. On the transmission sweep these reach 0.020 and 0.030, inside the
    # frontier band -- which is why they were listed as validated, and why that listing was wrong.
    # Re-measured on full-dataset generation at three draws, llama3.3:70b records conditional-seen
    # error 0.133 against an UNCONDITIONED control's 0.145 and a frontier model's 0.030: it closes
    # 12% of the gap to no conditioning at all, closes nothing on 2-way TV (0.238 against the
    # control's identical 0.238), and is worse than the control on 1-way TV. None of these models
    # has a reasoning mode, which is the likely mechanism.
    ModelProfile(r"llama3\.3.*70b", Tier.SWEEP_ONLY, 0.020, 0.988,
                 "self-hostable; passes the transmission sweep but on full-dataset generation is "
                 "close to indistinguishable from an unconditioned prompt"),
    ModelProfile(r"qwen2\.5.*72b", Tier.SWEEP_ONLY, 0.030, 1.011,
                 "self-hostable; sweep-only, same caveat as llama3.3:70b"),
    ModelProfile(r"mistral-small.*24b", Tier.ADEQUATE, 0.049, 0.877,
                 "beats larger models from other families; magnitude approximate"),
    ModelProfile(r"gemma2.*27b", Tier.ADEQUATE, 0.067, 0.890, "magnitude approximate"),
    ModelProfile(r"gpt-oss.*20b", Tier.ADEQUATE, 0.078, 0.913,
                 "reasoning model: budget output tokens generously, see generate.py"),
    ModelProfile(r"qwen2\.5.*32b", Tier.ADEQUATE, 0.098, 1.014,
                 "slope is near 1 but midpoints are badly off; judge on magnitude error"),
    ModelProfile(r"qwen2\.5.*14b", Tier.ADEQUATE, 0.105, 1.011,
                 "slope near 1 but generated 81% for a 50% target"),
    ModelProfile(r"qwen2\.5.*7b", Tier.INSUFFICIENT, 0.406, 0.208,
                 "does not track the released statistics; output is effectively unconditioned"),
]

# Below this magnitude error a model is treated as accurate throughout.
VALIDATED_MAX_MAGNITUDE_ERROR = 0.035
# Above this, the model is not usable for this mechanism at all.
INSUFFICIENT_MIN_MAGNITUDE_ERROR = 0.20


def profile_for(model: str) -> ModelProfile:
    for p in PROFILES:
        if re.search(p.pattern, model, flags=re.IGNORECASE):
            return p
    return ModelProfile(r"", Tier.UNKNOWN, None, None,
                        "not in the measured set; transmission is unverified on any dataset")


def _small_model_hint(model: str) -> str | None:
    """A parameter count in the name is weak evidence, so it only ever produces a warning."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        b = float(m.group(1))
    except ValueError:
        return None
    if b < 13:
        return (f"the name suggests roughly {b:g}B parameters. Every model under ~13B we measured "
                f"failed to track the released statistics while still emitting valid rows.")
    return None


def check_model(model: str, *, allow_unvalidated: bool = False,
                acknowledge_insufficient: bool = False) -> ModelProfile:
    """Gate a model before any generation happens.

    Raises for a model measured as insufficient, for one that passes only the transmission sweep,
    and for an unmeasured model, unless the caller explicitly opts in. The opt-in flags are
    separate and none defaults to True, because the risks differ: an unknown model *might* be
    fine, whereas a model on the insufficient or sweep-only lists is known not to be.
    """
    p = profile_for(model)

    # A sweep-only model is the most dangerous entry in this table, because it LOOKS validated:
    # it reproduces a released rate accurately in isolation, and then attends weakly to the full
    # release. It is refused under the same flag as an insufficient model.
    if p.tier is Tier.SWEEP_ONLY and not acknowledge_insufficient:
        raise ModelCapabilityError(
            f"{model!r} passes the transmission sweep but FAILS full-dataset generation.\n"
            f"  sweep: magnitude error {p.magnitude_error}, slope {p.slope} — inside the frontier "
            f"band, which is why this model was previously listed as validated.\n"
            f"  full-dataset generation, 3 draws: it closes 12% of the conditional-error gap to an "
            f"UNCONDITIONED prompt, closes nothing at all on 2-way total variation, and is worse "
            f"than that control on 1-way total variation.\n"
            f"  {p.note}\n"
            f"  On the metrics this method exists to improve, its output is close to "
            f"indistinguishable from asking a model for plausible records with no conditioning — "
            f"and no output check reveals that. Use a validated model "
            f"({', '.join(validated_models())}), or pass acknowledge_insufficient=True if you are "
            f"deliberately reproducing the finding."
        )

    if p.tier is Tier.INSUFFICIENT and not acknowledge_insufficient:
        raise ModelCapabilityError(
            f"{model!r} is below the capability floor for CoRTeC.\n"
            f"  measured transmission slope {p.slope} (faithful is ~1.0), "
            f"magnitude error {p.magnitude_error}\n"
            f"  {p.note}\n"
            f"  This model will still produce well-formed synthetic rows. They will not carry "
            f"your data's conditional structure, and no output check will reveal that. "
            f"Use a validated model, or pass acknowledge_insufficient=True if you are "
            f"deliberately reproducing the failure."
        )

    if p.tier is Tier.UNKNOWN and not allow_unvalidated:
        hint = _small_model_hint(model)
        ev = generation_evidence_for(model)
        raise ModelCapabilityError(
            f"{model!r} has not been measured for transmission fidelity.\n"
            + (f"  {hint}\n" if hint else "")
            + (f"  However, we DO have other evidence about this model: {ev}\n"
               f"  That is a different experiment and does not clear this gate on its own.\n"
               if ev else "")
            + f"  Capability here is family-dependent, not just size-dependent: at comparable "
              f"scale a 24B model in our measurements recorded lower error than a 32B one. Size "
              f"alone cannot clear this gate.\n"
              f"  Either use a validated model ({', '.join(validated_models())}), "
              f"or pass allow_unvalidated=True and verify transmission on your own data before "
              f"trusting any output."
        )
    return p


# Models measured on FULL-DATASET GENERATION but not on the transmission sweep that every entry
# in PROFILES is scored against. These are two different experiments and their numbers are NOT
# interchangeable: the sweep forces one rate in three regenerated cohorts (a large, salient signal
# in a short prompt), while full generation asks the model to honour many cohorts' histograms,
# class balances and a multi-level conditional table at once. A model can do the first while
# attending weakly to the second, and nothing in the output says which is happening -- that gap is
# exactly what made a 70B model look interchangeable with a frontier one when it is not.
#
# So these do NOT get a PROFILES entry and are NOT auto-validated. The gate still refuses them.
# What this table changes is only the error message: a user with one of these keys deserves to
# know what we actually measured rather than being told "unmeasured" with no further detail.
MEASURED_ON_GENERATION_ONLY: dict[str, str] = {
    "gpt-5": ("measured on full-dataset generation at DEFAULT reasoning: conditional error 0.045, "
              "inside the 0.041-0.047 band of the frontier models we validated, and below what a "
              "real 300-record sample achieves (0.062). At reasoning_effort='minimal' the same "
              "model scored 0.173 -- worse than a no-information control. Not scored on the "
              "transmission sweep, so its PROFILES entry carries no magnitude error."),
    "gemini-3.1-pro": ("measured on full-dataset generation with reasoning on: conditional error "
                       "0.041, the lowest of any generator we measured. Not scored on the "
                       "transmission sweep, so its PROFILES entry carries no magnitude error."),
    "claude-opus-5": ("measured on full-dataset generation at default reasoning: conditional error "
                      "0.046. Matched by the `claude.*opus` PROFILES entry on the transmission "
                      "sweep."),
}


def generation_evidence_for(model: str) -> str | None:
    for stem, note in MEASURED_ON_GENERATION_ONLY.items():
        if str(model).lower().startswith(stem):
            return note
    return None


# The RECOMMENDATION, and it is a list of models rather than a list of properties. We tried to
# derive the properties that separate a working generator from a failing one and could not: four
# passing models across three vendors leaves every candidate property confounded with every other,
# "capability tier" is a vendor's label rather than a measurement, and the only numeric thresholds
# available come from the transmission sweep, which mispredicts generation. So we name what we
# measured. An unlisted model is refused unless the caller explicitly opts in.
#
# Conditional error on full-dataset generation, against a real 300-record sample (0.062) and a
# permuted-target no-information floor (0.152).
RECOMMENDED: dict[str, str] = {
    "gemini-3.1-pro":  "0.041 — lowest of any generator measured; reasoning on",
    "gpt-5":           "0.045 at DEFAULT reasoning. At reasoning_effort='minimal' the same model "
                       "scores 0.173, past the no-information floor — set effort explicitly",
    "claude-fable-5":  "0.046; reasoning on",
    "claude-opus-5":   "0.046, reached with reasoning verifiably NOT firing",
    "gemini-3.5-flash": "0.008 on the transmission sweep; the cost-efficient option of the set",
}


def validated_models() -> list[str]:
    """The models we recommend: measured on FULL-DATASET generation, which is what this tool runs.

    Two self-hosted 70B models used to appear here on the strength of the transmission sweep alone
    (see Tier.SWEEP_ONLY). Recommending them was recommending a configuration we went on to measure
    as not working for the thing the mechanism exists to do.
    """
    return list(RECOMMENDED)


def recommended_table() -> str:
    """The recommendation, with what each model measured, for printing to an operator."""
    w = max(len(k) for k in RECOMMENDED)
    lines = ["Recommended generators (conditional error on full-dataset generation;",
             " real sample at matched n = 0.062, no-information floor = 0.152):"]
    lines += [f"  {k:<{w}}  {v}" for k, v in RECOMMENDED.items()]
    lines.append("  An unlisted model is refused unless you pass allow_unvalidated=True; "
                 "if you do, verify transmission on your own data first.")
    return "\n".join(lines)


def warnings_for(model: str) -> list[str]:
    """Non-fatal advisories to surface in the run log and the audit trail."""
    p = profile_for(model)
    out: list[str] = []
    if p.tier is Tier.ADEQUATE:
        out.append(
            f"{model} is ADEQUATE, not validated: measured magnitude error "
            f"{p.magnitude_error:.3f}. Direction of relationships is reliable; absolute rates "
            f"carry error of roughly this size. Do not use it for a regulated rate without "
            f"verifying transmission on your own data."
        )
    if p.tier is Tier.UNKNOWN:
        ev = generation_evidence_for(model)
        out.append(
            f"{model} is UNVALIDATED and running under an explicit override. Verify transmission "
            f"on your own data before trusting the output."
            + (f" Note: {ev}" if ev else "")
        )
    if re.search(r"gpt-oss|deepseek-r1|.*thinking", model, flags=re.IGNORECASE):
        out.append(
            "this looks like a reasoning model: its chain of thought is billed against the same "
            "output budget as the CSV, and an exhausted budget returns empty content that looks "
            "like an inability to follow the schema. Output budget is raised automatically."
        )
    return out
