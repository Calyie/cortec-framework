"""
Regression tests named after the actual historical defects.

Every test here corresponds to a real bug that produced a plausible but WRONG conclusion during
the research behind this tool. They are named after the bug rather than the function, because the
thing that must not regress is the failure, not the implementation. If any of these fail, the tool
has lost a guarantee its own research proved was necessary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cortec.accounting import PrivacyLedger, PrivacyAccountingError, histogram_sensitivity
from cortec.generate import (Generator, ContextTruncationError, EmptyContentError,
                             GenerationError, allocate_rows)
from cortec.models import ModelCapabilityError, check_model, Tier, profile_for
from cortec.prompts import PromptIntegrityError, render_cohort_prompt, verify_integrity
from cortec.release import Release, release_statistics, ReleaseError
from cortec.schema import Schema, Band, DataValidationError, SchemaError


# ── fixtures ────────────────────────────────────────────────────────────────────────

def demo_schema() -> Schema:
    return Schema(
        name="demo",
        numerical={"age": (18, 90), "hours": (1, 99)},
        categorical={"edu": ["HS", "College", "Grad"], "result": ["None", "Norm", "High"]},
        target="y", positive="YES", negative="NO",
        bins={"age": [18, 30, 45, 60, 90], "hours": [1, 20, 40, 60, 99]},
        stratify=[("edu", [])],
        conditional=[("edu",), ("edu", "result")],
    )


def demo_frame(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edu = rng.choice(["HS", "College", "Grad"], size=n, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "hours": rng.integers(1, 99, n),
        "edu": edu,
        # 'None' is a REAL category here, exactly as in clinical data
        "result": rng.choice(["None", "Norm", "High"], size=n, p=[0.6, 0.25, 0.15]),
        "y": np.where(rng.random(n) < np.where(edu == "Grad", 0.6, 0.2), "YES", "NO"),
    })


# ── 1. the privacy under-charge ─────────────────────────────────────────────────────

def test_privacy_undercharge_is_impossible__multi_level_conditional_charges_every_level():
    """A multi-level conditional release once charged only ONE level's epsilon while releasing
    all of them: a 7x under-charge, i.e. a false privacy claim. Levels overlap (they describe the
    same people) and must compose sequentially."""
    led = PrivacyLedger(2.0)
    for level in range(7):
        led.spend(f"L{level}", kind="bounded_mean", epsilon=0.1, sensitivity=0.01,
                  composition="parallel", partition="c0", group=f"conditional_L{level}")
    # 7 levels x 0.1 must total 0.7, not 0.1
    assert led.total_epsilon == pytest.approx(0.7), \
        "levels must compose sequentially with one another"


def test_budget_overrun_is_refused_before_anything_is_released():
    led = PrivacyLedger(1.0)
    led.spend("a", kind="histogram", epsilon=0.9, sensitivity=1.0,
              composition="sequential", partition="p", group="g")
    with pytest.raises(PrivacyAccountingError, match="Nothing was released"):
        led.spend("b", kind="histogram", epsilon=0.5, sensitivity=1.0,
                  composition="sequential", partition="p", group="g")
    assert len(led.queries) == 1, "the refused query must not appear in the ledger"


def test_parallel_composition_is_max_not_sum_across_disjoint_cohorts():
    """The rule the whole budget allocation rests on. Disjoint cohorts cost the max."""
    led = PrivacyLedger(2.0)
    for cohort in range(12):
        led.spend(f"h{cohort}", kind="histogram", epsilon=0.1, sensitivity=1.0,
                  composition="parallel", partition=f"cohort{cohort}", group="marginals")
    assert led.total_epsilon == pytest.approx(0.1), \
        "12 disjoint cohorts must cost one cohort's worth, not 12x"


def test_histogram_sensitivity_is_one_regardless_of_bin_count():
    assert histogram_sensitivity(2) == 1.0
    assert histogram_sensitivity(500) == 1.0


def test_ledger_refuses_further_queries_once_sealed():
    """After the release is emitted, the private data must not be touched again."""
    led = PrivacyLedger(1.0)
    led.spend("a", kind="histogram", epsilon=0.1, sensitivity=1.0,
              composition="parallel", partition="p", group="g")
    led.seal()
    with pytest.raises(PrivacyAccountingError, match="sealed"):
        led.spend("b", kind="histogram", epsilon=0.1, sensitivity=1.0,
                  composition="parallel", partition="p", group="g")


# ── 2. context truncation ───────────────────────────────────────────────────────────

def test_ollama_context_truncation_detected():
    __import__("pytest").importorskip("httpx")  # the Ollama transport is an optional extra
    """Ollama's default context silently truncated the prompt, cutting off the conditional table.
    It looked like 'this model family ignores the DP statistics'."""
    # acknowledge_insufficient because this model is now gated as SWEEP_ONLY; this test is
    # about context truncation, not capability, and needs a concrete ollama model to construct.
    g = Generator(demo_schema(), backend="ollama", model="llama3.3:70b-instruct",
                  ollama_url="http://localhost:1", acknowledge_insufficient=True)
    g.num_ctx = 4096                      # the old default that caused this
    with pytest.raises(ContextTruncationError, match="truncat"):
        g._assert_prompt_fits("x" * 200_000)


def test_context_check_accounts_for_the_output_budget_too():
    __import__("pytest").importorskip("httpx")  # the Ollama transport is an optional extra
    # acknowledge_insufficient because this model is now gated as SWEEP_ONLY; this test is
    # about context truncation, not capability, and needs a concrete ollama model to construct.
    g = Generator(demo_schema(), backend="ollama", model="llama3.3:70b-instruct",
                  ollama_url="http://localhost:1", acknowledge_insufficient=True)
    g.num_ctx = 8192
    g.max_output_tokens = 8192            # output alone fills the window
    with pytest.raises(ContextTruncationError):
        g._assert_prompt_fits("x" * 4000)


# ── 3. reasoning model returning empty content ──────────────────────────────────────

def test_reasoning_model_empty_content_detected(monkeypatch):
    __import__("pytest").importorskip("httpx")  # the Ollama transport is an optional extra
    """gpt-oss:20b spent its entire output budget on hidden reasoning and returned content="".
    It looked like 'this model cannot follow the CSV schema'."""
    g = Generator(demo_schema(), backend="ollama", model="gpt-oss:20b",
                  ollama_url="http://localhost:1")

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": "", "thinking": "x" * 9185},
                    "done_reason": "length"}

    monkeypatch.setattr(g._http, "post", lambda *a, **k: FakeResp())
    with pytest.raises(EmptyContentError, match="output budget"):
        g._call("short prompt")
    assert g.stats.empty_content == 1


def test_reasoning_model_gets_a_larger_output_budget_and_a_warning():
    g = Generator(demo_schema(), backend="mock", model="gpt-oss:20b")
    assert g.max_output_tokens >= 8192
    assert any("reasoning model" in w for w in g.stats.warnings)


# ── 4. 'None' read as missing ───────────────────────────────────────────────────────

def test_none_string_is_a_category_not_a_missing_value():
    """pandas' default NA strings include 'None'. A clinical result of 'None' means the test was
    not ordered; reading it as missing deleted 83% of generated rows at a 100% call-success rate."""
    s = demo_schema()
    g = Generator(s, backend="mock", model="claude-fable-5")
    csv = ("age,hours,edu,result,y\n"
           "40,40,HS,None,NO\n"
           "50,45,Grad,None,YES\n"
           "60,20,College,Norm,NO\n")
    df = g.parse(csv)
    assert df is not None and len(df) == 3, "rows with result='None' must survive"
    assert (df["result"] == "None").sum() == 2


def test_schema_validation_accepts_none_as_a_declared_category():
    s = demo_schema()
    df = demo_frame()
    rep = s.validate(df, strict=True)
    assert rep["ok"]
    assert not any("result" in w and "not in the declared set" in w for w in rep["warnings"])


# ── 5. out-of-domain rows accepted ──────────────────────────────────────────────────

def test_rows_outside_the_public_domain_are_rejected():
    """A degenerate response produced a credit limit of 34 against a declared floor of 10,000 and
    an age of 0 against a floor of 21. Those rows entered the dataset silently."""
    g = Generator(demo_schema(), backend="mock", model="claude-fable-5")
    csv = ("age,hours,edu,result,y\n"
           "40,40,HS,None,NO\n"
           "0,40,HS,None,NO\n"        # age below the declared floor of 18
           "200,40,HS,None,NO\n")     # age above the declared ceiling of 90
    df = g.parse(csv)
    assert len(df) == 1
    assert g.stats.rows_out_of_bounds == 2


# ── 6. cohort/cell support ──────────────────────────────────────────────────────────

def test_cohort_below_n_min_rejected():
    """A rate over too few records is dominated by its own noise. n_min is a floor, not a knob."""
    s = demo_schema()
    with pytest.raises(ReleaseError, match="below the safe floor"):
        release_statistics(s, demo_frame(), epsilon_total=2.0, n_min=10)


def test_release_fails_loudly_when_no_cohort_has_support():
    s = demo_schema()
    with pytest.raises(ReleaseError, match="no cohort reached"):
        release_statistics(s, demo_frame(n=200), epsilon_total=2.0, n_min=5000)


def test_empty_conditional_table_is_refused_rather_than_shipped():
    """An empty conditional table means the output is unconditioned — the exact failure the
    method exists to avoid — so it must never be rendered into a prompt."""
    with pytest.raises(PromptIntegrityError, match="only channel"):
        render_cohort_prompt(
            cohort_name="c", n_rows=5, numeric_block="x", categorical_block="x",
            class_balance_block="x", conditional_block="   ", columns="a,b",
            target="y", positive="YES", negative="NO")


# ── 7. uniform row allocation ───────────────────────────────────────────────────────

def test_row_allocation_is_proportional_not_uniform():
    """Uniform allocation over-represented a 2%-of-population cohort by 12x, so every marginal
    measured afterwards described a deliberately wrong mixture."""
    rel = Release(schema_name="demo", epsilon_total=2.0, n_min=150, cohorts=[
        {"cohort_id": 0, "cohort_name": "tiny", "cohort_size": 200},
        {"cohort_id": 1, "cohort_name": "big", "cohort_size": 9800},
    ])
    alloc = allocate_rows(rel, 1000)
    assert sum(alloc) == 1000, "allocation must total exactly what was asked for"
    assert alloc[0] == pytest.approx(20, abs=1), "2% cohort gets ~2% of rows, not 50%"
    assert alloc[1] == pytest.approx(980, abs=1)


# ── 8. model capability gating ──────────────────────────────────────────────────────

def test_model_below_capability_floor_is_refused():
    """A 7B model scored a transmission slope of 0.21 while emitting perfectly valid CSV. It
    fails silently, so it is gated rather than warned about."""
    with pytest.raises(ModelCapabilityError, match="below the capability floor"):
        check_model("qwen2.5:7b-instruct")
    assert profile_for("qwen2.5:7b-instruct").tier is Tier.INSUFFICIENT


def test_unknown_model_is_refused_unless_explicitly_allowed():
    with pytest.raises(ModelCapabilityError, match="not been measured"):
        check_model("some-new-model:13b")
    p = check_model("some-new-model:13b", allow_unvalidated=True)
    assert p.tier is Tier.UNKNOWN


def test_capability_gate_is_not_decided_by_size_alone():
    """A 24B model beat a 32B one in our measurements, so a size threshold would be wrong."""
    assert profile_for("mistral-small:24b-instruct").magnitude_error < \
           profile_for("qwen2.5:32b-instruct").magnitude_error


def test_generator_refuses_insufficient_model_before_any_call_is_made():
    with pytest.raises(ModelCapabilityError):
        Generator(demo_schema(), backend="mock", model="qwen2.5:7b-instruct")


# ── 9. prompt integrity ─────────────────────────────────────────────────────────────

def test_prompt_template_is_hash_locked(monkeypatch):
    """Removing the conditional table or the counterintuitive instruction collapses the mechanism
    to header-only, while the pipeline keeps running and producing plausible rows."""
    verify_integrity()
    import cortec.prompts as P
    monkeypatch.setattr(P, "COHORT_TEMPLATE", "a weakened template")
    monkeypatch.setitem(P._LOCKED, "cohort", ("a weakened template", P._LOCKED["cohort"][1]))
    with pytest.raises(PromptIntegrityError, match="modified in memory"):
        P.verify_integrity()


def test_locked_template_cannot_be_silently_replaced():
    import cortec.prompts as P
    with pytest.raises(PromptIntegrityError, match="locked template"):
        P.register_template("cohort", "something else")


def test_system_prompt_keeps_the_counterintuitive_instruction():
    from cortec.prompts import system_prompt
    assert "EVEN IF THIS CONTRADICTS YOUR EXPECTATIONS" in system_prompt()


# ── validation happens before budget is spent ───────────────────────────────────────

def test_bad_data_is_rejected_before_any_budget_is_spent():
    """Budget cannot be refunded, so validation must precede the first query."""
    s = demo_schema()
    df = demo_frame().drop(columns=["edu"])
    with pytest.raises(DataValidationError, match="NO privacy budget was spent"):
        release_statistics(s, df, epsilon_total=2.0)


def test_constant_target_is_rejected():
    s = demo_schema()
    df = demo_frame()
    df["y"] = "NO"
    with pytest.raises(DataValidationError, match="only one class"):
        s.validate(df, strict=True)


# ── end-to-end accounting ───────────────────────────────────────────────────────────

def test_end_to_end_release_never_exceeds_the_declared_budget():
    s = demo_schema()
    rel = release_statistics(s, demo_frame(), epsilon_total=2.0, n_min=150, seed=7)
    assert rel.audit["within_budget"]
    assert rel.audit["epsilon_accounted"] <= 2.0 + 1e-9
    assert rel.n_cohorts >= 1
    assert rel.conditional_levels, "the conditional table must not be empty"


def test_generation_is_free_and_repeatable():
    """Unlimited draws from one release, at no further privacy cost — the central claim."""
    s = demo_schema()
    rel = release_statistics(s, demo_frame(), epsilon_total=2.0, n_min=150, seed=7)
    spent = rel.audit["epsilon_accounted"]
    g = Generator(s, backend="mock", model="claude-fable-5")
    a = g.generate(rel, 60, verbose=False)
    b = g.generate(rel, 60, verbose=False)
    assert len(a) > 0 and len(b) > 0
    assert rel.audit["epsilon_accounted"] == spent, "generation must not change the accounting"


def test_release_spends_exactly_the_declared_budget_not_merely_less():
    """An under-charge is a FALSE PRIVACY CLAIM, and 'within budget' does not catch it.

    Regression: the suppression query reads every record, so it is not disjoint from the
    per-cohort queries. Filing it in the 'marginals' group put it inside the parallel max(),
    where a larger cohort cost absorbed it — the release then reported 1.90 against a declared
    2.00 and still passed every 'within_budget' assertion.
    """
    s = demo_schema()
    rel = release_statistics(s, demo_frame(), epsilon_total=2.0, n_min=150, seed=11)
    assert rel.audit["epsilon_accounted"] == pytest.approx(2.0, abs=1e-9), (
        "the release must account for the full declared budget; a shortfall means some query's "
        "cost was silently absorbed by a composition rule that does not apply to it")


def test_a_query_over_all_records_is_not_absorbed_by_a_parallel_max():
    """The general form of the bug above, stated directly on the ledger."""
    led = PrivacyLedger(2.0)
    for c in range(5):
        led.spend(f"cohort{c}", kind="histogram", epsilon=0.9, sensitivity=1.0,
                  composition="parallel", partition=f"c{c}", group="marginals")
    led.spend_suppression_counts(epsilon=0.1)      # own group, reads everything
    assert led.total_epsilon == pytest.approx(1.0), \
        "a whole-dataset query must ADD to the disjoint cohort cost, not hide inside its max"


def test_audit_report_is_recomputable_by_hand():
    """A compliance reader must be able to re-derive the total from the report alone."""
    s = demo_schema()
    rel = release_statistics(s, demo_frame(), epsilon_total=2.0, n_min=150, seed=1)
    r = rel.audit
    total = sum(g["group_total"] for g in r["groups"].values())
    assert total == pytest.approx(r["epsilon_accounted"])
    for g in r["groups"].values():
        # by hand, INCLUDING nested partitions: a key `cohort/class` is a subset of `cohort`,
        # so its records are also touched by the parent's queries -- the lineage sum is the cost
        # of that partition; disjoint lineages take the max
        per = g["parallel_by_partition"]
        lineage = {k: sum(per.get("/".join(k.split("/")[:i + 1]), 0.0)
                          for i in range(len(k.split("/")))) for k in per}
        expected = g["sequential_sum"] + (max(lineage.values()) if lineage else 0.0)
        assert g["group_total"] == pytest.approx(expected)


# ── 10. cell-rate expressibility (the saturation defect) ────────────────────────────

def test_positive_count_rounding_is_stochastic_not_biased():
    """Deterministic rounding turns a released 0.554 over one row into 1.000 EVERY time, which is
    the saturation that made generate() score MAE 0.253 with slope 1.64 on real data. Stochastic
    rounding keeps the expected rate equal to the released rate."""
    from cortec.generate import positives_for_cell
    rng = np.random.default_rng(0)
    draws = [positives_for_cell(0.554, 1, rng) for _ in range(4000)]
    assert set(draws) == {0, 1}, "a single row can only be 0 or 1 positive"
    mean = float(np.mean(draws))
    assert abs(mean - 0.554) < 0.03, (
        f"expected rate {mean:.3f} should match the released 0.554; a deterministic round would "
        f"give 1.000 and reproduce the saturation defect")


def test_positive_count_is_exact_when_the_rate_is_expressible():
    from cortec.generate import positives_for_cell
    rng = np.random.default_rng(1)
    assert positives_for_cell(0.5, 10, rng) == 5
    assert positives_for_cell(0.0, 9, rng) == 0
    assert positives_for_cell(1.0, 9, rng) == 9


def test_positive_count_never_exceeds_the_row_count():
    from cortec.generate import positives_for_cell
    rng = np.random.default_rng(2)
    for r in (0.0, 0.1, 0.554, 0.999, 1.0):
        for n in (1, 3, 12):
            k = positives_for_cell(r, n, rng)
            assert 0 <= k <= n


def test_cell_prompt_states_an_integer_count_not_a_probability():
    """The whole point of the cell path: the model is told a count, so it has no discretion over
    the target column."""
    from cortec.prompts import render_cell_prompt
    text, rec = render_cell_prompt(
        cell_description="  education = Masters", n_rows=12, n_positive=7,
        numeric_block="  age: 25-60", categorical_block="  occupation: Exec 40%",
        columns="age,education,income", target="income",
        positive=">50K", negative="<=50K")
    assert "EXACTLY 7" in text and "hard count, not a probability" in text
    assert "remaining 5" in text
    assert rec.template == "cell"


def test_cell_prompt_rejects_an_impossible_count():
    from cortec.prompts import render_cell_prompt
    with pytest.raises(PromptIntegrityError, match="not a valid count"):
        render_cell_prompt(cell_description="x", n_rows=5, n_positive=9,
                           numeric_block="", categorical_block="", columns="a",
                           target="y", positive="Y", negative="N")


def test_cell_wise_generation_reproduces_released_rates_end_to_end():
    """Mock backend honours the count instruction, so this asserts the plumbing carries the rate
    from the release through to the output."""
    s = demo_schema()
    rel = release_statistics(s, demo_frame(n=4000), epsilon_total=2.0, n_min=150, seed=3)
    g = Generator(s, backend="mock", model="claude-fable-5", rows_per_call=10, seed=5)
    out = g.generate_by_cell(rel, 200, level=0, verbose=False)
    assert len(out) > 0
    assert "_cell" in out.columns, "each record must record the cell it was generated for"


# ── autoconfig: derived by default, hand-tuning still available ─────────────────────

def _demo_frame(n=4000, seed=0):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"age": rng.integers(20, 80, n),
                       "income": rng.integers(10_000, 120_000, n),
                       "region": rng.choice(list("ABCDE"), n),
                       "tier": rng.choice(["bronze", "silver", "gold"], n, p=[.5, .3, .2])})
    p = 0.05 + 0.5 * (df.tier == "gold") + 0.2 * (df.age > 60)
    df["churn"] = np.where(rng.random(n) < p.clip(0, 1), "YES", "NO")
    return df


def _demo_schema(**over):
    from cortec import Schema
    base = dict(name="demo", numerical={"age": (18, 90), "income": (0, 200_000)},
                categorical={"region": list("ABCDE"), "tier": ["bronze", "silver", "gold"]},
                target="churn", positive="YES", negative="NO",
                bins={"age": [18, 30, 45, 60, 90],
                      "income": [0, 30_000, 60_000, 100_000, 200_000]})
    base.update(over)
    return Schema(**base)


def test_autoconfig_is_the_default_and_produces_a_hierarchy():
    """An empty `conditional` must be derived, not silently released unconditioned."""
    from cortec import release_statistics
    r = release_statistics(_demo_schema(), _demo_frame(), epsilon_total=2.0, n_min=50, seed=0)
    assert len(r.conditional_levels) > 1, "autoconfig did not derive any conditional level"


def test_declared_hierarchy_is_honoured_over_autoconfig():
    """Hand-tuning stays available: a declared hierarchy is used exactly as given."""
    from cortec import release_statistics
    declared = [(), ("tier",), ("tier", "age")]
    r = release_statistics(_demo_schema(conditional=declared), _demo_frame(),
                           epsilon_total=2.0, n_min=50, seed=0)
    assert len(r.conditional_levels) == len(declared)


def test_selection_budget_is_charged_not_free():
    """The selection reads private data. If it were free the release would overspend silently."""
    from cortec import release_statistics
    r = release_statistics(_demo_schema(), _demo_frame(), epsilon_total=2.0, n_min=50, seed=0)
    assert r.epsilon_total == 2.0
    spent = sum(q.get("epsilon", 0.0) for q in getattr(r, "ledger_entries", []) or [])
    if spent:
        assert spent <= 2.0 + 1e-9, f"release spent {spent} against a declared budget of 2.0"


def test_opting_out_without_declaring_fails_loudly():
    """autoconfig=False and no declared hierarchy must error, not release an unconditioned table."""
    import pytest
    from cortec import release_statistics
    from cortec.release import ReleaseError
    with pytest.raises(ReleaseError, match="autoconfig"):
        release_statistics(_demo_schema(), _demo_frame(), epsilon_total=2.0, n_min=50,
                           seed=0, autoconfig=False)


def test_published_counts_are_noised_not_exact_private_counts():
    """Regression: `cohort_size` and cell `support` were emitted as EXACT private counts.

    The `rate` beside `support` was correctly noised and charged; the counts were neither. Because
    a ledger can only sum the queries it is told about, the audit still reported a clean eps=2.0
    while the release published the true number of records in every cohort and every cell. A
    verifier that totals declared queries cannot detect an undeclared one, so this is asserted
    against the OUTPUT rather than against the accounting.

    Noise is detected by re-releasing the same data under different seeds: an unnoised count is
    deterministic, a Laplace-noised one is not.
    """
    s, df = demo_schema(), demo_frame()
    runs = []
    for seed in (1, 2, 3, 4):
        rel = release_statistics(s, df, epsilon_total=2.0, n_min=150, seed=seed)
        runs.append((
            [c["cohort_size"] for c in rel.cohorts],
            [v["support"] for lv in rel.conditional_levels for v in lv["cells"].values()],
        ))

    assert any(r[0] != runs[0][0] for r in runs), (
        "cohort_size is identical across seeds, so it is not noised — the release is publishing "
        "exact counts of private records")
    assert any(r[1] != runs[0][1] for r in runs), (
        "cell support is identical across seeds, so it is not noised")

    assert all(v >= 0 for r in runs for v in r[0] + r[1]), "counts must stay non-negative"


def test_published_counts_are_charged_to_the_ledger():
    """The counts above must also APPEAR in the accounting, in their own groups.

    Cohort counts and cell counts each partition the data (parallel within a group) but overlap
    each other and across levels (a record is in one cohort, and in one cell of every level), so
    they must be filed as separate groups that compose sequentially. Filing them together lets the
    parallel max() absorb one and under-charge — a false privacy claim.
    """
    rel = release_statistics(demo_schema(), demo_frame(), epsilon_total=2.0, n_min=150, seed=11)
    groups = rel.audit["groups"]
    assert "published_counts_cohorts" in groups, "cohort-size queries are not in the ledger"
    assert any(k.startswith("published_counts_cells") for k in groups), (
        "cell-support queries are not in the ledger")
    n_levels = len(rel.conditional_levels)
    assert sum(k.startswith("published_counts_cells") for k in groups) == n_levels, (
        "each conditional level's cell counts need their own group: levels overlap, so one "
        "shared group would collapse them into a single parallel max() and under-charge")
    assert rel.audit["epsilon_accounted"] == pytest.approx(2.0, abs=1e-9)


def _leaf_fields(obj, prefix=""):
    """Flatten a nested release structure to {dotted.path: scalar}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_leaf_fields(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_leaf_fields(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


# Numeric fields that are legitimately identical across seeds because they carry no private
# information: positional indices, and bin edges/bounds that come from the DECLARED public schema.
_PUBLIC_NUMERIC = ("cohort_id", ".level", ".bin_edges[", ".bounds[")


def test_no_unnoised_private_quantity_survives_in_the_release():
    """Property test, general form of the `support`/`cohort_size` leak.

    Every number in a release that is derived from the private data must be noised, and a noised
    number changes when the release is re-run at a different seed on identical data. So: release
    the same frame under several seeds, and any numeric leaf that is byte-identical every time is
    either public (an index, or a declared bin edge) or an unnoised private quantity.

    This is deliberately a property over the OUTPUT rather than a check of the ledger. The bug this
    generalises was invisible to the accounting: the leaked counts were never entered into the
    ledger, and a ledger can only total the queries it is told about.
    """
    s, df = demo_schema(), demo_frame()
    snaps = []
    for seed in (1, 2, 3, 4, 5):
        rel = release_statistics(s, df, epsilon_total=2.0, n_min=150, seed=seed)
        snaps.append(_leaf_fields({"cohorts": rel.cohorts, "levels": rel.conditional_levels}))

    suspects = []
    for k, v0 in snaps[0].items():
        if isinstance(v0, bool) or not isinstance(v0, (int, float)):
            continue
        # a share clipped at the [0, 1] boundary can be identical across seeds without being
        # unnoised (a zero count noised and clipped to zero half the time): exclude the boundary
        # values, as the research pipeline's test does, rather than read a clip as a leak
        if v0 in (0, 0.0, 1, 1.0):
            continue
        if any(tag in k for tag in _PUBLIC_NUMERIC):
            continue
        if all(sn.get(k) == v0 for sn in snaps):
            suspects.append((k, v0))

    assert not suspects, (
        "these numeric release fields are identical across 5 seeds, so they are not noised — "
        "each is either an unnoised private quantity or a public value that belongs in "
        f"_PUBLIC_NUMERIC: {suspects[:10]}")


def test_autoconfig_path_reports_unspent_conditional_budget(capsys):
    """Regression: with AUTOCONFIG on, part of the conditional budget was silently never spent.

    autoconfig derives k columns and the conditional budget is split k+1 ways up front. A level
    whose cells all fall below n_min releases nothing, so its share evaporates: heart_cleveland
    declared epsilon=2.0 and accounted 1.63, german_credit 1.71. The release is then noisier than
    the caller paid for, and worst on small data where noise already dominates.

    `test_release_spends_exactly_the_declared_budget_not_merely_less` could not catch this: it uses
    a DECLARED schema, which skips autoconfig entirely. **A test that exercises one code path says
    nothing about the other.**
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 240                      # small on purpose: finer levels cannot reach n_min
    df = pd.DataFrame({
        "age": rng.integers(20, 80, n),
        "score": rng.normal(50, 10, n).round(1),
        "a": rng.choice(["p", "q", "r"], n),
        "b": rng.choice(["s", "t"], n),
        "y": rng.choice(["YES", "NO"], n, p=[0.35, 0.65]),
    })
    sc = Schema(name="tiny-auto", numerical={"age": (18, 90), "score": (0, 100)},
                categorical={"a": ["p", "q", "r"], "b": ["s", "t"]},
                target="y", positive="YES", negative="NO",
                bins={"age": [18, 40, 60, 90], "score": [0, 40, 60, 100]})
    rel = release_statistics(sc, df, epsilon_total=2.0, n_min=150, seed=3, autoconfig=True)

    spent = rel.audit["epsilon_accounted"]
    if spent < 2.0 - 1e-9:
        out = capsys.readouterr().out
        assert "was NOT spent" in out, (
            f"accounted only {spent:.3f} of 2.0 and said nothing — an unspent budget makes the "
            f"release noisier than declared and must be reported, not left in the audit")


# ── enterprise backend support ──────────────────────────────────────────────────────
# The tool previously had code paths for Anthropic and Ollama only, while its capability table
# vouched for a Gemini tier it could not call and said nothing about OpenAI. A user with either of
# those keys could not run the tool at all -- including on the generator most of the research
# behind it was produced with.

def test_every_enterprise_backend_is_constructible():
    """Refuses to regress to an Anthropic-only tool.

    Constructs each backend with a dummy key. No call is made, so this costs nothing and runs
    offline; it fails if a backend is missing from the dispatch or its client cannot be built.
    """
    import os
    import pytest
    from cortec.generate import Generator, GenerationError
    from cortec.schema import Schema

    schema = Schema(name="t", numerical={"a": (0.0, 10.0)}, categorical={"b": ["x", "y"]},
                    target="y_", positive="YES", negative="NO", bins={"a": [0, 5, 10]},
                    stratify=[], conditional=[])
    for backend, model, env in (("anthropic", "claude-fable-5", "ANTHROPIC_API_KEY"),
                                ("openai", "gpt-5", "OPENAI_API_KEY"),
                                ("gemini", "gemini-3.5-flash", "GEMINI_API_KEY")):
        prev = os.environ.get(env)
        os.environ[env] = "test-key-not-used"
        try:
            try:
                g = Generator(schema, backend=backend, model=model, allow_unvalidated=True)
            except ImportError:
                pytest.skip(f"{backend} SDK not installed in this environment")
            except GenerationError as e:
                if "pip install cortec[" in str(e):
                    pytest.skip(f"{backend} SDK not installed in this environment: {e}")
                raise
            assert g.backend == backend
            assert g.reasoning == "on", "reasoning must default ON: it is worth 3.8x on conditional error"
        finally:
            if prev is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = prev


def test_an_unknown_backend_names_the_supported_ones():
    from cortec.generate import Generator, GenerationError
    from cortec.schema import Schema
    schema = Schema(name="t", numerical={"a": (0.0, 10.0)}, categorical={"b": ["x"]},
                    target="y_", positive="YES", negative="NO", bins={"a": [0, 10]},
                    stratify=[], conditional=[])
    try:
        Generator(schema, backend="bedrock-but-misspelled", model="claude-fable-5")
        raise AssertionError("an unknown backend must be refused")
    except GenerationError as e:
        for expected in ("anthropic", "openai", "gemini", "ollama"):
            assert expected in str(e), f"{expected!r} missing from the error message"


def test_price_table_uses_longest_prefix_not_family_prefix():
    """A cheap tier must not be billed at its family's frontier rate.

    A prefix match that fired on the family stem charged one model 6.7x its true rate, which
    tripped its budget cap at roughly a tenth of the spend the operator had authorised.
    """
    import os
    from cortec.generate import Generator
    from cortec.schema import Schema
    import pytest
    schema = Schema(name="t", numerical={"a": (0.0, 10.0)}, categorical={"b": ["x"]},
                    target="y_", positive="YES", negative="NO", bins={"a": [0, 10]},
                    stratify=[], conditional=[])
    pytest.importorskip("google.genai")
    os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
    try:
        flash = Generator(schema, backend="gemini", model="gemini-3.5-flash")
    except ImportError:
        pytest.skip("google-genai not installed")
    assert (flash._price_in, flash._price_out) == (0.30, 2.50), \
        "a flash tier billed at pro rates: the longest-prefix rule is not being applied"


def test_a_depleted_account_is_fatal_not_a_parse_failure():
    """A billing/quota refusal must abort immediately, not look like a schema problem.

    Surfaced through the generic retry path, a depleted account reports '0% of calls produced
    usable rows' -- which sends the operator to inspect their schema while the real cause is
    billing, and spends the retry budget chasing a condition that cannot improve.
    """
    from cortec.generate import _classify_api_error, FatalAPIError

    # Each string isolates ONE marker. An earlier version of this test used realistic multi-marker
    # messages -- the observed Gemini refusal contains both "RESOURCE_EXHAUSTED" and "credits" --
    # so it still passed with a marker deleted, i.e. it could not fail for the reason it claimed.
    # Mutation-check this test by removing any single marker below; exactly one case must fail.
    # "resource exhausted" was a fatal marker until a live Vertex run showed it is how that
    # surface spells a transient rate limit ("Resource exhausted. Please try again later.");
    # it is now waited out (see test_a_transient_rate_limit_is_waited_out...) unless the message
    # also names a cap that will not clear by waiting
    from cortec.generate import RateLimitedError
    assert isinstance(_classify_api_error("Gemini", RuntimeError("ClientError: 429 RESOURCE_EXHAUSTED for this project")), RateLimitedError)
    # a daily cap is not transient, so it is not waited out either; it goes through the ordinary retry path
    assert not isinstance(_classify_api_error("Gemini", RuntimeError("ClientError: the resource exhausted its daily allocation")), RateLimitedError)
    isolated = {
        "credit":             "AccountError: your credit balance is zero",
        "billing":            "SetupError: enable billing for this project",
        "quota":              "RateLimitError: daily quota reached for this model",
        "exceeded your current": "PlanError: you exceeded your current plan",
        "usage limit":        "PlatformError: monthly usage limit reached",
        "invalid api key":    "AuthError: the supplied invalid api key was rejected",
        "invalid_api_key":    "OpenAIError: error code invalid_api_key",
        "api key not valid":  "ClientError: API key not valid, pass a valid one",
        "authentication":     "AuthnError: authentication failed",
        "unauthorized":       "HTTPError: 401 unauthorized",
        "permission":         "PermissionDeniedError: the caller lacks permission",
        "payment":            "HTTPError: 402 payment required",
        "account is not active": "StatusError: account is not active",
        # observed live on Vertex AI: a model served only from `global`, requested regionally
        "not_found":          "ClientError: 404 NOT_FOUND. Publisher model gemini-x was rejected",
        "does not exist":     "NotFoundError: 404 the model `gpt-x` does not exist",
        # credential failures, each observed or documented by the vendor
        "invalid_grant":      "RefreshError: ('invalid_grant: Invalid grant', {'error': 'invalid_grant'})",
        "unauthenticated":    "ClientError: 401 UNAUTHENTICATED. Request had an invalid token",
        "forbidden":          "HTTPStatusError: 403 Forbidden for this resource",
        "account not found":  "RefreshError: Invalid grant: account not found",
    }
    # Self-enforcing: each probe must hit EXACTLY the marker it is named for. Without this the
    # test silently degrades as markers are added -- which is how the first version of it came to
    # pass with a marker deleted.
    from cortec.generate import _FATAL_MARKERS

    # No marker may be subsumed by a shorter one: the shorter always matches first, so the longer
    # is unreachable and would sit in the list looking like coverage it does not provide.
    subsumed = {a: [b for b in _FATAL_MARKERS if b != a and b in a] for a in _FATAL_MARKERS}
    assert not any(subsumed.values()), \
        f"unreachable markers: { {a: b for a, b in subsumed.items() if b} }"
    assert set(isolated) == set(_FATAL_MARKERS), (
        f"every fatal marker needs an isolated probe; "
        f"missing {set(_FATAL_MARKERS) - set(isolated)}, extra {set(isolated) - set(_FATAL_MARKERS)}")

    for marker, msg in isolated.items():
        hit = [m for m in _FATAL_MARKERS if m in msg.lower()]
        assert hit == [marker], (
            f"probe for {marker!r} matches {hit} -- it is not isolating one marker, so deleting "
            f"{marker!r} would not make this test fail. Reword the probe.")
        assert isinstance(_classify_api_error("V", Exception(msg)), FatalAPIError), \
            f"marker {marker!r} is not being classified as fatal: {msg!r}"

    # Transient conditions must stay retryable, or a blip aborts a paid run that would recover.
    for msg in ["APITimeoutError: request timed out",
                "APIConnectionError: connection reset by peer",
                "InternalServerError: 503 service unavailable",
                "ValueError: could not parse response"]:
        assert not isinstance(_classify_api_error("V", Exception(msg)), FatalAPIError), \
            f"a transient failure was misclassified as fatal: {msg!r}"

    # And the real, observed refusal -- the one that previously surfaced as a parse failure.
    observed = ("ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
                "'Your prepayment credits are depleted.'}}")
    err = _classify_api_error("Gemini", Exception(observed))
    assert isinstance(err, FatalAPIError)
    assert "no additional privacy budget" in str(err), \
        "the operator must be told that resuming is free in epsilon"


def test_reasoning_suppression_is_an_argued_choice_and_is_warned_about():
    import os
    from cortec.generate import Generator, GenerationError
    from cortec.schema import Schema
    schema = Schema(name="t", numerical={"a": (0.0, 10.0)}, categorical={"b": ["x"]},
                    target="y_", positive="YES", negative="NO", bins={"a": [0, 10]},
                    stratify=[], conditional=[])
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    g = Generator(schema, backend="mock", model="claude-fable-5", reasoning="suppressed")
    assert any("suppressed" in w for w in g.stats.warnings), \
        "suppressing reasoning must warn: it measured 3.8x worse conditional fidelity"
    try:
        Generator(schema, backend="mock", model="claude-fable-5", reasoning="cheap")
        raise AssertionError("an unrecognised reasoning setting must be refused")
    except GenerationError:
        pass


def test_reasoning_is_verified_from_reported_tokens_not_from_the_flag():
    """Setting the flag is not evidence that reasoning ran.

    A model can report a large reasoning-token count on one batch and none on another, so a user
    who trusts the flag can ship the suppressed configuration unknowingly. (The specific pair we
    originally cited for this turned out to be an artifact -- see the test below -- but the
    principle survives it: only what the vendor reports back is evidence.)
    """
    import os
    from cortec.generate import Generator
    from cortec.schema import Schema
    schema = Schema(name="t", numerical={"a": (0.0, 10.0)}, categorical={"b": ["x"]},
                    target="y_", positive="YES", negative="NO", bins={"a": [0, 10]},
                    stratify=[], conditional=[])
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    g = Generator(schema, backend="mock", model="claude-fable-5", reasoning="on")
    g.stats.calls = 4
    g.stats.calls_without_reasoning = 4
    g._assert_reasoning_fired()
    assert any("ZERO reasoning tokens" in w for w in g.stats.warnings)

    g2 = Generator(schema, backend="mock", model="claude-fable-5", reasoning="on")
    g2.stats.calls = 4
    g2.stats.calls_without_reasoning = 1
    g2._assert_reasoning_fired()
    assert any("intermittent" in w for w in g2.stats.warnings)


# ── 7. "unmeasured" reported as "zero" ──────────────────────────────────────────────
# This one cost a published claim. A vendor SDK omits the token-details object on its STREAMING
# path, and a large output budget is what forces streaming -- so the reporting gap opens exactly
# in the configuration a reasoning model requires. Counting those calls as "zero reasoning
# tokens" produced an apparent intermittency in the model that was really an artifact of the
# transport, and a control that looked like it isolated output headroom while actually varying
# reasoning at the same time.

def _probe_schema():
    from cortec.schema import Schema
    return Schema(name="t", numerical={"a": (0.0, 10.0)}, categorical={"b": ["x"]},
                  target="y_", positive="YES", negative="NO", bins={"a": [0, 10]},
                  stratify=[], conditional=[])


def test_unreported_reasoning_count_is_not_recorded_as_zero():
    import os
    from cortec.generate import Generator
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    g = Generator(_probe_schema(), backend="mock", model="claude-fable-5", reasoning="on")
    g.stats.calls = 2
    g._note_reasoning(None)          # vendor reported nothing at all (the streaming path)
    g._note_reasoning(0)             # vendor explicitly reported zero
    assert g.stats.calls_reasoning_unmeasured == 1, (
        "a call whose reasoning was never measured must NOT be counted as a call observed to "
        "have none -- that conflation is the defect this test exists for")
    assert g.stats.calls_without_reasoning == 1

    g._assert_reasoning_fired()
    unmeasured = [w for w in g.stats.warnings if "could NOT be verified" in w]
    assert unmeasured, "an unverifiable call must be surfaced as unverifiable"
    assert "do not read this as reasoning being off" in unmeasured[0].lower()


def test_a_reasoning_block_counts_as_evidence_even_with_no_token_count():
    """Block presence is observable on both transports; the token count is not."""
    import os
    from cortec.generate import Generator
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    g = Generator(_probe_schema(), backend="mock", model="claude-fable-5", reasoning="on")
    g.stats.calls = 1
    g._note_reasoning(None, saw_block=True)
    assert g.stats.calls_with_reasoning_block == 1
    assert g.stats.calls_reasoning_unmeasured == 0, (
        "reasoning was directly observed on this call, so it is not unmeasured")
    g._assert_reasoning_fired()
    assert not any("could NOT be verified" in w for w in g.stats.warnings)


def test_anthropic_backend_actually_requests_reasoning():
    """`reasoning="on"` must reach the wire. It previously did not.

    The Anthropic branch sent no effort setting for either value of `reasoning`, so the flag was
    inert: the model ran on its default in both cases. "Suppressed" did not suppress, "on" did
    not enable, and the resulting call-to-call variation was attributed to the model rather than
    to never having asked.
    """
    import inspect
    from cortec.generate import Generator
    src = inspect.getsource(Generator._call_once)
    anthropic_branch = src.split('if self.backend == "anthropic":')[1].split('if self.backend == "openai":')[0]
    assert "output_config" in anthropic_branch, (
        "the Anthropic branch must send output_config.effort -- without it the reasoning setting "
        "never reaches the API and is silently ignored")
    assert "self.reasoning" in anthropic_branch, (
        "the effort sent must depend on the requested reasoning setting")


# ── 8. cell-wise generation silently erasing subpopulations ─────────────────────────
# The worst failure in this file, because nothing downstream detects it. Cell-wise emits rows only
# for released cells; population outside them is generated at rate ZERO. On a health-survey release
# keeping 4 cells over 25.4% of the population, the output had no Asian, Mexican-American,
# other-Hispanic or multiracial records at all -- 40% of people -- while TSTR stayed at 0.729
# against a cohort-wise 0.737. Every number a practitioner checks looked fine.

def _release_with_coverage(frac: float = None, *, bands_released: int = 5, n_bands: int = 5):
    """A release over one numeric conditioning column with `n_bands` equal-mass bands, of which
    the conditional level names only `bands_released`.

    Per-column coverage is therefore bands_released / n_bands. That is the quantity the guard
    gates on: mass in bands no released cell names is emitted at essentially rate zero.
    """
    from cortec.release import Release
    edges = [float(i) for i in range(n_bands + 1)]
    props = [1.0 / n_bands] * n_bands
    rel = Release(schema_name="demo", epsilon_total=2.0, n_min=150, cohorts=[{
        "cohort_id": 0, "cohort_name": "all", "cohort_size": 10000,
        "numerical": {"a": {"bin_edges": edges, "proportions": props,
                            "mean": 2.5, "std": 1.0, "bounds": [0.0, float(n_bands)]}},
        "categorical": {},
    }])
    rel.conditional_levels = [{
        "columns": ["a"],
        "cells": {f"a[{edges[i]:g},{edges[i+1]:g})": {"support": 10000 / max(bands_released, 1),
                                                      "rate": 0.3}
                  for i in range(bands_released)},
    }]
    return rel


def test_cellwise_refuses_a_release_that_does_not_cover_the_population():
    import os
    from cortec.generate import Generator, GenerationError
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    g = Generator(_probe_schema(), backend="mock", model="claude-fable-5")

    # 1 of 5 bands released -> 20% of the column's mass. On the release this models, that
    # configuration erased four of six racial groups.
    with pytest.raises(GenerationError, match="unsafe on this release"):
        g.generate_by_cell(_release_with_coverage(bands_released=1), n_rows=400)

    try:
        g.generate_by_cell(_release_with_coverage(bands_released=1), n_rows=400)
    except GenerationError as e:
        msg = str(e)
        assert "cohort-wise" in msg or "generate()" in msg, "must name the safe alternative"
        assert "n_min" in msg, "must name the knob that changes coverage"
        assert "20%" in msg, f"must quantify the uncovered column; got: {msg}"

    # An explicit, informed override stays available -- refusal, not prohibition.
    try:
        g.generate_by_cell(_release_with_coverage(bands_released=1), n_rows=400,
                           allow_low_coverage=True)
    except GenerationError as e:
        assert "unsafe on this release" not in str(e), "allow_low_coverage must bypass the guard"
    except Exception:
        pass


def test_cellwise_allows_a_release_that_does_cover_the_population():
    """Releases at 91% and 98% per-column coverage were measured to be FINE -- at the higher one
    cell-wise beat cohort-wise on every aggregate metric. An earlier aggregate-coverage rule
    refused both, which would have cost real quality for no safety. The guard must not fire here.
    """
    import os
    from cortec.generate import Generator, GenerationError
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    g = Generator(_probe_schema(), backend="mock", model="claude-fable-5")
    for released, total in ((5, 5), (10, 10), (19, 20)):   # 100%, 100%, 95%
        try:
            g.generate_by_cell(_release_with_coverage(bands_released=released, n_bands=total),
                               n_rows=400)
        except GenerationError as e:
            assert "unsafe on this release" not in str(e), (
                f"guard fired at {released}/{total} per-column coverage: {e}")
        except Exception:
            pass


def test_guard_is_per_column_not_aggregate():
    """Two releases far apart in AGGREGATE cell coverage did identical damage because both
    released one band of the same column; two releases an aggregate rule refused were fine.
    The gate must therefore read per-column mass, not total cell mass.
    """
    import os
    from cortec.generate import Generator, GenerationError
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    g = Generator(_probe_schema(), backend="mock", model="claude-fable-5")

    # Same per-column coverage (1 of 10 bands) at two very different cell counts: both must refuse.
    for n_cells in (1, 1):
        with pytest.raises(GenerationError, match="unsafe on this release"):
            g.generate_by_cell(_release_with_coverage(bands_released=1, n_bands=10), n_rows=400)

    # 9 of 10 bands = 90% exactly -> at the floor, must NOT refuse.
    try:
        g.generate_by_cell(_release_with_coverage(bands_released=9, n_bands=10), n_rows=400)
    except GenerationError as e:
        assert "unsafe on this release" not in str(e), f"fired at exactly the floor: {e}"
    except Exception:
        pass


def test_both_implementations_share_one_coverage_floor():
    """The research pipeline and the shipped tool must not drift apart on the guard they enforce.

    A paper that reports a floor and a tool that applies a different one is a false claim with
    extra steps.
    """
    import re
    from pathlib import Path
    from cortec.generate import CELL_COVERAGE_FLOOR as TOOL
    # This package is distributed standalone, where the research tree is absent. The check is a
    # cross-repository consistency guard, so it SKIPS rather than fails when run outside the
    # research checkout -- otherwise every user installing the tarball sees a red suite on a
    # condition they cannot act on, which is how a suite stops being read at all.
    src_path = Path(__file__).resolve().parents[3] / "src" / "llm_generator.py"
    if not src_path.exists():
        pytest.skip("research pipeline not present; this is the standalone package")
    m = re.search(r"^CELL_COVERAGE_FLOOR\s*=\s*([0-9.]+)", src_path.read_text(), re.M)
    assert m, "research pipeline has no CELL_COVERAGE_FLOOR"
    assert float(m.group(1)) == TOOL, (
        f"pipeline floor {m.group(1)} != tool floor {TOOL}; they must be one number")


def test_coarsening_and_the_prompt_fix_are_coupled():
    """Cell keys may be coarsened ONLY while the prompt explains the groups. Never one alone.

    This started as a tripwire asserting the library does not coarsen at all, on the reasoning
    that a cell pinning one exact value has no within-group distribution to lose. That immunity
    was real but it was bought with a budgeting bug: autoconfig derived a coarsening, counted
    cells with it, and then keyed cells on the raw level -- reporting 16 cells on a 26-level
    discharge column and building 104.

    Applying the coarsening fixes the count and re-opens research defect 15, where a cell keyed on
    an opaque `g0` told the generator which values were legal and not how often each occurs. So the
    two changes are coupled, and this test pins the coupling in BOTH directions: whichever a future
    edit removes, this fails.
    """
    import inspect
    from cortec import release as R
    from cortec import generate as G

    keys_coarsen = "coarsen" in inspect.getsource(R._cell_keys)
    prompt_explains = "coarsen" in inspect.getsource(G.cell_description)
    assert keys_coarsen == prompt_explains, (
        f"cell keys coarsen={keys_coarsen} but the cell prompt explains groups={prompt_explains}. "
        f"Coarsened keys without within-group shares in the prompt IS research defect 15; "
        f"uncoarsened keys with a coarsening-aware budget is the cell-count mismatch.")
    if keys_coarsen:
        # and the expansion must actually carry frequencies, not just the member list
        out = G.cell_description("c=g0", ["c"],
                                 {"c": {"a": "g0", "b": "g0"}}, {"c": {"a": 0.9, "b": 0.1}})
        assert "g0" not in out and "%" in out, f"groups expanded without shares: {out!r}"


def test_seeding_a_release_is_loudly_flagged_as_voiding_the_guarantee():
    """A seeded release is byte-reproducible, so its epsilon is not a guarantee.

    The default (seed=None) draws from OS entropy and is correct. Seeding exists for tests. It used
    to pass silently -- and the noise helper's own docstring invited callers to "seed it for
    reproducibility" -- so the single path that voids the guarantee was also the most inviting one.
    Anyone holding a seeded release and its seed can subtract the noise and recover the private
    counts exactly.
    """
    import io, contextlib
    import numpy as np, pandas as pd
    from cortec import Schema, release_statistics

    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame({"age": rng.integers(18, 80, n),
                       "bmi": rng.normal(27, 5, n).round(1),
                       "sex": rng.choice(["M", "F"], n),
                       "y": rng.choice(["YES", "NO"], n, p=[.3, .7])})
    sc = Schema(name="d", numerical={"age": (18, 80), "bmi": (10, 60)},
                categorical={"sex": ["M", "F"]}, target="y", positive="YES", negative="NO",
                bins={"age": [18, 40, 60, 80], "bmi": [10, 25, 30, 60]},
                stratify=[], conditional=[()])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        seeded = release_statistics(sc, df, epsilon_total=2.0, n_min=150, seed=42)
    assert "SEEDED" in buf.getvalue().upper(), "a seeded release must warn on stdout"
    assert seeded.audit.get("guarantee_void") is True, (
        "the voided guarantee must be recorded in the AUDIT, not only printed — the release is "
        "read later from JSON by someone who never saw the console")
    assert any("epsilon" in w.lower() for w in seeded.audit.get("warnings", []))

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        clean = release_statistics(sc, df, epsilon_total=2.0, n_min=150)
    assert "guarantee_void" not in clean.audit, "an unseeded release must carry no such flag"
    assert "SEEDED" not in buf2.getvalue().upper(), "an unseeded release must not warn"


def test_unseeded_release_is_the_default_and_is_not_reproducible():
    import numpy as np, pandas as pd
    from cortec import Schema, release_statistics
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame({"age": rng.integers(18, 80, n), "bmi": rng.normal(27, 5, n).round(1),
                       "sex": rng.choice(["M", "F"], n),
                       "y": rng.choice(["YES", "NO"], n, p=[.3, .7])})
    sc = Schema(name="d", numerical={"age": (18, 80), "bmi": (10, 60)},
                categorical={"sex": ["M", "F"]}, target="y", positive="YES", negative="NO",
                bins={"age": [18, 40, 60, 80], "bmi": [10, 25, 30, 60]},
                stratify=[], conditional=[()])
    a = release_statistics(sc, df, epsilon_total=2.0, n_min=150)
    b = release_statistics(sc, df, epsilon_total=2.0, n_min=150)
    assert (a.cohorts[0]["numerical"]["age"]["proportions"]
            != b.cohorts[0]["numerical"]["age"]["proportions"]), (
        "two default releases of the same data are identical — the noise source is seeded "
        "somewhere and the guarantee is void")


# ─────────────────────────────────────────────────────────────────────────────────────
# Defect: autoconfig derived a coarsening, COUNTED cells with it, and never APPLIED it.
# On a 26-level discharge column it reported 16 cells at the finest level and built 104 --
# 3.2 records per cell against its own MIN_RECORDS_PER_CELL of 8, so the depth chooser
# selected a depth it would have refused had it counted what it actually builds.
#
# Applying the coarsening fixes the count and immediately creates the SECOND defect
# (research defect 15): the cell key becomes an opaque `disposition=g0`. Both are pinned
# here, because fixing only the first is worse than fixing neither.
# ─────────────────────────────────────────────────────────────────────────────────────
class TestCoarseningIsAppliedAndItsGroupsAreExplained:

    def _hospital(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        codes = [str(i) for i in range(1, 27)]
        p = np.array([0.59, 0.14, 0.13, 0.04] + [0.10 / 22] * 22)
        p = p / p.sum()
        disp = rng.choice(codes, size=n, p=p)
        age = rng.integers(18, 90, n)
        # Real conditional structure, or the release correctly refuses a noise-dominated table
        # and the fixture tests nothing. Rate varies strongly by disposition and by age.
        rate = np.where(np.isin(disp, ["1", "5", "9", "13"]), 0.08, 0.45) + (age > 60) * 0.20
        df = pd.DataFrame({
            "age": age, "los": rng.integers(1, 14, n), "disposition": disp,
            "y": np.where(rng.random(n) < rate, "YES", "NO")})
        sc = Schema(name="hosp", numerical={"age": (18, 90), "los": (1, 14)},
                    categorical={"disposition": codes},
                    target="y", positive="YES", negative="NO",
                    bins={"age": [18, 35, 50, 65, 90], "los": [1, 4, 7, 14]})
        return sc, df

    def test_cells_counted_equals_cells_built(self):
        from dataclasses import replace
        from cortec.autoconfig import derive
        from cortec.release import _cell_keys

        sc, df = self._hospital()
        d = derive(sc, df, epsilon_total=2.0, n_records=1000, n_min=150, seed=None)
        assert d.coarsen, "this fixture exists to exercise coarsening; none was derived"
        sc2 = replace(sc, conditional=d.levels, coarsen=d.coarsen)
        built = _cell_keys(df, tuple(d.levels[-1]), sc2).nunique()
        assert built == d.cells_at_finest, (
            f"autoconfig budgeted for {d.cells_at_finest} cells and built {built}; the depth "
            f"chooser is selecting against a cell count it does not produce")

    def test_release_carries_the_coarsening_onto_the_schema(self):
        sc, df = self._hospital(n=20000)
        rel = release_statistics(sc, df, epsilon_total=2.0, n_min=150, n_records=1000)
        keys = [k for lv in rel.conditional_levels for k in (lv.get("cells") or {})]
        assert keys, "no conditional table released"
        # if coarsening reached the release, some key names a group label, not a raw code
        assert any("=g" in k for k in keys), \
            f"coarsening was derived but not applied to the released cells: {keys[:4]}"
        # and no released cell may still pin a raw 26-level code
        raw = [k for k in keys if any(f"disposition={c}" in k for c in map(str, range(1, 27)))]
        assert not raw, f"cells still keyed on raw levels despite a coarsening: {raw[:3]}"

    def test_a_coarsened_group_is_never_shown_as_an_opaque_label(self):
        """Research defect 15: `disposition = g0` tells the generator nothing."""
        from cortec.generate import cell_description

        coarsen = {"disposition": {"1": "g0", "5": "g0", "13": "g0", "3": "g1", "6": "g1"}}
        shares = {"disposition": {"1": 0.592, "5": 0.011, "13": 0.004, "3": 0.136, "6": 0.127}}
        out = cell_description("disposition=g0 & age[18,35)", ["disposition", "age"],
                               coarsen, shares)
        assert "g0" not in out, f"opaque group label reached the prompt: {out!r}"
        assert "is one of" in out and "%" in out, \
            f"group expanded without within-group shares, which is defect 15: {out!r}"
        # the dominant member must dominate, not be spread uniformly across the group
        assert "1 97%" in out or "1 98%" in out, \
            f"within-group shares are not renormalised correctly: {out!r}"

    def test_a_single_member_group_collapses_to_the_value(self):
        from cortec.generate import cell_description
        out = cell_description("disposition=g2", ["disposition"], {"disposition": {"7": "g2"}}, {})
        assert out.strip() == "disposition = 7", out


# ─────────────────────────────────────────────────────────────────────────────────────
# Epsilon protects one ROW. Under group privacy a person contributing k rows receives
# k*epsilon, so on encounter-level data the per-person guarantee is k times weaker than
# the audited number. The tool used to report a clean epsilon with no indication of this:
# a hospital releasing encounter data got "epsilon 2.0" and no hint that its heaviest
# patient (40 encounters in the real Diabetes 130 dataset) was at 80.
# ─────────────────────────────────────────────────────────────────────────────────────
class TestThePrivacyUnitIsStatedNotAssumed:

    def test_undeclared_privacy_unit_is_recorded_as_undeclared_not_as_one(self):
        from cortec.accounting import PrivacyLedger
        led = PrivacyLedger(2.0)
        led.spend("q", kind="count", epsilon=2.0, sensitivity=1.0,
                  composition="sequential", partition="a", group="g")
        rep = led.audit_report()
        assert rep["max_rows_per_person"] == "UNDECLARED", \
            "an undeclared privacy unit must not be silently assumed to be 1 row per person"
        assert "UNDECLARED" in str(rep["epsilon_per_person"])
        assert "PRIVACY UNIT UNDECLARED" in led.summary()

    def test_group_privacy_is_computed_and_flagged_when_vacuous(self):
        from cortec.accounting import PrivacyLedger
        led = PrivacyLedger(2.0, max_rows_per_person=40)
        led.spend("q", kind="count", epsilon=2.0, sensitivity=1.0,
                  composition="sequential", partition="a", group="g")
        rep = led.audit_report()
        # the real Diabetes 130 figure: 40 encounters x eps_row 2.0
        assert rep["epsilon_per_person"] == 80.0, rep["epsilon_per_person"]
        s = led.summary()
        assert "EPSILON PER PERSON" in s
        assert "outside any range normally considered meaningful" in s, \
            "an eps_person of 80 must be flagged as vacuous, not merely printed"

    def test_one_row_per_person_is_stated_explicitly(self):
        from cortec.accounting import PrivacyLedger
        led = PrivacyLedger(2.0, max_rows_per_person=1)
        led.spend("q", kind="count", epsilon=2.0, sensitivity=1.0,
                  composition="sequential", partition="a", group="g")
        assert led.audit_report()["epsilon_per_person"] == 2.0
        assert "outside any range" not in led.summary()

    def test_release_threads_the_declaration_into_the_audit(self):
        sc, df = demo_schema(), demo_frame()
        rel = release_statistics(sc, df, epsilon_total=2.0, n_min=150, max_rows_per_person=3)
        assert rel.audit["max_rows_per_person"] == 3
        assert rel.audit["epsilon_per_person"] == pytest.approx(3 * rel.audit["epsilon_accounted"])
        assert rel.audit["privacy_unit"] == "one dataset row"

    def test_a_nonsense_declaration_is_refused_before_budget_is_spent(self):
        from cortec.accounting import PrivacyLedger, PrivacyAccountingError  # noqa: F401
        for bad in (0, -1, 1.5, "40"):
            with pytest.raises(ValueError):
                PrivacyLedger(2.0, max_rows_per_person=bad)


# ─────────────────────────────────────────────────────────────────────────────────────
# The capability table used to assign tiers from the TRANSMISSION SWEEP alone, and two
# self-hosted 70B models scored inside the frontier band on it. Measured on full-dataset
# generation the same models close 12% of the conditional-error gap to an UNCONDITIONED
# prompt and nothing on 2-way TV. The paper withdrew the "70B matches a frontier model"
# conclusion; the tool kept recommending those models by default. These pin the alignment.
# ─────────────────────────────────────────────────────────────────────────────────────
class TestSweepOnlyModelsAreNotRecommended:

    def test_seventy_b_models_are_not_listed_as_validated(self):
        from cortec.models import validated_models
        listed = " ".join(validated_models()).lower()
        for m in ("llama", "qwen"):
            assert m not in listed, (
                f"{m} is recommended as validated, but it passes only the transmission sweep; "
                f"the paper withdrew the claim that 70B-class models match a frontier model")

    def test_a_sweep_only_model_is_refused_by_default(self):
        from cortec.models import check_model, ModelCapabilityError, Tier, profile_for
        assert profile_for("llama3.3:70b-instruct").tier is Tier.SWEEP_ONLY
        with pytest.raises(ModelCapabilityError, match="FAILS full-dataset generation"):
            check_model("llama3.3:70b-instruct")

    def test_the_refusal_explains_why_the_sweep_number_is_not_enough(self):
        from cortec.models import check_model, ModelCapabilityError
        try:
            check_model("qwen2.5:72b-instruct")
        except ModelCapabilityError as e:
            msg = str(e)
        else:
            pytest.fail("sweep-only model was not refused")
        # it must say what the sweep measured AND what full generation measured, or the operator
        # cannot tell why a model with a good-looking number is being refused
        assert "transmission sweep" in msg and "full-dataset generation" in msg
        assert "UNCONDITIONED" in msg
        assert "acknowledge_insufficient=True" in msg

    def test_it_can_still_be_run_deliberately(self):
        from cortec.models import check_model, Tier
        p = check_model("llama3.3:70b-instruct", acknowledge_insufficient=True)
        assert p.tier is Tier.SWEEP_ONLY



# ─────────────────────────────────────────────────────────────────────────────────────
# The recommendation is a LIST OF MODELS, not a list of properties. We could not derive
# properties that separate a working generator from a failing one: four passing models
# across three vendors leaves everything confounded, "capability tier" is a vendor label,
# and the only numeric thresholds available come from the transmission sweep, which
# mispredicts generation. So any other model is admitted on a MEASUREMENT, never on a
# description of itself.
def test_a_refusal_is_not_reported_as_an_exhausted_output_budget():
    """Found by running the tool end to end: a live refusal was misdiagnosed.

    A refusal and an exhausted output budget both arrive as "no text content", and the remedies are
    opposite -- one needs a different prompt, cell size or model, the other needs a larger budget.
    The old message told the operator to raise max_output_tokens after a refusal, which sends them
    to re-run the identical request at greater cost. Same shape as the yield-guard defect: a
    confident diagnosis pointing at the wrong subsystem.
    """
    import inspect
    from cortec import generate as G
    src = inspect.getsource(G.Generator._call_once)
    assert 'stop == "refusal"' in src, "refusal is no longer distinguished from an empty budget"
    assert "ModelRefusalError" in src
    # and the two messages must give OPPOSITE advice
    i = src.index('stop == "refusal"')
    refusal_msg = src[i:i + 1400]
    assert "will not fix it" in refusal_msg and "coarser conditional level" in refusal_msg
    assert G.ModelRefusalError is not G.EmptyContentError
    assert issubclass(G.ModelRefusalError, G.GenerationError)


class TestTheAuditIsInternallyRecomputable:
    """Every query's recorded noise scale must equal sensitivity/epsilon.

    The audit trail exists so a privacy engineer can recompute the release by hand from the dict
    alone. That only works if the recorded sensitivity and epsilon are the ones that produced the
    recorded scale. The autoconfig selection query used to declare sensitivity 1.0 while drawing
    noise at scale m/eps -- so the audit implied eps = 1/m x what was actually charged, and anyone
    checking the arithmetic would have reached a different number than the mechanism used. The
    cause was that derive() computed its own scale and the caller charged the ledger separately;
    they agreed only by convention.
    """

    def _autoconfigured(self):
        rng = np.random.default_rng(0)
        n = 20000
        codes = [str(i) for i in range(1, 27)]
        p = np.array([0.59, 0.14, 0.13, 0.04] + [0.10 / 22] * 22)
        p = p / p.sum()
        disp = rng.choice(codes, size=n, p=p)
        age = rng.integers(18, 90, n)
        rate = np.where(np.isin(disp, ["1", "5", "9", "13"]), 0.08, 0.45) + (age > 60) * 0.20
        df = pd.DataFrame({"age": age, "los": rng.integers(1, 14, n), "disposition": disp,
                           "y": np.where(rng.random(n) < rate, "YES", "NO")})
        sc = Schema(name="hosp", numerical={"age": (18, 90), "los": (1, 14)},
                    categorical={"disposition": codes},
                    target="y", positive="YES", negative="NO",
                    bins={"age": [18, 35, 50, 65, 90], "los": [1, 4, 7, 14]})
        return release_statistics(sc, df, epsilon_total=2.0, n_min=150, n_records=1000)

    def test_the_scale_actually_drawn_is_the_scale_the_ledger_charged(self):
        """Instrument the DRAW, not the ledger's own arithmetic.

        A first version of this test compared each query's recorded noise_scale against its
        recorded sensitivity/epsilon -- which `spend()` computes itself, so it agreed by
        construction and passed even with the defect deliberately reinstated. A test that cannot
        fail is worse than no test. This one records the scales actually passed to the RNG and
        requires each to be one the ledger handed out.
        """
        import numpy as _np
        from cortec import autoconfig as _ac, release as _rel
        drawn: list[float] = []

        class _RecordingRNG:
            """numpy Generators are immutable, so wrap rather than patch the method."""
            def __init__(self, inner):
                self._inner = inner

            def laplace(self, loc=0.0, scale=1.0, size=None):
                drawn.append(float(scale))
                return self._inner.laplace(loc, scale, size)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        real_default_rng = _np.random.default_rng
        spy = lambda *a, **k: _RecordingRNG(real_default_rng(*a, **k))   # noqa: E731
        _ac.np.random.default_rng = spy
        _rel.np.random.default_rng = spy
        try:
            rel = self._autoconfigured()
        finally:
            _ac.np.random.default_rng = real_default_rng
            _rel.np.random.default_rng = real_default_rng

        charged = {round(q["noise_scale"], 9) for q in rel.audit["queries"]}
        assert drawn, "no noise was drawn"
        unaccounted = sorted({round(s, 9) for s in drawn} - charged)
        assert not unaccounted, (
            f"noise drawn at scale(s) {unaccounted} that the ledger never returned; "
            f"ledger issued {sorted(charged)}. A statistic was noised outside the audit trail.")

    def test_the_selection_query_is_charged_exactly_once(self):
        rel = self._autoconfigured()
        sel = [q for q in rel.audit["queries"] if q["group"] == "autoconfig"]
        assert len(sel) == 1, f"selection charged {len(sel)} times"

    def test_autoconfigured_release_still_closes_the_budget_exactly(self):
        rel = self._autoconfigured()
        assert rel.audit["within_budget"]
        assert abs(rel.audit["epsilon_accounted"] - 2.0) < 1e-9, rel.audit["epsilon_accounted"]


def test_the_coverage_guard_understands_coarsened_cell_keys():
    """Found by reading the coverage guard line by line against the coarsening fix.

    Cell keys carry the COARSENED label (`disposition=g0`) while a cohort's categorical block is
    keyed on raw levels (`disposition=1`). The guard compared the two directly, matched nothing on
    every coarsened column, computed 0% coverage and refused a release whose groups were entirely
    covered — making cell-wise generation impossible on any auto-configured release with a
    high-cardinality categorical.

    Two things had to change: the guard aggregates the released proportions by group, and the
    coarsening travels on the Release (release_statistics applies it to a LOCAL copy of the schema,
    so the caller's Schema object never learns about it).
    """
    from cortec.generate import Generator, CELL_COVERAGE_FLOOR

    rng = np.random.default_rng(0)
    n = 20000
    codes = [str(i) for i in range(1, 27)]
    p = np.array([0.59, 0.14, 0.13, 0.04] + [0.10 / 22] * 22)
    p = p / p.sum()
    disp = rng.choice(codes, size=n, p=p)
    age = rng.integers(18, 90, n)
    rate = np.where(np.isin(disp, ["1", "5", "9", "13"]), 0.08, 0.45) + (age > 60) * 0.20
    df = pd.DataFrame({"age": age, "los": rng.integers(1, 14, n), "disposition": disp,
                       "y": np.where(rng.random(n) < rate, "YES", "NO")})
    sc = Schema(name="hosp", numerical={"age": (18, 90), "los": (1, 14)},
                categorical={"disposition": codes},
                target="y", positive="YES", negative="NO",
                bins={"age": [18, 35, 50, 65, 90], "los": [1, 4, 7, 14]})
    rel = release_statistics(sc, df, epsilon_total=2.0, n_min=150, n_records=1000)

    assert rel.coarsen, "the release must carry the coarsening it applied to its cell keys"
    assert "disposition" in rel.coarsen

    gen = Generator(sc, backend="anthropic", model="claude-fable-5", budget_usd=0.0)
    # Every group of the coarsened column is released, so coverage is 1.0 and the guard must pass.
    # It cannot reach the network here; anything other than a coverage refusal means it passed.
    try:
        gen.generate_by_cell(rel, n_rows=200, level=len(rel.conditional_levels) - 1, verbose=False)
    except Exception as e:
        assert "unsafe on this release" not in str(e), (
            f"the coverage guard refused a fully-covered coarsened column: {str(e)[:200]}")


def test_release_round_trips_its_coarsening_through_json():
    """The coarsening must survive to_json/from_json or a reloaded release fails the guard."""
    import tempfile, os
    from cortec.release import Release
    rel = Release(schema_name="x", epsilon_total=2.0, n_min=150,
                  coarsen={"c": {"a": "g0", "b": "g0"}})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.json")
        rel.to_json(path)
        back = Release.from_json(path)
    assert back.coarsen == {"c": {"a": "g0", "b": "g0"}}


def test_a_coarsening_is_validated_like_every_other_schema_field():
    """`coarsen` was added after the rest and initially skipped validation.

    A map naming an undeclared column, or a level outside the declared value set, would not raise:
    it would simply never match, and the column would be keyed on raw values while the coverage
    guard and the cell prompt both expected groups.
    """
    base = dict(name="x", numerical={"a": (0, 1)}, categorical={"c": ["p", "q"]},
                target="y", positive="Y", negative="N")
    Schema(**base, coarsen={"c": {"p": "g0", "q": "g0"}})        # valid
    for bad, why in [({"nosuch": {"p": "g0"}}, "undeclared column"),
                     ({"c": {}}, "empty map"),
                     ({"c": {"p": "g0", "ZZZ": "g1"}}, "undeclared level")]:
        with pytest.raises(SchemaError):
            Schema(**base, coarsen=bad)


def test_generation_evidence_notes_agree_with_the_profile_table():
    """Two model tables describe the same models, and they drifted apart.

    `MEASURED_ON_GENERATION_ONLY` was written when GPT-5 and Gemini 3.1 Pro had no profile, and its
    notes said so in prose ("not in PROFILES").  Both were later promoted to VALIDATED on
    full-dataset generation and the notes were not touched, so the tool shipped a capability table
    contradicted by the capability table beside it -- and §11.2 of the paper repeated the stale
    version.  Neither table is derivable from the other, so assert they agree.
    """
    from cortec.models import MEASURED_ON_GENERATION_ONLY, Tier, profile_for

    for stem, note in MEASURED_ON_GENERATION_ONLY.items():
        p = profile_for(stem)
        if "not in PROFILES" in note:
            assert p.tier is Tier.UNKNOWN, (
                f"{stem}: its note says it is not in PROFILES, but PROFILES tiers it {p.tier.value}")
        if "Not scored on the transmission sweep" in note:
            # the sweep is where magnitude_error comes from; borrowing a sibling's is the error
            # the profile table exists to prevent, so an unscored model must carry None.
            assert p.magnitude_error is None or p.tier is Tier.UNKNOWN, (
                f"{stem}: unscored on the sweep, yet carries magnitude_error={p.magnitude_error}")


class TestVacuousPerPersonEpsilonIsRefusedNotDocumented:
    """The privacy unit was a checklist item; on longitudinal data that is not enough.

    eps protects one ROW. A person contributing k rows gets k*eps, so Diabetes 130's heaviest
    patient (40 encounters) receives eps_person = 80 at a declared eps_row = 2.0 -- a number with no
    meaning. The tool used to compute that, print it, and proceed. Where the caller has DECLARED k
    we know the per-person guarantee before spending a unit of budget, so shipping a vacuous one
    quietly is a refusable condition, not a note.

    Undeclared is deliberately a different case: k cannot be computed without spending budget, and
    defaulting to 1 would manufacture the per-person claim this gate exists to prevent -- so it is a
    loud banner and a recorded warning that travels with the release.
    """

    def test_a_declared_vacuous_privacy_unit_is_refused(self):
        from cortec.release import release_statistics, ReleaseError
        with pytest.raises(ReleaseError) as e:
            release_statistics(demo_schema(), demo_frame(n=4000), epsilon_total=2.0,
                               n_min=150, max_rows_per_person=40)
        msg = str(e.value)
        assert "epsilon_per_person=80" in msg
        # the refusal must name the way out, or it is an obstacle rather than a guardrail
        assert "aggregate to one row per person" in msg.lower()
        assert "acknowledge_vacuous_privacy_unit" in msg

    def test_the_refusal_can_be_acknowledged_and_the_acknowledgement_is_recorded(self):
        from cortec.release import release_statistics
        rel = release_statistics(demo_schema(), demo_frame(n=4000), epsilon_total=2.0, n_min=150,
                                 max_rows_per_person=40, acknowledge_vacuous_privacy_unit=True)
        assert rel.audit["epsilon_per_person"] == 80.0
        assert rel.audit["epsilon_per_person_vacuous"] is True
        # it has to survive into the SAVED artefact: the person who reads release.json later
        # never saw the console banner
        assert any("VACUOUS" in w for w in rel.audit.get("warnings", []))

    def test_a_declared_one_row_per_person_release_is_clean(self):
        from cortec.release import release_statistics
        rel = release_statistics(demo_schema(), demo_frame(n=4000), epsilon_total=2.0,
                                 n_min=150, max_rows_per_person=1)
        assert rel.audit["epsilon_per_person"] == 2.0
        assert rel.audit["epsilon_per_person_vacuous"] is False
        assert rel.audit["privacy_unit_declared"] is True
        assert not rel.audit.get("warnings")

    def test_an_undeclared_privacy_unit_warns_in_the_saved_release(self):
        from cortec.release import release_statistics
        rel = release_statistics(demo_schema(), demo_frame(n=4000), epsilon_total=2.0, n_min=150)
        assert rel.audit["privacy_unit_declared"] is False
        # "UNKNOWN" and "not vacuous" must not be the same value to a compliance reader
        assert rel.audit["epsilon_per_person_vacuous"] == "UNKNOWN"
        assert any("UNDECLARED" in w for w in rel.audit.get("warnings", []))

    def test_the_vacuity_verdict_is_machine_readable_not_only_printed(self):
        """The human-readable warning lives in summary(); audit.json is what pipelines read."""
        from cortec.accounting import PrivacyLedger, VACUOUS_EPSILON_PER_PERSON
        for k, want in ((1, False), (5, False), (6, True), (40, True)):
            led = PrivacyLedger(2.0, max_rows_per_person=k)
            led.spend("h", kind="histogram", epsilon=2.0, sensitivity=1.0,
                      composition="parallel", partition="p", group="g")
            rep = led.audit_report()
            assert rep["epsilon_per_person_vacuous"] is want, f"k={k}"
            assert rep["vacuous_threshold"] == VACUOUS_EPSILON_PER_PERSON


class TestTheAuditArtefactCannotBeReadAsAPrivacyProof:
    """The Stage C report was renamed away from "certificate"; the release audit was not touched.

    That left the artefact a compliance team ACTUALLY files -- `release.audit`, written out by
    `PrivacyLedger.write_audit` -- asserting `"guarantee": "record-level eps-differential privacy"`
    and `"within_budget": true` with no statement anywhere of where that guarantee stops. A
    machine-readable file that claims a guarantee and states no boundary reads as a proof of privacy
    safety, which is the exact misreading the Stage C rename exists to prevent.

    NIST SP 800-226 asks for "where the guarantee does not hold" explicitly.
    """

    def _report(self, **kw):
        from cortec.accounting import PrivacyLedger
        led = PrivacyLedger(kw.pop("epsilon", 2.0), **kw)
        led.spend("h", kind="histogram", epsilon=1.0, sensitivity=1.0,
                  composition="parallel", partition="p", group="g")
        return led, led.audit_report()

    def test_the_audit_says_what_it_is_and_what_it_is_not(self):
        _, rep = self._report()
        assert next(iter(rep)) == "_what_this_is", "must be the first key a reader meets"
        is_not = rep["_what_this_is"]["is_not"].lower()
        for word in ("privacy audit", "re-identification", "certificate"):
            assert word in is_not, word

    def test_the_audit_states_where_the_guarantee_does_not_hold(self):
        _, rep = self._report()
        caveats = rep["where_this_guarantee_does_not_hold"]
        assert len(caveats) >= 4
        blob = " ".join(caveats).lower()
        # the four the paper maps to NIST SP 800-226 in §5.3
        assert "one row" in blob                      # privacy unit
        assert "mironov" in blob                      # floating-point Laplace
        assert "pretraining" in blob                  # outside the DP boundary
        assert "charge_suppression" in blob           # uncharged suppression if opted into

    def test_every_caveat_that_names_a_field_names_one_that_exists(self):
        """A caveat pointing at a field the report does not carry is worse than no caveat."""
        _, rep = self._report()
        blob = " ".join(rep["where_this_guarantee_does_not_hold"])
        for field in ("max_rows_per_person", "epsilon_per_person", "charge_suppression"):
            if field in blob:
                assert field in rep, f"caveat points at {field!r}, absent from the report"

    def test_the_printed_summary_carries_the_same_disclaimer(self):
        """A reader who sees only the console must not leave thinking this asserts safety."""
        led, _ = self._report()
        s = led.summary()
        assert "NOT A PRIVACY AUDIT" in s
        assert "WHERE THIS GUARANTEE DOES NOT HOLD" in s

    def test_no_field_anywhere_in_the_audit_is_named_for_certification(self):
        _, rep = self._report()
        names: list[str] = []

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    names.append(k)
                    walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)
        walk(rep)
        assert not [k for k in names if "certif" in k.lower()], names


# ── cell-wise: the model is ASKED for exactly k positives; a miss must be reported ─────

def test_cell_wise_mock_reproduces_the_requested_positive_counts_exactly():
    """With a backend that honours the instruction, emitted == requested on every call, and the
    output rate per cell tracks the released rate. (The earlier mock drew a random label per row,
    which made this test vacuous.)"""
    s = demo_schema()
    rel = release_statistics(s, demo_frame(n=4000), epsilon_total=2.0, n_min=150, seed=3)
    g = Generator(s, backend="mock", model="claude-fable-5", rows_per_call=10, seed=5)
    out = g.generate_by_cell(rel, 300, level=0, verbose=False)
    assert g.stats.positives_requested > 0
    assert g.stats.positives_emitted == g.stats.positives_requested
    assert g.stats.positives_off_by_more_than_one == 0
    assert not any("exactly k" in w for w in g.stats.warnings)


def test_cell_wise_reports_a_model_that_ignores_the_requested_count(monkeypatch):
    """A live Gemini run emitted 22 positives where 21 were asked in a 46-row cell. The count is a
    request, not a constraint; a miss larger than one row must be counted and warned about, never
    silently relabelled and never silently accepted."""
    s = demo_schema()
    rel = release_statistics(s, demo_frame(n=4000), epsilon_total=2.0, n_min=150, seed=3)
    g = Generator(s, backend="mock", model="claude-fable-5", rows_per_call=10, seed=5)
    real_mock = g._mock

    def all_positive(prompt):                       # a model that ignores k entirely
        text = real_mock(prompt)
        head, *rows = text.splitlines()
        fixed = [",".join(r.split(",")[:-1] + [s.positive]) for r in rows]
        return "\n".join([head] + fixed)
    monkeypatch.setattr(g, "_mock", all_positive)
    out = g.generate_by_cell(rel, 300, level=0, verbose=False)
    assert len(out) > 0
    assert g.stats.positives_emitted > g.stats.positives_requested
    assert g.stats.positives_off_by_more_than_one > 0
    assert any("exactly k" in w for w in g.stats.warnings), g.stats.warnings
    # and the output is what the model produced -- the tool did not relabel to hide the miss
    assert (out[s.target] == s.positive).all()


# ── ledger: nested partitions charge as one chain with their parent, in parallel with siblings ──

def test_nested_partitions_compose_sequentially_with_their_parent_and_in_parallel_with_siblings():
    """Class-conditional histograms: class balance on the cohort, one histogram block per class
    beneath it. A record in cohort_3/YES is touched by the balance query AND its block, so the
    charge is balance + one block -- never balance alone (bare keys called the child disjoint from
    the parent) and never balance + both blocks (one key for both classes)."""
    from cortec.accounting import PrivacyLedger
    led = PrivacyLedger(10.0)
    led.spend("balance", kind="histogram", epsilon=0.1, sensitivity=1.0, composition="parallel",
              partition="cohort_3", group="marginals")
    for cls in ("YES", "NO"):
        for col in ("age", "bmi", "grp"):
            led.spend(f"hist[{cls}][{col}]", kind="histogram", epsilon=0.1, sensitivity=1.0,
                      composition="parallel", partition=f"cohort_3/{cls}", group="marginals")
    # a second cohort with a pooled block only, for the cross-cohort max
    led.spend("balance", kind="histogram", epsilon=0.1, sensitivity=1.0, composition="parallel",
              partition="cohort_4", group="marginals")
    for col in ("age", "bmi", "grp"):
        led.spend(f"hist[{col}]", kind="histogram", epsilon=0.1, sensitivity=1.0,
                  composition="parallel", partition="cohort_4", group="marginals")
    assert abs(led.total_epsilon - 0.4) < 1e-9, led.total_epsilon   # balance + ONE block of 3
    rep = led.audit_report()["by_group"]["marginals"] if "by_group" in led.audit_report() else None
    if rep is not None:
        assert abs(rep["parallel_cost"] - 0.4) < 1e-9
        assert abs(rep["parallel_by_lineage"]["cohort_3/YES"] - 0.4) < 1e-9
        assert abs(rep["parallel_by_lineage"]["cohort_3"] - 0.1) < 1e-9


# ── class-conditional histograms: released per (cohort, class), charged as one chain ──────

def test_class_conditional_blocks_are_released_at_unchanged_epsilon():
    """One histogram block per (cohort, class) where both classes clear n_min; the ledger charges
    balance + ONE block per record through nested partitions, so epsilon_accounted is exactly the
    declared total -- the same as the pooled release."""
    s = demo_schema()
    frame = demo_frame(n=6000)
    rel = release_statistics(s, frame, epsilon_total=2.0, n_min=150, seed=3)
    split = [c for c in rel.cohorts if c.get("by_class")]
    assert split, "no cohort released class blocks"
    assert rel.audit["epsilon_accounted"] == pytest.approx(2.0, abs=1e-9)
    marg = rel.audit["groups"]["marginals"]
    assert any("/" in k for k in marg["parallel_by_partition"]), "class blocks must be nested partitions"
    for c in split:
        for cls in (s.positive, s.negative):
            blk = c["by_class"][cls]
            assert set(blk["numerical"]) == set(s.numerical)
            assert set(blk["categorical"]) == set(s.categorical)
    pooled = release_statistics(s, frame, epsilon_total=2.0, n_min=150, seed=3, class_conditional=False)
    assert not any(c.get("by_class") for c in pooled.cohorts)
    assert pooled.audit["epsilon_accounted"] == pytest.approx(rel.audit["epsilon_accounted"], abs=1e-9)


def test_pooled_histogram_is_the_mixture_of_the_class_blocks():
    """Post-processing: the pooled block must be derivable from the class blocks and the class
    balance, or a third histogram query was spent without being charged."""
    s = demo_schema()
    rel = release_statistics(s, demo_frame(n=6000), epsilon_total=2.0, n_min=150, seed=5)
    for c in (x for x in rel.cohorts if x.get("by_class")):
        w = c["class_balance"][s.positive]
        for col in s.numerical:
            hp = np.array(c["by_class"][s.positive]["numerical"][col]["proportions"])
            hq = np.array(c["by_class"][s.negative]["numerical"][col]["proportions"])
            mix = w * hp + (1 - w) * hq
            assert np.allclose(mix / mix.sum(), c["numerical"][col]["proportions"], atol=2e-4), col


def test_prompts_carry_the_per_outcome_distributions():
    from cortec.generate import build_prompt
    s = demo_schema()
    rel = release_statistics(s, demo_frame(n=6000), epsilon_total=2.0, n_min=150, seed=7)
    c = next(x for x in rel.cohorts if x.get("by_class"))
    prompt, _ = build_prompt(s, rel, c, 25)
    assert "BY OUTCOME" in prompt
    assert f"rows with {s.target} = {s.positive}" in prompt and f"rows with {s.target} = {s.negative}" in prompt
    g = Generator(s, backend="mock", model="claude-fable-5", rows_per_call=10, seed=1)
    w = np.array([x["cohort_size"] for x in rel.cohorts], float); w = w / w.sum()
    if all(x.get("by_class") for x in rel.cohorts):
        num_block, cat_block = g._mixture_blocks(rel, w)
        assert "BY OUTCOME" in num_block and f"records with {s.target} = {s.positive}" in num_block
    out = g.generate_by_cell(rel, 200, level=0, verbose=False)
    assert len(out) > 0


def test_autoconfig_does_not_coarsen_a_numeric_column_with_many_bins():
    """release_statistics(autoconfig=True) crashed with `coarsen column 'age' is not a declared
    categorical column` whenever a numeric column had more public bins than MAX_LEVELS_PER_COLUMN:
    derive() built a coarsening map for it and Schema rejected the map. Numeric columns are banded
    by their public edges and must never be coarsened; the derived release must still carry the
    class-conditional blocks the default promises."""
    import numpy as np
    import pandas as pd
    from cortec import Schema, release_statistics
    rng = np.random.default_rng(0)
    n = 6000
    age = rng.integers(17, 91, n)
    hours = rng.integers(1, 100, n)
    sex = rng.choice(["Male", "Female"], n)
    y = np.where(rng.random(n) < 0.15 + 0.5 * (age > 45) * (hours > 40), "YES", "NO")
    df = pd.DataFrame({"age": age, "hours": hours, "sex": sex, "target": y})
    schema = Schema(name="numeric-bins", numerical={"age": (17, 91), "hours": (1, 100)},
                    categorical={"sex": ["Female", "Male"]}, target="target", positive="YES", negative="NO",
                    bins={"age": [17, 25, 35, 45, 55, 65, 91], "hours": [1, 20, 35, 40, 41, 50, 100]})
    assert len(schema.bins["age"]) - 1 > 4  # more bins than the coarsening limit
    rel = release_statistics(schema, df, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             n_records=1000, seed=1)
    assert rel.cohorts, "an auto-configured release must have cohorts"
    assert any(c.get("by_class") for c in rel.cohorts), \
        "the auto-configured release must carry class-conditional blocks"
    assert not any(k in schema.numerical for k in (rel.coarsen or {})), "no numeric column may be coarsened"


def test_a_record_at_the_domain_maximum_belongs_to_the_top_band_not_to_an_oob_cohort():
    """Autoconfig bands are half-open [lo, hi) with the last hi at the declared maximum, so on Adult
    the 460 records at education_num = 16 formed an 'oob' cohort. That cohort was released like any
    other, no generated row could ever map to it, and select_to_release refused every pool."""
    import pandas as pd
    from cortec.schema import Schema, Band
    from cortec.release import _cohort_keys
    schema = Schema(name="t", numerical={"x": (1.0, 16.0)}, categorical={"g": ["a", "b"]},
                    target="y", positive="1", negative="0", bins={"x": [1, 5, 12, 16]},
                    stratify=[("x", [Band("lo", 1.0, 5.0), Band("mid", 5.0, 12.0), Band("hi", 12.0, 16.0)])])
    keys = _cohort_keys(schema, pd.DataFrame({"x": [1, 4.9, 5, 15.9, 16], "g": ["a"] * 5, "y": ["0"] * 5}))
    assert list(keys) == ["lo", "lo", "mid", "hi", "hi"]
    with pytest.raises(SchemaError, match="tile its declared domain"):
        Schema(name="t", numerical={"x": (1.0, 16.0)}, categorical={"g": ["a"]}, target="y", positive="1",
               negative="0", bins={"x": [1, 5, 12, 16]},
               stratify=[("x", [Band("lo", 1.0, 5.0), Band("hi", 12.0, 16.0)])])


def test_an_integer_coded_category_written_with_a_decimal_point_is_normalised_not_kept_as_a_new_category(monkeypatch):
    """One batch wrote SEX as "2.0"; pandas typed the column float, concatenation re-typed every
    row of the pool, and the selection step and the evaluator then saw a category ("2.0") the data
    does not have: conditional error 0.22 against 0.01 from a pool whose generation was fine."""
    from cortec.schema import Schema
    schema = Schema(name="t", numerical={"x": (0.0, 10.0)}, categorical={"sex": ["1", "2"]},
                    target="y", positive="1", negative="0", bins={"x": [0, 5, 10]})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    g = Generator(schema, backend="anthropic", model="claude-fable-5")
    df = g.parse("x,sex,y\n3,2.0,1\n4,1.0,0\n5,3,1\n6,,0\n")
    assert list(df["sex"]) == ["2", "1"]
    assert df["sex"].dtype == object and g.stats.rows_undeclared_category == 2


def test_a_transient_rate_limit_is_waited_out_not_treated_as_fatal_or_as_a_failed_call(monkeypatch):
    """`resource exhausted` sat in the fatal markers, so one transient Vertex 429 ("Resource
    exhausted. Please try again later.") stopped a whole run; in the research harness two in a row
    aborted a draw under the yield rule. The vendor's request to wait is now waited out."""
    from cortec.generate import _classify_api_error, RateLimitedError, FatalAPIError
    from cortec.schema import Schema
    e = _classify_api_error("Gemini", RuntimeError("ClientError: 429 RESOURCE_EXHAUSTED. Resource exhausted. Please try again later."))
    assert isinstance(e, RateLimitedError)
    assert isinstance(_classify_api_error("Gemini", RuntimeError("429: exceeded its monthly spending cap; see billing")), FatalAPIError)
    schema = Schema(name="t", numerical={"x": (0.0, 10.0)}, categorical={"g": ["a"]}, target="y",
                    positive="1", negative="0", bins={"x": [0, 5, 10]})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    g = Generator(schema, backend="anthropic", model="claude-fable-5")
    waits = []; monkeypatch.setattr("cortec.generate.time.sleep", lambda s: waits.append(s))
    n = {"c": 0}
    def once(prompt):
        n["c"] += 1
        if n["c"] <= 2:
            raise RateLimitedError("Gemini: 429")
        return "x,g,y\n1,a,1\n"
    monkeypatch.setattr(g, "_call_once", once)
    assert g._call("p").startswith("x,g,y") and waits == [15.0, 30.0] and g.stats.rate_limit_waits == 2


def test_a_published_cohort_size_never_falls_below_n_min(monkeypatch):
    """At epsilon = 0.3 on NHANES a 469-record cohort's noised size clipped to 0; it was allocated
    1 of 600 rows and the youngest age band vanished from the output. A cohort is released only when
    it holds n_min records, so the published size is clamped there (post-processing)."""
    import sys
    import cortec.release as RL
    from cortec.schema import Schema
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.integers(0, 10, n), "g": rng.choice(["a", "b"], n), "y": rng.choice(["0", "1"], n)})
    schema = Schema(name="t", numerical={"x": (0.0, 10.0)}, categorical={"g": ["a", "b"]}, target="y",
                    positive="1", negative="0", bins={"x": [0, 5, 10]}, conditional=[("g",)])
    real = RL.laplace_noise
    src_lines = open(RL.__file__).read().splitlines()

    def fake(rng_, scale, size=None):
        # drive only the cohort-size draw to a huge negative value; every other query keeps real noise
        caller = sys._getframe(1)
        if size is None and caller.f_code.co_filename == RL.__file__ and "_cs = int(max(n_min" in src_lines[caller.f_lineno - 1]:
            return -1e6
        return real(rng_, scale, size)
    monkeypatch.setattr(RL, "laplace_noise", fake)
    rel = RL.release_statistics(schema, df, epsilon_total=0.3, n_min=150, max_rows_per_person=1, autoconfig=False, seed=1)
    assert rel.cohorts and all(c["cohort_size"] >= 150 for c in rel.cohorts)
