"""
Regression tests for cortec-hybrid, named after the failures they prevent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cortec.schema import Schema, DataValidationError
from cortec_hybrid.core import (ConditionalTable, DomainTooLargeError, check_feasible,
                                  correct, estimate_domain_size, estimate_gain,
                                  release_conditional_table, relabel)


def schema(bins_per_num=4) -> Schema:
    return Schema(
        name="lite-demo",
        numerical={"age": (18, 90)},
        categorical={"edu": ["HS", "College", "Grad"], "region": ["N", "S"]},
        target="y", positive="YES", negative="NO",
        bins={"age": list(np.linspace(18, 90, bins_per_num + 1))},
        conditional=[("edu",)],
    )


def frame(n=4000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edu = rng.choice(["HS", "College", "Grad"], n, p=[0.5, 0.3, 0.2])
    # a deliberately COUNTERINTUITIVE relationship: more education -> lower positive rate
    p = np.where(edu == "Grad", 0.10, np.where(edu == "College", 0.35, 0.70))
    return pd.DataFrame({"age": rng.integers(18, 90, n), "edu": edu,
                         "region": rng.choice(["N", "S"], n),
                         "y": np.where(rng.random(n) < p, "YES", "NO")})


def marginal_synth_output(n=1500, seed=1) -> pd.DataFrame:
    """Stand-in for a marginal synthesiser: right marginals, target unrelated to features —
    which is the specific weakness this tool exists to correct."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.integers(18, 90, n),
                         "edu": rng.choice(["HS", "College", "Grad"], n, p=[0.5, 0.3, 0.2]),
                         "region": rng.choice(["N", "S"], n),
                         "y": rng.choice(["YES", "NO"], n, p=[0.5, 0.5])})


# ── feasibility, before hours are burned ────────────────────────────────────────────

def test_a_schema_outside_the_measured_envelope_is_refused_up_front():
    """AIM failed to converge in 3+ hours on real 15- and 19-column schemas, across attempts.
    A user must learn that in the first second, not the third hour."""
    big = Schema(name="big",
                 numerical={f"n{i}": (0, 100) for i in range(8)},
                 categorical={f"c{i}": [f"v{j}" for j in range(8)] for i in range(6)},
                 target="y", positive="YES", negative="NO",
                 bins={f"n{i}": list(range(0, 101, 10)) for i in range(8)})
    with pytest.raises(DomainTooLargeError, match="outside the region where AIM converged"):
        check_feasible(big, n_rows=24_000)


def test_the_gate_does_not_use_domain_size_because_our_own_ablation_refuted_it():
    """Domain size does not predict AIM convergence, and gating on it refuses the wrong schemas.

    Measured: UCI Adult has the LARGEST domain of every schema tested (4.4e13) and is the one that
    fits, while a healthcare prefix 6.5 million times smaller times out. Within one dataset the arm
    with the LARGER domain fit 425 s FASTER. This test pins the behaviour that follows: a schema
    with a huge domain but a narrow, low-cardinality shape must NOT be refused.
    """
    # Adult's shape: 15 columns, few wide numerics, ~26k rows. Huge domain, and it fits.
    adult_like = Schema(
        name="adult_like",
        numerical={"age": (17, 90), "hours": (1, 99), "edu_num": (1, 16)},
        categorical={f"c{i}": [f"v{j}" for j in range(14)] for i in range(11)},
        target="y", positive="YES", negative="NO",
        bins={"age": [17, 30, 45, 60, 90], "hours": [1, 35, 45, 99], "edu_num": [1, 9, 13, 16]})
    rep = check_feasible(adult_like, n_rows=26_048, raise_on_fail=False)
    assert rep["domain_size"] > 1e12, "fixture should have a large domain to be meaningful"
    assert rep["feasible"], (
        f"a large-domain but narrow schema was refused: {rep['reasons']}. Domain size is not a "
        f"predictor of convergence and must not gate.")
    assert rep["domain_size_is_not_predictive"] is True


def test_row_count_is_part_of_the_gate():
    """The same schema at the same domain moved from a 26-minute fit to a timeout purely by going
    from 26,048 rows to 81,410. A gate that ignores n cannot see that."""
    s = Schema(name="rows", numerical={f"n{i}": (0, 100) for i in range(3)},
               categorical={f"c{i}": [f"v{j}" for j in range(4)] for i in range(8)},
               target="y", positive="YES", negative="NO",
               bins={f"n{i}": [0, 50, 100] for i in range(3)})
    assert check_feasible(s, n_rows=26_048, raise_on_fail=False)["feasible"]
    assert not check_feasible(s, n_rows=81_410, raise_on_fail=False)["feasible"]


def test_omitting_n_rows_says_so_rather_than_silently_skipping_a_check():
    s = schema()
    rep = check_feasible(s, raise_on_fail=False)
    assert any("n_rows was not supplied" in n for n in rep.get("notes", [])), \
        "skipping one of the three predictive factors must be stated, not silent"


def test_small_schema_is_allowed():
    rep = check_feasible(schema(), n_rows=5_000)
    assert rep["feasible"] and rep["domain_size"] == estimate_domain_size(schema())


# ── privacy ─────────────────────────────────────────────────────────────────────────

def test_table_release_stays_within_its_declared_budget():
    t = release_conditional_table(schema(), frame(), ("edu",), epsilon=0.5, n_min=150, seed=3)
    assert t.audit["within_budget"]
    assert t.audit["epsilon_accounted"] <= 0.5 + 1e-9


def test_many_cells_cost_one_query_by_parallel_composition():
    """Cells partition the data, so a richer table costs no more epsilon — the property that
    makes richness affordable, and richness is what makes the correction work."""
    a = release_conditional_table(schema(), frame(), ("edu",), epsilon=0.5, n_min=150, seed=3)
    b = release_conditional_table(schema(), frame(), ("edu", "region"), epsilon=0.5,
                                  n_min=150, seed=3)
    assert len(b) > len(a)
    assert b.audit["epsilon_accounted"] == pytest.approx(a.audit["epsilon_accounted"])


def test_relabelling_spends_no_additional_budget():
    """Relabelling is post-processing of two already-DP artifacts."""
    t = release_conditional_table(schema(), frame(), ("edu",), epsilon=0.5, n_min=150, seed=3)
    spent = t.audit["epsilon_accounted"]
    relabel(schema(), marginal_synth_output(), t, seed=0)
    assert t.audit["epsilon_accounted"] == spent


def test_bad_data_rejected_before_budget_is_spent():
    s = schema()
    df = frame().drop(columns=["edu"])
    with pytest.raises(DataValidationError, match="NO privacy budget was spent"):
        release_conditional_table(s, df, ("edu",), epsilon=0.5)
    # a cell column the schema does not declare is refused by name, not by a pandas KeyError
    with pytest.raises(ValueError, match="'edu_level' is not in the schema"):
        release_conditional_table(s, frame(), ("edu_level",), epsilon=0.5)
    with pytest.raises(ValueError, match="is the target"):
        release_conditional_table(s, frame(), (s.target,), epsilon=0.5)


def test_n_min_floor_is_enforced():
    with pytest.raises(ValueError, match="safe floor"):
        release_conditional_table(schema(), frame(), ("edu",), epsilon=0.5, n_min=10)


def test_empty_table_is_refused_rather_than_silently_correcting_nothing(capsys):
    from cortec.report import REFUSAL_TYPES, guard
    from cortec_hybrid import CorrectionError
    with pytest.raises(CorrectionError, match="correct nothing"):
        release_conditional_table(schema(), frame(n=300), ("edu", "region"),
                                  epsilon=0.5, n_min=250)
    # the refusal is registered with cortec's guard, so a run script prints it in the standard
    # layout and exits instead of showing a traceback
    assert CorrectionError in REFUSAL_TYPES
    with pytest.raises(SystemExit) as e, guard():
        release_conditional_table(schema(), frame(n=300), ("edu", "region"),
                                  epsilon=0.5, n_min=250)
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "cortec-hybrid" in out and "Refused · Correction" in out
    assert "correct nothing" in out and "not a fault in the tool" in out


# ── the correction itself ───────────────────────────────────────────────────────────

def test_correction_recovers_the_counterintuitive_rate():
    """The whole point: a marginal synthesiser whose target is unrelated to features gets its
    per-cell rates restored, including where they contradict a naive expectation."""
    s, priv = schema(), frame()
    syn = marginal_synth_output()
    out, table, gain = correct(s, priv, syn, columns=("edu",), epsilon=0.5, seed=0)
    for cell in ("edu=Grad", "edu=HS"):
        got = (out[out.edu == cell.split("=")[1]]["y"] == "YES").mean()
        assert abs(got - table.cells[cell]) < 0.02, f"{cell} rate not recovered"
    # and the direction is the counterintuitive one present in the private data
    assert (out[out.edu == "Grad"]["y"] == "YES").mean() < \
           (out[out.edu == "HS"]["y"] == "YES").mean()
    assert gain["conditional_error_reduction"] > 0


def test_correction_leaves_feature_marginals_untouched():
    """Only the target column is rewritten; a correction that wrecked marginals is not a win."""
    s, priv = schema(), frame()
    syn = marginal_synth_output()
    out, _, gain = correct(s, priv, syn, columns=("edu",), epsilon=0.5, seed=0)
    for c in ("age", "edu", "region"):
        pd.testing.assert_series_equal(out[c], syn[c], check_names=False)
    assert gain["feature_marginal_shift"] == pytest.approx(0.0, abs=1e-12)


def test_ranking_is_skipped_when_the_inputs_own_ordering_is_noise():
    """We assumed rank preservation always helped. It does not, and this test is why the code
    now measures instead of assuming: when the input synthetic data's target is unpredictable
    from its features (the case this tool exists for), a model fitted on it learns noise, and
    ordering by that noise cost 0.09 AUC against plain i.i.d. assignment."""
    from cortec_hybrid.core import _rank_scores
    s = schema()
    assert _rank_scores(s, marginal_synth_output(), seed=0) is None, \
        "a noise target must not be used as a ranking signal"
    # but a synthetic set whose target IS predictable from its features keeps its ordering
    informative = frame(n=1500, seed=5)
    assert _rank_scores(s, informative, seed=0) is not None


def test_rank_preservation_never_underperforms_iid_after_the_fix():
    """With the informativeness check in place, the ranked path must not lose to i.i.d. — that
    regression is exactly what the check exists to prevent."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import OrdinalEncoder

    s, priv = schema(), frame()
    syn = marginal_synth_output()
    t = release_conditional_table(s, priv, ("edu",), epsilon=0.5, n_min=150, seed=3)
    ranked = relabel(s, syn, t, preserve_ranking=True, seed=0)
    iid = relabel(s, syn, t, preserve_ranking=False, seed=0)

    test = frame(n=2000, seed=99)

    def tstr(train):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cats = ["edu", "region"]
        Xtr = train[["age"] + cats].copy(); Xte = test[["age"] + cats].copy()
        enc.fit(pd.concat([Xtr[cats], Xte[cats]]).astype(str))
        Xtr[cats] = enc.transform(Xtr[cats].astype(str))
        Xte[cats] = enc.transform(Xte[cats].astype(str))
        ytr = (train["y"] == "YES").astype(int)
        if ytr.nunique() < 2:
            return 0.5
        m = RandomForestClassifier(n_estimators=120, random_state=0).fit(Xtr, ytr)
        return roc_auc_score((test["y"] == "YES").astype(int), m.predict_proba(Xte)[:, 1])

    # both reproduce the per-cell RATE; they differ in whether within-cell ordering survives
    assert tstr(ranked) >= tstr(iid) - 1e-9


def test_cells_with_no_released_rate_are_left_alone():
    """A cell suppressed for low support must not be silently rewritten."""
    s, priv = schema(), frame()
    syn = marginal_synth_output()
    t = release_conditional_table(s, priv, ("edu",), epsilon=0.5, n_min=150, seed=3)
    t.cells.pop("edu=Grad", None)
    out = relabel(s, syn, t, seed=0)
    grad = syn.edu == "Grad"
    pd.testing.assert_series_equal(out.loc[grad, "y"], syn.loc[grad, "y"], check_names=False)


def test_estimate_gain_reports_no_win_when_there_is_none():
    """Honest reporting: if the synthesiser was already calibrated, say so rather than claiming
    an improvement."""
    s, priv = schema(), frame()
    t = release_conditional_table(s, priv, ("edu",), epsilon=0.5, n_min=150, seed=3)
    already = relabel(s, marginal_synth_output(), t, seed=0)   # already corrected
    gain = estimate_gain(s, already, already, t)
    assert gain["conditional_error_reduction"] == pytest.approx(0.0)
    assert not gain["worth_it"]
    assert "consider keeping the budget" in gain["verdict"]


def test_published_support_is_noised_not_exact_private_counts():
    """Regression: `ConditionalTable.support` published EXACT private cell counts.

    The rate beside it was correctly noised and charged; the count was neither, so the audit
    reported a clean epsilon while the release carried the true number of records in every cell.
    This is the same defect that was found in cortec's release.py, and it survived here because
    the whole test module failed at collection (no `cortec` on the path) — see tests/conftest.py.

    Asserted against the OUTPUT, not the accounting: a ledger can only total the queries it is
    told about, which is precisely why the leak was invisible.
    """
    import numpy as np
    import pandas as pd
    from cortec.schema import Schema
    from cortec_hybrid.core import release_conditional_table

    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "age": rng.integers(20, 80, n),
        "grp": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
        "y": rng.choice(["YES", "NO"], n, p=[0.3, 0.7]),
    })
    sc = Schema(name="t", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]},
                target="y", positive="YES", negative="NO")

    runs = [sorted(release_conditional_table(sc, df, ("grp",), epsilon=1.0, n_min=150,
                                             seed=s).support.values())
            for s in (1, 2, 3, 4)]
    truth = sorted(df["grp"].value_counts().tolist())

    assert any(r != runs[0] for r in runs), (
        "support is identical across seeds, so it is not noised — the release publishes exact "
        "counts of private records")
    assert all(r != truth for r in runs), "a release reproduced the true cell counts exactly"
    assert all(v >= 0 for r in runs for v in r), "counts must stay non-negative"


def test_published_support_is_charged_to_the_ledger():
    """The counts must appear in the accounting, in their own group, and spend the full budget."""
    import numpy as np
    import pandas as pd
    from cortec.schema import Schema
    from cortec_hybrid.core import release_conditional_table

    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "age": rng.integers(20, 80, n),
        "grp": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
        "y": rng.choice(["YES", "NO"], n, p=[0.3, 0.7]),
    })
    sc = Schema(name="t", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]},
                target="y", positive="YES", negative="NO")
    t = release_conditional_table(sc, df, ("grp",), epsilon=1.0, n_min=150, seed=1)

    groups = t.audit["groups"]
    assert "published_counts" in groups, "count queries are not in the ledger"
    assert "conditional" in groups, "rate queries are not in the ledger"
    # counts and rates read the same records, so they compose SEQUENTIALLY across groups
    assert t.audit["epsilon_accounted"] == pytest.approx(1.0, abs=1e-9)


def test_relabel_rate_matching_is_unbiased_in_small_cells():
    """Regression: `relabel` used plain round(), which biases the very cells it exists to correct.

    At a released rate of 0.30, deterministic rounding gives a realised rate of 0.000 for a 1-row
    cell (every positive vanishes), 0.500 for 2 rows and 0.400 for 5. Stochastic rounding holds
    0.300 throughout. A rate-matching tool must not bias the rates it matches.
    """
    import numpy as np
    import pandas as pd
    from cortec.schema import Schema
    from cortec_hybrid.core import relabel, ConditionalTable

    sc = Schema(name="t", numerical={"x": (0.0, 1.0)}, categorical={"grp": ["a"]},
                target="y", positive="YES", negative="NO")
    table = ConditionalTable(("grp",), {"grp=a": 0.30}, {"grp=a": 500}, 1.0, {})

    realised = []
    for trial in range(400):
        # a 3-row cell: 0.30 * 3 = 0.9, which plain round() sends to 1 every time (0.333)
        syn = pd.DataFrame({"x": [0.1, 0.2, 0.3], "grp": ["a"] * 3, "y": ["NO"] * 3})
        out = relabel(sc, syn, table, preserve_ranking=False, seed=trial)
        realised.append((out["y"] == "YES").mean())

    mean_rate = float(np.mean(realised))
    assert abs(mean_rate - 0.30) < 0.04, (
        f"realised rate {mean_rate:.3f} against a released 0.30 — rounding is biased")
    assert len(set(realised)) > 1, "every trial gave the same count: rounding is deterministic"


def test_relabel_defaults_to_iid_not_rank_preservation():
    """Regression: rank preservation was the default and measured WORSE in 9 of 9 runs.

    `_rank_scores` gates on whether the synthetic target is predictable from the synthetic
    features. A synthesiser generates its target AS a function of those features, so that score is
    near-perfect regardless of whether the relationship is CORRECT — renal_registry scored CV AUC
    0.981 while its real downstream AUC was 0.631. The gate measured self-consistency, not
    validity, and was highest exactly when the synthesiser had confidently learned a wrong
    relationship, so it selected FOR the failure it was meant to prevent.
    """
    import inspect
    from cortec_hybrid.core import relabel, correct

    assert inspect.signature(relabel).parameters["preserve_ranking"].default is False, (
        "relabel must default to i.i.d. assignment; rank preservation cost up to 0.085 AUC")
    src = inspect.getsource(correct)
    assert "preserve_ranking=False" in src, "correct() must not re-enable rank preservation"


def test_the_same_schema_gives_the_same_cells_in_both_tools():
    """A Schema carrying a coarsening must key cells identically here and in `cortec`.

    These are two tools over one Schema type. `cortec.release._cell_keys` applies a declared
    coarsening; this one used to ignore it, so moving a schema between the tools silently changed
    the conditional table it produced -- and nothing in either output said so.
    """
    from dataclasses import replace
    import pandas as pd
    from cortec.release import _cell_keys as cortec_keys
    from cortec_hybrid.core import _cell_keys as hybrid_keys
    from cortec.schema import Schema

    sc = Schema(name="x", numerical={"age": (0, 100)},
                categorical={"code": ["a", "b", "c", "d"]},
                target="y", positive="YES", negative="NO",
                bins={"age": [0, 50, 100]},
                coarsen={"code": {"a": "g0", "b": "g0", "c": "g1", "d": "g1"}})
    df = pd.DataFrame({"age": [10, 60, 30, 80], "code": ["a", "b", "c", "d"],
                       "y": ["YES", "NO", "YES", "NO"]})
    a = list(cortec_keys(df, ("code",), sc))
    b = list(hybrid_keys(df, ("code",), sc))
    assert a == b, f"cell keys diverge between the tools: {a} vs {b}"
    assert all("=g" in k for k in b), f"coarsening not applied in hybrid: {b}"

    # and with no coarsening declared, both still agree and pin raw values
    sc2 = replace(sc, coarsen={})
    assert list(cortec_keys(df, ("code",), sc2)) == list(hybrid_keys(df, ("code",), sc2))


def test_hybrid_records_the_privacy_unit():
    import numpy as np, pandas as pd
    from cortec.schema import Schema
    from cortec_hybrid.core import release_conditional_table

    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"age": rng.integers(18, 90, n),
                       "code": rng.choice(["a", "b"], n),
                       "y": rng.choice(["YES", "NO"], n)})
    sc = Schema(name="x", numerical={"age": (18, 90)}, categorical={"code": ["a", "b"]},
                target="y", positive="YES", negative="NO", bins={"age": [18, 50, 90]})
    tbl = release_conditional_table(sc, df, ("code",), epsilon=0.5, n_min=150,
                                    max_rows_per_person=7)
    assert tbl.audit["max_rows_per_person"] == 7
    assert tbl.audit["epsilon_per_person"] == pytest.approx(7 * tbl.audit["epsilon_accounted"])


def test_a_vacuous_declared_privacy_unit_is_refused_here_too():
    """This tool is the one most likely to be pointed at an institution's existing table.

    It corrects a marginal synthesiser's output, so it gets aimed at whatever the institution
    already runs -- including encounter-level clinical data, which is exactly where a row-level
    epsilon gets read as a per-person one. It accepted `max_rows_per_person` and acted on a vacuous
    value only by reporting it, while `cortec` refused one.
    """
    import numpy as np
    import pandas as pd
    import pytest

    from cortec.schema import Schema
    from cortec_hybrid.core import release_conditional_table

    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"age": rng.integers(20, 70, n),
                       "grp": rng.choice(["a", "b"], n),
                       "y": rng.choice(["Y", "N"], n)})
    sc = Schema(name="t", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b"]},
                target="y", positive="Y", negative="N")

    with pytest.raises(ValueError) as e:
        release_conditional_table(sc, df, ("grp",), epsilon=2.0, max_rows_per_person=40)
    assert "epsilon_per_person=80" in str(e.value)
    assert "aggregate to one row per person" in str(e.value).lower()

    # acknowledged, it proceeds and the ledger records the real number
    tbl = release_conditional_table(sc, df, ("grp",), epsilon=2.0, max_rows_per_person=40,
                                    acknowledge_vacuous_privacy_unit=True)
    assert tbl.audit["epsilon_per_person"] == 80.0
    assert tbl.audit["epsilon_per_person_vacuous"] is True

    # and a declared one-row-per-person release stays clean
    tbl = release_conditional_table(sc, df, ("grp",), epsilon=2.0, max_rows_per_person=1)
    assert tbl.audit["epsilon_per_person"] == 2.0
    assert tbl.audit["epsilon_per_person_vacuous"] is False


def test_the_audit_inherits_the_not_a_privacy_proof_framing():
    """The hybrid reuses cortec's ledger, so it must inherit the disclaimer, not reimplement it.

    Its `table.audit` is the same class of artefact as a Stage A release audit: a machine-readable
    file a compliance team files, asserting a DP guarantee. It must state what it is not, and where
    the guarantee stops, for the same reason Stage C was renamed away from "certificate".
    """
    import numpy as np
    import pandas as pd

    from cortec.schema import Schema
    from cortec_hybrid.core import release_conditional_table

    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"age": rng.integers(20, 70, n),
                       "grp": rng.choice(["a", "b"], n),
                       "y": rng.choice(["Y", "N"], n)})
    sc = Schema(name="t", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b"]},
                target="y", positive="Y", negative="N")
    audit = release_conditional_table(sc, df, ("grp",), epsilon=2.0,
                                      max_rows_per_person=1).audit

    assert next(iter(audit)) == "_what_this_is"
    is_not = audit["_what_this_is"]["is_not"].lower()
    assert "privacy audit" in is_not and "certificate" in is_not
    assert len(audit["where_this_guarantee_does_not_hold"]) >= 4
    names: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                names.append(k)
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)
    walk(audit)
    assert not [k for k in names if "certif" in k.lower()]


def test_worth_it_says_what_it_does_not_predict():
    """`worth_it` is a CALIBRATION criterion and users read it as a recommendation.

    We measured the two coming apart: across 11 datasets the correction improved calibration in
    essentially every case while downstream AUC moved in both directions -- of 9 runs flagged
    worth_it, downstream AUC fell on 6 in one run and 3 of 10 in a repeat. The paper documents that;
    the tool did not, so the user holding the verdict never saw it.
    """
    import numpy as np
    import pandas as pd

    from cortec.schema import Schema
    from cortec_hybrid.core import estimate_gain, release_conditional_table, relabel

    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"age": rng.integers(20, 70, n),
                       "grp": rng.choice(["a", "b"], n),
                       "y": rng.choice(["Y", "N"], n)})
    sc = Schema(name="t", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b"]},
                target="y", positive="Y", negative="N")
    tbl = release_conditional_table(sc, df, ("grp",), epsilon=2.0, max_rows_per_person=1)
    after = relabel(sc, df, tbl, seed=0)
    g = estimate_gain(sc, df, after, tbl)

    assert "worth_it" in g
    assert "calibration" in g["worth_it_measures"].lower()
    nd = g["worth_it_does_not_predict"].lower()
    assert "downstream" in nd, "must name what it does not predict"
    assert "both directions" in nd, "must say the direction was not stable"
    assert "held-out" in nd, "must tell the user what to do instead"


def test_published_support_is_never_below_n_min():
    """Same rule as cortec's release: a cell is in the table only because it holds >= n_min
    records, so its published (noised) support is clamped there rather than at zero."""
    from cortec_hybrid.core import release_conditional_table
    sch, df = schema(), frame()
    cols = tuple(sch.categorical_cols[:1])
    supports = []
    for seed in (1, 2, 3, 4):
        tbl = release_conditional_table(sch, df, cols, epsilon=0.05, n_min=150, seed=seed)
        supports += list(tbl.support.values())
    assert supports and min(supports) >= 150 and max(supports) > 150
