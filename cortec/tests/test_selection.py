"""Pool-and-rake selection: returning a generated pool to the released marginals.

Each test is named after the failure it prevents. The pool here is synthetic and biased on purpose,
so the selection has something to correct, and the release is a real Stage A release over the
same schema, so the cells the selection targets are the cells the release names."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cortec import Schema, Band, release_statistics, select_to_release, inclusion_weights
from cortec.select import _cells


def _schema():
    return Schema(name="sel", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]},
                  target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 90]},
                  stratify=[("age", [Band("young", 18, 40), Band("old", 40, 90)])],
                  conditional=[("grp",)])


def _private(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    grp = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    y = np.where(rng.random(n) < 0.15 + 0.5 * (age > 60) + 0.2 * (grp == "c"), "YES", "NO")
    return pd.DataFrame({"age": age, "grp": grp, "y": y})


def _biased_pool(n=3000, seed=1):
    """A pool whose marginals are wrong on purpose: too old, too many 'a', too few positives."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n) + rng.integers(0, 15, n)
    age = np.clip(age, 18, 89)
    grp = rng.choice(["a", "b", "c", "invented"], n, p=[0.7, 0.15, 0.1, 0.05])
    y = np.where(rng.random(n) < 0.08 + 0.5 * (age > 60), "YES", "NO")
    return pd.DataFrame({"age": age, "grp": grp, "y": y})


def _marginal_gap(schema, release, df):
    """Mean absolute gap between df's cell shares and the release's, over cohort x class x bins."""
    cells = _cells(schema, df, release)
    total = sum(c["cohort_size"] for c in release.cohorts); gaps = []
    for coh in release.cohorts:
        m = cells["_cohort"].values == coh["cohort_name"]
        gaps.append(abs(m.mean() - coh["cohort_size"] / total))
        for cls, is_pos in (("YES", True), ("NO", False)):
            mm = m & (cells["_pos"].values == is_pos)
            gaps.append(abs(mm.sum() / max(m.sum(), 1) - coh["class_balance"][cls]))
            block = (coh.get("by_class") or {}).get(cls) or coh
            h = np.asarray(block["numerical"]["age"]["proportions"], float); h /= h.sum()
            for b, p in enumerate(h):
                gaps.append(abs((cells["age"].values[mm] == b).mean() - p) if mm.sum() else p)
    return float(np.mean(gaps))


def test_selection_returns_the_pool_to_the_released_marginals():
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    pool = _biased_pool()
    sel = select_to_release(schema, rel, pool, 300, seed=0)
    assert 290 <= len(sel) <= 300
    before, after = _marginal_gap(schema, rel, pool), _marginal_gap(schema, rel, sel)
    assert after < before / 2, f"selection did not move the pool toward the release: {before:.3f} -> {after:.3f}"
    assert set(sel.columns) == set(pool.columns)


def test_a_suppressed_category_is_never_selected():
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    pool = _biased_pool()
    assert (pool.grp == "invented").any()
    sel = select_to_release(schema, rel, pool, 300, seed=1)
    assert not (sel.grp == "invented").any(), "a category absent from the release must get zero mass"


def test_a_pooled_cohort_is_constrained_at_cohort_level_not_per_class():
    """Regression: imposing a POOLED histogram on each class separately forced both classes to the
    same marginals and erased the feature-target structure (Adult held-out conditional error rose
    and GBM utility fell before this was fixed). With class blocks removed from the release, the
    per-class weight sums must follow the class balance and NOT be flattened across classes."""
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, class_conditional=False, seed=3)
    assert not any(c.get("by_class") for c in rel.cohorts)
    pool = _private(3000, seed=9)
    w = inclusion_weights(schema, rel, pool, 300)
    cells = _cells(schema, pool, rel)
    # within the old cohort, positives are far more likely to be old-old (age>60): a per-class
    # pooled constraint would pull the age distribution of positives toward the cohort's mixture
    old = cells["_cohort"].values == "old"
    pos, neg = old & cells["_pos"].values, old & ~cells["_pos"].values
    age_pos = np.average(cells["age"].values[pos], weights=w[pos] + 1e-12)
    age_neg = np.average(cells["age"].values[neg], weights=w[neg] + 1e-12)
    raw_pos, raw_neg = cells["age"].values[pos].mean(), cells["age"].values[neg].mean()
    assert age_pos - age_neg > 0.5 * (raw_pos - raw_neg), \
        "selection flattened the class difference the pool carried"


def test_selection_is_reproducible_with_a_seed_and_refuses_a_short_pool():
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    pool = _biased_pool()
    a = select_to_release(schema, rel, pool, 200, seed=5); b = select_to_release(schema, rel, pool, 200, seed=5)
    assert a.equals(b)
    with pytest.raises(ValueError, match="fewer than"):
        select_to_release(schema, rel, pool.iloc[:100], 300)


def test_generate_selected_is_generate_when_pool_factor_is_one(monkeypatch):
    from cortec.generate import Generator
    schema = _schema()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    g = Generator(schema, backend="anthropic", model="claude-fable-5")
    calls = {}
    def fake_generate(release, n_rows, verbose=True):
        calls["n"] = n_rows; return _biased_pool(n_rows)
    monkeypatch.setattr(g, "generate", fake_generate)
    rel = release_statistics(schema, _private(), epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    out = g.generate_selected(rel, 120, pool_factor=1)
    assert calls["n"] == 120 and len(out) == 120
    out3 = g.generate_selected(rel, 120, pool_factor=3, seed=0)
    assert calls["n"] == 360 and 110 <= len(out3) <= 120


# ── exact per-batch counts ─────────────────────────────────────────────────────────────

def test_batch_quotas_sum_to_the_batch_on_every_column_and_use_only_released_categories():
    from cortec.generate import batch_quotas
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    coh = rel.cohorts[0]
    for n in (7, 20, 25):
        q = batch_quotas(schema, coh, n, np.random.default_rng(1))
        assert q["n"] == n and 0 <= q["positives"] <= n
        assert sum(c for _, c in q["numerical"]["age"]) == n
        assert sum(c for _, c in q["categorical"]["grp"]) == n
        assert all(c in coh["categorical"]["grp"] for c, _ in q["categorical"]["grp"])


def test_quota_prompt_carries_the_counts_and_stays_hash_locked(monkeypatch):
    from cortec.generate import Generator, build_prompt, batch_quotas
    from cortec import prompts
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    coh = rel.cohorts[0]
    q = batch_quotas(schema, coh, 20, np.random.default_rng(2))
    with_q, rec = build_prompt(schema, rel, coh, 20, quotas=q)
    without, _ = build_prompt(schema, rel, coh, 20)
    assert "EXACT COUNTS FOR THIS BATCH OF 20 ROWS" in with_q and f"exactly {q['positives']} rows" in with_q
    assert "EXACT COUNTS" not in without
    assert rec.version == prompts.TEMPLATE_VERSION == "cortec-prompt-1.1.0"
    prompts.verify_integrity()   # the template with the optional block is the locked one
    # the generator option threads the counts through every call
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    g = Generator(schema, backend="anthropic", model="claude-fable-5", quota=True, quota_seed=0)
    seen = {}
    monkeypatch.setattr(g, "_call", lambda prompt: seen.setdefault("p", prompt) or "csv")
    monkeypatch.setattr(g, "parse", lambda text: pd.DataFrame({"age": [45, 52], "grp": ["a", "b"], "y": ["YES", "NO"]}))
    g.generate(rel, 2, verbose=False)
    assert "EXACT COUNTS FOR THIS BATCH OF" in seen["p"]


def test_remaining_quotas_absorb_a_short_batch_so_the_cohort_totals_still_hold():
    """A batch that returns fewer valid rows than asked used to leave the cohort's counts short
    for good: each batch was apportioned afresh from the release. With remaining-count quotas
    the next batch asks for exactly what is still owed."""
    from cortec.generate import cohort_quota_targets, remaining_quotas, emitted_counts, _absorb
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    coh = rel.cohorts[0]; rng = np.random.default_rng(0)
    target = cohort_quota_targets(schema, coh, 60, rng)
    done = {"n": 0, "positives": 0, "numerical": {}, "categorical": {}}

    def honour(q, lose=0):
        n = q["n"]; cols = {}
        for c, items in q["numerical"].items():
            vals = []
            for lab, cnt in items:
                vals += [float(lab.split("-")[0])] * cnt
            cols[c] = (vals + [vals[0]] * n)[:n]
        for c, items in q["categorical"].items():
            vals = []
            for k, cnt in items:
                vals += [k] * cnt
            cols[c] = (vals + [vals[0]] * n)[:n]
        cols["y"] = ["YES"] * q["positives"] + ["NO"] * (n - q["positives"])
        return pd.DataFrame(cols)[["age", "grp", "y"]].iloc[lose:]
    for b, lose in ((25, 4), (25, 0), (14, 0)):
        df = honour(remaining_quotas(target, done, b, rng), lose)
        _absorb(done, emitted_counts(schema, coh, df))
    assert done["n"] == 60 and done["positives"] == target["positives"]
    for kind in ("numerical", "categorical"):
        for c, t in target[kind].items():
            assert {k: v for k, v in done[kind][c].items() if v} == {k: v for k, v in t.items() if v}, (c, t, done[kind][c])


def test_quota_is_the_default_and_the_conditional_share_is_a_fifth(monkeypatch):
    from cortec.generate import Generator
    from cortec import release
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert Generator(_schema(), backend="anthropic", model="claude-fable-5").quota is True
    import inspect
    assert inspect.signature(release.release_statistics).parameters["conditional_fraction"].default == 0.2


def test_refinement_lowers_the_distance_to_the_released_cells():
    """The rake alone leaves integer-resolution error; the swap refinement must bring the selection
    closer to the released cell counts, and must never select a suppressed category."""
    from cortec.select import _release_cell_matrix, refine_selection, inclusion_weights
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    pool = _biased_pool().reset_index(drop=True)
    A, t, w = _release_cell_matrix(schema, rel, pool, 300)
    chosen = np.zeros(len(pool), bool); chosen[np.random.default_rng(0).permutation(len(pool))[:300]] = True
    d0 = float((w * np.abs(A[chosen].sum(0) - t)).sum())
    refined = refine_selection(schema, rel, pool, chosen.copy(), seed=0)
    d1 = float((w * np.abs(A[refined].sum(0) - t)).sum())
    assert refined.sum() == 300 and d1 < 0.5 * d0, f"refinement should close most of the gap: {d0:.1f} -> {d1:.1f}"
    assert not (pool.grp.values[refined] == "invented").any()


def test_selection_respects_cohorts_derived_by_autoconfig():
    """Regression: with autoconfig the caller's Schema has no stratification, so `_cohort_keys`
    mapped every row to one cohort and the selection's cohort constraints matched nothing. The
    release now carries the derived rule, and a selection from an auto-configured release must
    reproduce the released cohort shares."""
    schema = Schema(name="auto", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]},
                    target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 90]})
    private = _private(6000)
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             n_records=1000, seed=3)
    assert rel.stratify and len(rel.cohorts) >= 2
    eff = rel.effective_schema(schema)
    from cortec.release import _cohort_keys
    pool = private.sample(3000, random_state=5).reset_index(drop=True)
    # bias the pool toward one cohort so the selection has to correct it
    keys = _cohort_keys(eff, pool)
    first = rel.cohorts[0]["cohort_name"]
    pool = pd.concat([pool, pool[keys.values == first]], ignore_index=True)
    sel = select_to_release(schema, rel, pool, 300, seed=0)
    total = sum(c["cohort_size"] for c in rel.cohorts)
    shares = _cohort_keys(eff, sel).value_counts(normalize=True)
    for c in rel.cohorts:
        assert abs(float(shares.get(c["cohort_name"], 0.0)) - c["cohort_size"] / total) < 0.03, c["cohort_name"]
    # and the round trip keeps the rule
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "r.json"); rel.to_json(path)
    from cortec import Release
    assert Release.from_json(path).stratify == rel.stratify


def test_a_pool_that_stopped_early_is_refused_and_the_short_cohorts_are_named():
    """A generation run that stops at its spend cap leaves its later cohorts unfilled. Selection can
    only choose among rows that exist, so the tool refuses such a pool and names the cohorts,
    rather than silently returning a synthetic set whose cohort shares are wrong."""
    from cortec.select import pool_coverage
    from cortec.release import _cohort_keys
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    pool = _private(3000, seed=9)
    keys = _cohort_keys(rel.effective_schema(schema), pool)
    truncated = pool[keys.values != "old"].reset_index(drop=True)      # the "old" cohort never generated
    cov = pool_coverage(schema, rel, truncated, 300)
    assert cov["old"]["have"] == 0 and cov["old"]["ratio"] < 1.1
    with pytest.raises(ValueError, match="old has 0 rows"):
        select_to_release(schema, rel, truncated, 300, seed=0)
    with pytest.warns(UserWarning):
        out = select_to_release(schema, rel, truncated, 300, seed=0, allow_short_pool=True)
    assert len(out) <= 300


def test_sub_bin_values_are_the_releases_in_class_block_cohorts_and_the_generators_elsewhere(monkeypatch):
    """The release fixes bin counts and nothing finer. In a cohort with class-conditional blocks the
    generator's placement inside a bin is prior and is replaced by a uniform draw inside the bin
    (never crossing the row's stratification band); in a pooled cohort it is the only carrier of
    the class signal and is kept. No bin count changes, so no binned metric changes."""
    from cortec.select import release_subbin_values
    from cortec.release import _cohort_keys
    schema = _schema(); private = _private(3000, seed=4)
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1, autoconfig=False, seed=3)
    eff = rel.effective_schema(schema)
    cc = {c["cohort_name"] for c in rel.cohorts if c.get("by_class")}
    assert cc and len(cc) < len(rel.cohorts) or cc   # at least one class-block cohort
    out = release_subbin_values(schema, rel, private, seed=0)
    edges = np.asarray(schema.bin_edges("age"), float)
    b0 = np.clip(np.digitize(private["age"].values, edges[1:-1]), 0, len(edges) - 2)
    b1 = np.clip(np.digitize(out["age"].values, edges[1:-1]), 0, len(edges) - 2)
    assert (b0 == b1).all()                                                  # every bin count unchanged
    assert (_cohort_keys(eff, out).values == _cohort_keys(eff, private).values).all()   # no row changes cohort
    assert (out["age"] % 1 == 0).all() and out["age"].between(*schema.numerical["age"]).all()
    coh = _cohort_keys(eff, private).astype(str).values
    in_cc = np.isin(coh, list(cc))
    if (~in_cc).any():
        assert out.loc[~in_cc, "age"].equals(private.loc[~in_cc, "age"])        # pooled cohorts untouched
    assert not out.loc[in_cc, "age"].equals(private.loc[in_cc, "age"])          # class-block cohorts redrawn
    # the selection applies it by default and can be told not to
    pool = _private(2000, seed=9)
    a = select_to_release(schema, rel, pool, 300, seed=0)
    b = select_to_release(schema, rel, pool, 300, seed=0, sub_bin="generator")
    assert len(a) == len(b) and not a["age"].equals(b["age"])
    with pytest.raises(ValueError):
        select_to_release(schema, rel, pool, 300, seed=0, sub_bin="always")
