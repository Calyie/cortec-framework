"""Stage C: the utility transmission bound, as shipped in the package.

Each test is named after the failure it prevents. The release is a real Stage A release over a
small schema whose three conditional cells carry very different target rates, so that a real
sample clears a 0.15 tolerance and a permuted one does not, which is what a verdict depends on."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cortec import Schema, Band, release_statistics, transmission_bound, bound_with_controls, BoundError
from cortec.bound import MIN_SYNTH_ROWS_PER_CELL

EPS_CERT = 1.0


def _schema():
    return Schema(name="bound", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]},
                  target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 90]},
                  stratify=[("age", [Band("young", 18, 40), Band("old", 40, 90)])],
                  conditional=[("grp",)])


def _data(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    grp = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    rate = 0.10 + 0.25 * (grp == "b") + 0.40 * (grp == "c")
    y = np.where(rng.random(n) < rate, "YES", "NO")
    return pd.DataFrame({"age": age, "grp": grp, "y": y})


@pytest.fixture(scope="module")
def world():
    schema = _schema()
    private, holdout = _data(6000, 0), _data(3000, 1)
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    return schema, private, holdout, rel


def test_bound_spends_epsilon_cert_once_in_parallel_and_the_ledger_seals(world):
    schema, private, holdout, rel = world
    rep = bound_with_controls(schema, rel, private, _data(2000, 5), holdout, epsilon_cert=EPS_CERT)
    acc = rep.accounting
    assert abs(acc["epsilon_accounted"] - EPS_CERT) < 1e-9, "three conditions must not spend three times"
    assert acc["n_queries"] == rep.synthetic.n_cells and acc["sealed"] is True
    assert all(q["composition"] == "parallel" for q in acc["queries"])
    assert rep.dp_claim["epsilon_total_per_row"] == pytest.approx(2.0 + EPS_CERT)
    assert rep.dp_claim["epsilon_per_person"] == pytest.approx(2.0 + EPS_CERT)


def test_uncovered_cells_score_the_trivial_bound_so_a_subset_cannot_clear(world):
    schema, private, holdout, rel = world
    only_a = _data(2000, 7)
    only_a = only_a[only_a["grp"] == "a"]
    r = transmission_bound(schema, rel, private, only_a, epsilon_cert=EPS_CERT)
    assert r.n_cells_uncovered == 2 and r.worst_case_bound == 1.0


def test_noise_scale_is_the_public_floor_not_the_private_cell_size(world):
    schema, private, holdout, rel = world
    r = transmission_bound(schema, rel, private, _data(2000, 9), epsilon_cert=EPS_CERT, alpha=0.05)
    sizes = {c.cell: c.n_private for c in r.cells}
    assert max(sizes.values()) > 2 * min(sizes.values()), "the fixture needs cells of different sizes"
    expected = (1.0 / rel.n_min) / EPS_CERT * np.log(r.n_cells / 0.05)
    assert all(abs(c.noise_halfwidth - expected) < 1e-12 for c in r.cells)


def test_no_verdict_is_issued_when_the_controls_do_not_discriminate(world):
    schema, private, holdout, rel = world
    synth = _data(2000, 11)
    loose = bound_with_controls(schema, rel, private, synth, holdout, epsilon_cert=EPS_CERT, tolerance=1.0)
    assert loose.floor.within_bound is True and loose.discriminating is False and loose.verdict is None
    tight = bound_with_controls(schema, rel, private, synth, holdout, epsilon_cert=EPS_CERT, tolerance=0.15)
    assert tight.ceiling.within_bound is True and tight.floor.within_bound is False
    assert tight.discriminating is True and tight.verdict == "within bound"


def test_report_is_headed_utility_transmission_bound_and_uses_real_booleans(world, tmp_path):
    schema, private, holdout, rel = world
    rep = bound_with_controls(schema, rel, private, _data(2000, 13), holdout, epsilon_cert=EPS_CERT)
    path = rep.to_json(str(tmp_path / "bound.json"))
    raw = open(path).read()
    d = json.loads(raw)
    assert list(d.keys())[0] == "_what_this_is" and d["_what_this_is"]["artifact"] == "utility transmission bound"
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k; yield from keys(v)
        elif isinstance(o, list):
            for v in o: yield from keys(v)
    assert not [k for k in keys(d) if "certif" in k.lower()], "no field may be named as a certificate"
    assert d["_verdict"] in ("within bound", "outside tolerance", None)
    assert d["synthetic"]["within_bound"] in (True, False) and not isinstance(d["synthetic"]["within_bound"], str)
    assert "_standards_not_claimed" in d and "HIPAA Expert Determination" in d["_standards_not_claimed"]
    assert "Utility transmission bound" in rep.summary() and "not a privacy audit" in rep.summary()


def test_two_runs_draw_different_noise_so_a_seed_cannot_void_the_spend(world):
    schema, private, holdout, rel = world
    synth = _data(2000, 17)
    a = transmission_bound(schema, rel, private, synth, epsilon_cert=EPS_CERT)
    b = transmission_bound(schema, rel, private, synth, epsilon_cert=EPS_CERT)
    assert [c.p_hat for c in a.cells] != [c.p_hat for c in b.cells]


def test_thin_cells_are_reported_not_trusted(world):
    schema, private, holdout, rel = world
    synth = _data(2000, 19)
    synth = pd.concat([synth[synth["grp"] != "c"], synth[synth["grp"] == "c"].head(5)])
    r = transmission_bound(schema, rel, private, synth, epsilon_cert=EPS_CERT)
    assert r.n_cells_thin == 1 and [c for c in r.cells if c.synth_rows == 5][0].synth_rows < MIN_SYNTH_ROWS_PER_CELL


def test_refuses_to_state_a_per_person_figure_it_was_not_given(world):
    schema, private, holdout, rel = world
    stripped = type(rel)(**{**rel.__dict__, "audit": {}})
    with pytest.raises(BoundError, match="max_rows_per_person"):
        bound_with_controls(schema, stripped, private, _data(500, 21), holdout, epsilon_cert=EPS_CERT)
    ok = bound_with_controls(schema, stripped, private, _data(500, 21), holdout, epsilon_cert=EPS_CERT,
                             max_rows_per_person=3)
    assert ok.dp_claim["max_rows_per_person"] == 3 and ok.dp_claim["epsilon_per_person"] == pytest.approx(3 * (2.0 + EPS_CERT))


def test_private_data_that_does_not_match_the_release_is_refused_before_spending(world):
    schema, private, holdout, rel = world
    wrong = private.copy(); wrong["grp"] = "b"
    with pytest.raises(BoundError, match="does not match the release"):
        transmission_bound(schema, rel, wrong, _data(500, 23), epsilon_cert=EPS_CERT)
