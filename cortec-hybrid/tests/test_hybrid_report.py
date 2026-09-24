"""The correction's standard record: the same layout and exports as cortec's stages."""
import os

import numpy as np
import pandas as pd
import pytest

from cortec import Schema, record, show
from cortec_hybrid import correct


@pytest.fixture(autouse=True)
def _colour_off(monkeypatch):
    monkeypatch.setenv("CORTEC_COLOR", "0")


def _data(seed=0):
    rng = np.random.default_rng(seed)
    grp = rng.choice(["a", "b", "c"], 900)
    rate = {"a": 0.1, "b": 0.3, "c": 0.6}
    y = np.where(rng.uniform(size=900) < np.vectorize(rate.get)(grp), "YES", "NO")
    private = pd.DataFrame({"x": rng.uniform(0, 1, 900), "grp": grp, "y": y})
    synthetic = private.sample(300, random_state=1).reset_index(drop=True)
    synthetic["y"] = rng.permutation(synthetic["y"].values)      # no conditional signal at all
    return private, synthetic


def test_correction_record_reports_before_after_and_the_calibration_only_verdict(tmp_path, capsys):
    schema = Schema(name="toy", numerical={"x": (0, 1)}, categorical={"grp": ["a", "b", "c"]},
                    target="y", positive="YES", negative="NO")
    private, synthetic = _data()
    triple = correct(schema, private, synthetic, columns=("grp",), epsilon=0.5, n_min=50, seed=0)
    rec = show(triple, schema="toy", n_rows=300)
    out = capsys.readouterr().out
    assert rec.tool == "cortec-hybrid" and rec.stage == "Correction" and "── cortec-hybrid" in out
    assert rec.value("released cells") == 3 and rec.value("epsilon for the table") == 0.5
    assert rec.value("conditional error after") < rec.value("conditional error before")
    assert rec.value("worth it (calibration only)") is True and rec.verdict_role == "ok"
    assert rec.table("released conditional table").columns == ["cell", "rate", "support"]
    assert any(n.startswith("It does not predict downstream utility") for n in rec.notes)
    files = rec.save(str(tmp_path), "corr")
    assert os.path.exists(files["json"]) and os.path.exists(files["table_csv:released conditional table"])
    assert record(triple).to_dict()["values"]["cells corrected"] == 3
