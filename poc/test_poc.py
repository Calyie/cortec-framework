"""Offline smoke test for the proof of concept: every step runs on the mock generator over a toy
table shaped like NHANES, so the pipeline is exercised without a network, a key or a download."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _toy(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    df = pd.DataFrame({
        "age_years": age, "bmi": rng.uniform(16, 45, n).round(1), "waist_cm": rng.uniform(60, 150, n).round(1),
        "systolic_bp": rng.integers(90, 180, n), "diastolic_bp": rng.integers(50, 110, n),
        "sex": rng.choice(["male", "female"], n),
        "race_ethnicity": rng.choice(["mexican_american", "other_hispanic", "white_nh", "black_nh", "asian_nh", "other_multi"], n),
        "education": rng.choice(["lt_9th", "9_11th", "hs_grad", "some_college", "college_grad"], n),
        "income_bracket": rng.choice(["under_1x", "1_2x", "2_4x", "over_4x"], n)})
    p = 0.02 + 0.30 * (age - 18) / 62
    df["diabetes"] = np.where(rng.random(n) < p, "YES", "NO")
    return df


def test_poc_runs_end_to_end_on_the_mock_backend(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    _toy(2400, 0).to_csv(data / "nhanes_train.csv", index=False)
    _toy(600, 1).to_csv(data / "nhanes_holdout.csv", index=False)
    out = tmp_path / "out"
    import os
    # the subprocess must find the package whether or not it is installed: the package root sits beside poc/
    env = {"POC_DATA_DIR": str(data),
           "PYTHONPATH": os.pathsep.join(p for p in (str(HERE.parent / "cortec"), os.environ.get("PYTHONPATH", "")) if p)}
    r = subprocess.run([sys.executable, str(HERE / "run_poc.py"), "--dataset", "nhanes", "--backend", "mock",
                        "--n-rows", "200", "--pool-factor", "2", "--out", str(out)],
                       env={**os.environ, **env}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for f in ("release.json", "synthetic.csv", "bound.json", "REPORT.md", "RESULTS.md", "results.json",
              "models/lr.joblib", "models/rf.joblib", "models/gbm.joblib", "models/scores.json"):
        assert (out / f).exists(), f
    scores = json.loads((out / "models" / "scores.json").read_text())
    assert set(scores["synthetic"]) == {"LR", "RF", "GBM"} and scores["holdout_rows"] == 600
    assert set(scores["reference"]) == {"LR", "RF", "GBM"} and scores["reference_draws"] == 5
    rep = (out / "REPORT.md").read_text()
    for phrase in ("## Results: marginal fidelity and downstream utility against real references", "**marginal fidelity result**",
                   "**downstream utility result**", "share of real-sample utility", "## The saved models",
                   "models/lr.joblib", "--verify"):
        assert phrase in rep, phrase
    assert "Results · marginal fidelity and downstream utility" in r.stdout and "the saved models" in r.stdout
    # the saved models score exactly what the tool's evaluation scored for the synthetic row
    results = json.loads((out / "results.json").read_text())
    g2 = next(t for t in results["tables"] if t["name"] == "downstream utility result")
    for row in g2["rows"]:
        assert abs(row[1] - scores["synthetic"][row[0]]) < 1e-9, row
    # replay and verify work from the artefacts just written, without a backend
    r2 = subprocess.run([sys.executable, str(HERE / "run_poc.py"), "--dataset", "nhanes", "--replay", "--out", str(out)],
                        env={**os.environ, **env}, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    assert "replay the saved artefacts" in r2.stdout and "Results · marginal fidelity and downstream utility" in r2.stdout
    r3 = subprocess.run([sys.executable, str(HERE / "run_poc.py"), "--dataset", "nhanes", "--verify", "--out", str(out)],
                        env={**os.environ, **env}, capture_output=True, text=True)
    assert r3.returncode == 0 and "saved models scored" in r3.stdout
