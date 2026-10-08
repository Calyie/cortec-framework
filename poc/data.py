"""Public data for the proof of concept, built the way the paper builds it.

NHANES 2017-2018: four CDC files (DEMO_J, GHB_J, BMX_J, BPX_J) merged on SEQN; HbA1c >= 6.5% is
the target; ages 18 to 80; an unmeasured HbA1c is unknown, not negative; incomplete cases dropped;
then a stratified 80/20 split with seed 42. UCI Adult: the adult.data file with its documented
column names; "?" kept as a category, as the paper does; strings stripped; the same split. Both are public
benchmarks. Downloads are cached under poc/data/ so a second run needs no network.
"""
from __future__ import annotations
import io, os, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("POC_DATA_DIR", HERE / "data"))   # tests point this at a toy table
CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                 "hours_per_week", "native_country", "income"]


def _get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _split(df: pd.DataFrame, target: str, name: str) -> tuple[Path, Path]:
    train, holdout = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    DATA.mkdir(parents=True, exist_ok=True)
    tr, ho = DATA / f"{name}_train.csv", DATA / f"{name}_holdout.csv"
    train.reset_index(drop=True).to_csv(tr, index=False)
    holdout.reset_index(drop=True).to_csv(ho, index=False)
    return tr, ho


def build_nhanes() -> tuple[Path, Path]:
    raw_p = DATA / "nhanes_2017_raw.csv"
    if raw_p.exists():
        raw = pd.read_csv(raw_p)
    else:
        print("downloading four NHANES 2017-2018 files from the CDC ...", flush=True)
        d = pd.read_sas(io.BytesIO(_get(f"{CDC}/DEMO_J.xpt")), format="xport")[
            ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]]
        for f, cols in (("GHB_J.xpt", ["SEQN", "LBXGH"]), ("BMX_J.xpt", ["SEQN", "BMXBMI", "BMXWAIST"]),
                        ("BPX_J.xpt", ["SEQN", "BPXSY1", "BPXDI1"])):
            d = d.merge(pd.read_sas(io.BytesIO(_get(f"{CDC}/{f}")), format="xport")[cols], on="SEQN", how="inner")
        raw = d
        DATA.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_p, index=False)
    assert len(raw) >= 3000, "unexpected row count in the raw NHANES download"
    bounds = {"age_years": (18, 80), "bmi": (14.0, 60.0), "waist_cm": (55.0, 175.0),
              "systolic_bp": (70, 220), "diastolic_bp": (30, 130)}
    sex = {1: "male", 2: "female"}
    race = {1: "mexican_american", 2: "other_hispanic", 3: "white_nh", 4: "black_nh", 6: "asian_nh", 7: "other_multi"}
    edu = {1: "lt_9th", 2: "9_11th", 3: "hs_grad", 4: "some_college", 5: "college_grad"}
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")  # noqa: E731
    df = pd.DataFrame({"age_years": num("RIDAGEYR"), "bmi": num("BMXBMI"), "waist_cm": num("BMXWAIST"),
                       "systolic_bp": num("BPXSY1"), "diastolic_bp": num("BPXDI1"),
                       "sex": num("RIAGENDR").map(sex), "race_ethnicity": num("RIDRETH3").map(race),
                       "education": num("DMDEDUC2").map(edu)})
    df["income_bracket"] = pd.cut(num("INDFMPIR"), [-0.01, 1.0, 2.0, 4.0, 5.1],
                                  labels=["under_1x", "1_2x", "2_4x", "over_4x"]).astype(object)
    a1c = num("LBXGH")
    df["diabetes"] = np.where(a1c >= 6.5, "YES", "NO")
    df.loc[a1c.isna(), "diabetes"] = np.nan
    df = df[df["age_years"].between(18, 80)]
    for c, (lo, hi) in bounds.items():
        df[c] = df[c].clip(lo, hi)
    df = df.dropna().reset_index(drop=True)
    return _split(df, "diabetes", "nhanes")


def build_adult() -> tuple[Path, Path]:
    raw_p = DATA / "adult.data"
    if not raw_p.exists():
        print("downloading UCI Adult (adult.data) ...", flush=True)
        DATA.mkdir(parents=True, exist_ok=True)
        raw_p.write_bytes(_get(UCI))
    # The paper keeps the rows whose workclass, occupation or native country is "?" and treats "?"
    # as a category, so every one of the 32,561 rows is used; the proof of concept does the same.
    df = pd.read_csv(raw_p, names=ADULT_COLUMNS, skipinitialspace=True, keep_default_na=False)
    df = df.reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return _split(df, "income", "adult")


BUILDERS = {"nhanes": build_nhanes, "adult": build_adult}


def build(name: str) -> tuple[Path, Path]:
    """Return (train_csv, holdout_csv), building them on first use."""
    tr, ho = DATA / f"{name}_train.csv", DATA / f"{name}_holdout.csv"
    if tr.exists() and ho.exists():
        return tr, ho
    return BUILDERS[name]()


if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or ["nhanes", "adult"]):
        tr, ho = build(n)
        a, b = pd.read_csv(tr), pd.read_csv(ho)
        print(f"{n}: train {len(a)} rows, holdout {len(b)} rows -> {tr}, {ho}")
