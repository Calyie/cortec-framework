"""example.py -- a runnable, self-contained tour of cortec's three stages and its output.

No API key and no data files are needed:

    pip install .            # from this directory (or: pip install cortec)
    python example.py

It builds a small SYNTHETIC private table in code, runs the pipeline end to end, and prints each
stage in cortec's standard layout: a boxed header, the named values, the tables, any warnings and
the verdict, with result numbers in blue on a terminal. The differentially private release
(Stage A) and the utility bound (Stage C) are real; generation (Stage B) uses the offline `mock`
backend, so the synthetic NUMBERS are placeholders. Swap the backend for a real model to generate
real rows (the release and the bound are unchanged):

    Generator(schema, backend="anthropic", model="claude-fable-5")    # needs ANTHROPIC_API_KEY
    Generator(schema, backend="gemini",    model="gemini-3.5-flash")  # needs GEMINI_API_KEY

Each `show(...)` also RETURNS a record whose `.save(dir, name)` exports one JSON, one Markdown, one
CSV per table and a figure (see docs/result-format.md)."""
import numpy as np, pandas as pd
from cortec import Schema, release_statistics, Generator, bound_with_controls, show


def synthetic_private_table(n, seed):
    """A stand-in for the private data you would never show a model. Entirely fabricated."""
    rng = np.random.default_rng(seed)
    edu = rng.choice(["HS", "College", "Grad"], size=n, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({
        "age":    rng.integers(18, 90, n),
        "hours":  rng.integers(1, 99, n),
        "edu":    edu,
        "result": rng.choice(["None", "Norm", "High"], size=n, p=[0.6, 0.25, 0.15]),
        "y":      np.where(rng.random(n) < np.where(edu == "Grad", 0.6, 0.2), "YES", "NO"),
    })


schema = Schema(
    name="demo",
    numerical={"age": (18, 90), "hours": (1, 99)},
    categorical={"edu": ["HS", "College", "Grad"], "result": ["None", "Norm", "High"]},
    target="y", positive="YES", negative="NO",
    bins={"age": [18, 30, 45, 60, 90], "hours": [1, 20, 40, 60, 99]},
    stratify=[("edu", [])],
    conditional=[("edu",), ("edu", "result")],
)

train   = synthetic_private_table(2400, seed=0)   # used to build the release
holdout = synthetic_private_table(600,  seed=1)   # never shown to Stage A; the ceiling reads it

# Stage A -- the only step that reads the private data; the whole DP budget is spent here, once.
release = release_statistics(schema, train, epsilon_total=2.0, n_min=150, max_rows_per_person=1)
show(release, n_private_rows=len(train))

# Stage B -- reads only the release, so it costs no further privacy budget.
gen = Generator(schema, backend="mock", model="claude-fable-5", reasoning="on")
synthetic = gen.generate(release, n_rows=300).drop(columns=["_cohort", "_cell"], errors="ignore")
show(gen, n_rows=len(synthetic), positive_rate=float((synthetic["y"] == "YES").mean()))

# Stage C -- a differentially private bound on the private-vs-synthetic conditional gap.
report = bound_with_controls(schema, release, train, synthetic, holdout,
                             epsilon_cert=1.0, alpha=0.05, tolerance=0.15)
show(report, schema=schema.name)
