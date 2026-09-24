"""The `cortec-hybrid` command (also `python -m cortec_hybrid`). `cortec-hybrid run ...` corrects
a synthetic table you already generated (README step 1); `cortec-hybrid --help` prints what you
can run; `cortec-hybrid <name>` prints one function or class with its arguments and defaults."""
import sys

import cortec_hybrid
from cortec import help as _help

RUN_ORDER = [                      # the value column fits the page width after the key column
    ("1  check_feasible", "refuse a schema the synthesiser cannot fit (README 5)"),
    ("2  correct", "release the table, relabel, measure gain (README 3)"),
    ("   release_conditional_table", "the release step alone: DP P(target | cell) (README 3)"),
    ("   relabel", "the relabel step alone: only the target column changes"),
    ("   estimate_gain", "measure step alone, from released values (README 4)"),
    ("   cortec.show", "print and export the result in one layout (README 4)"),
    ("   cortec.install_guard", "print a refusal in that layout, not a traceback"),
]

GROUPS = [
    ("results", ["ConditionalTable"]),
    ("feasibility", ["estimate_domain_size"]),
]

# the words this tool's output uses, then the shared ones from cortec's glossary
TERMS = [
    ("feasibility", "whether a marginal synthesiser (AIM) is likely to fit this schema, from the "
                    "attribute count, the row count and high-cardinality numeric columns; the "
                    "domain size is reported and not used"),
    ("released conditional table, rate, support", "the DP table P(positive | cell): the rate with "
                                                  "Laplace noise, and the cell size with noise. "
                                                  "Cells with fewer than n_min private records "
                                                  "are not released, and their rows are left as "
                                                  "they were"),
    ("conditional error (before, after)", "the mean absolute gap between each cell's target rate "
                                          "in the table and the released rate, before and after "
                                          "relabelling; `reduction` is the difference"),
    ("feature marginal shift", "the mean total-variation change of the categorical marginals "
                               "between input and output: zero, because only the target column "
                               "is rewritten"),
    ("cells corrected", "released cells whose rows were relabelled"),
    ("worth it (calibration only)", "true when conditional error fell by more than 0.02. A "
                                    "calibration criterion; it does not predict downstream "
                                    "utility, so read the evaluation table beside it"),
] + [t for t in _help.TERMS_CORTEC
     if t[0] in ("epsilon", "privacy unit", "n_min", "1-way TV", "TSTR-LR, TSTR-RF, TSTR-GBM",
                 "ceiling", "floor", "reference rows", "absent categories")]


def run(argv):
    """correct a synthetic table you already have (--help: settings)"""
    from .run import main as run_main
    return run_main(argv)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _help.main(cortec_hybrid, argv, tool="cortec-hybrid", module="cortec_hybrid",
                      command="cortec-hybrid", run_order=RUN_ORDER, groups=GROUPS,
                      readme="README.md", subcommands={"run": run}, terms=TERMS)


if __name__ == "__main__":
    sys.exit(main())
