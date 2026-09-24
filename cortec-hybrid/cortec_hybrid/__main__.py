"""The `cortec-hybrid` command (also `python -m cortec_hybrid`): the help page. `cortec-hybrid
--help` prints what you can run, in what order, with every argument and default read from the
code; `cortec-hybrid <name>` prints one name's documentation. The command runs nothing else."""
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


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _help.main(cortec_hybrid, argv, tool="cortec-hybrid", module="cortec_hybrid",
                      command="cortec-hybrid", run_order=RUN_ORDER, readme="README.md")


if __name__ == "__main__":
    sys.exit(main())
