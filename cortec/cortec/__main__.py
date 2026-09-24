"""`python -m cortec`: the help page. What you can run, in what order, with every argument and
default read from the code. `python -m cortec <name>` prints one name's full documentation."""
import sys

import cortec
from . import help as _help

RUN_ORDER = [                      # the value column fits the page width after the key column
    ("1  Schema", "declare the columns, bounds, bins and the target (README 4)"),
    ("2  release_statistics", "Stage A: spend the budget once on a release (README 5)"),
    ("3  Generator", "Stage B: .generate or .generate_selected from it (README 6)"),
    ("4  bound_with_controls", "Stage C: a DP utility bound with its controls (README 7)"),
    ("5  evaluate", "fidelity and utility beside real references (README 8)"),
    ("   show, record", "print and export any stage in one layout (README 11)"),
    ("   install_guard, guard", "print a refusal in that layout, not a traceback (README 10)"),
]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _help.main(cortec, argv, tool="cortec", module="cortec", run_order=RUN_ORDER,
                      readme="README.md", example="example.py")


if __name__ == "__main__":
    sys.exit(main())
