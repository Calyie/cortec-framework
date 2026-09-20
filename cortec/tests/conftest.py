"""Make the `cortec` package importable regardless of the working directory.

Without this the suite only collected when pytest was invoked from inside tools/cortec — run it
from the repo root and you got a collection ERROR, not results. That is exactly how the sibling
tool's DP defect stayed hidden: a suite that cannot be collected reports no failures.
"""
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]      # .../tools/cortec
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
