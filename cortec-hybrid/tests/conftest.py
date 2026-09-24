"""Make the sibling `cortec` package importable.

cortec-hybrid deliberately reuses cortec's audited privacy accounting rather than
reimplementing composition rules, but nothing put that package on the path — so the entire test
module failed at COLLECTION with ModuleNotFoundError and the tool has had zero effective test
coverage. A DP defect (exact private counts published in `ConditionalTable.support`) sat here
undetected as a direct result.
"""
import sys
from pathlib import Path

# Layout-agnostic on purpose. This package's root is parents[1]; the `cortec` package it depends
# on is a SIBLING of that root -- tools/cortec in the research monorepo, cortec-framework/cortec in the
# published tools repository. An earlier version hard-coded parents[3]/"tools"/"cortec", which is
# the monorepo shape only, and the whole suite failed at collection the moment the packages were
# published on their own.
_pkg = Path(__file__).resolve().parents[1]            # .../cortec-hybrid
for p in (_pkg.parent / "cortec", _pkg):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
