"""cortec-hybrid — conditional-signal correction for marginal DP synthesisers, no LLM required.

Narrow by design: it corrects the one axis marginal methods leave weakest (conditional target
structure) using a small, separately accounted slice of privacy budget. It is not a general
improvement over its inputs — measure with estimate_gain() and keep the budget if the gain is small.
"""
from cortec.report import register_refusal
from .core import (ConditionalTable, CorrectionError, DomainTooLargeError, check_feasible, correct,
                   estimate_domain_size, estimate_gain, relabel, release_conditional_table)

# this package's deliberate refusals, printed in cortec's standard layout by `cortec.guard()`
register_refusal(CorrectionError, DomainTooLargeError)

__version__ = "0.1.1"
__all__ = ["ConditionalTable", "CorrectionError", "DomainTooLargeError", "check_feasible", "correct",
           "estimate_domain_size", "estimate_gain", "relabel", "release_conditional_table",
           "__version__"]
