"""cortec-hybrid — conditional-signal correction for marginal DP synthesisers, no LLM required.

Narrow by design: it corrects the one axis marginal methods leave weakest (conditional target
structure) using a small, separately accounted slice of privacy budget. It is not a general
improvement over its inputs — measure with estimate_gain() and keep the budget if the gain is small.
"""
from .core import (ConditionalTable, DomainTooLargeError, check_feasible, correct,
                   estimate_domain_size, estimate_gain, relabel, release_conditional_table)

__version__ = "0.1.0"
__all__ = ["ConditionalTable", "DomainTooLargeError", "check_feasible", "correct",
           "estimate_domain_size", "estimate_gain", "relabel", "release_conditional_table",
           "__version__"]
