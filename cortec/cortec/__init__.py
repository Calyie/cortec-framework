"""CoRTeC — differentially private synthetic tabular data from a frozen language model.

The privacy budget is spent once, on a statistics release. Generation reads only that release, so
it is post-processing: it adds no privacy cost, and unlimited datasets may be drawn from one
release.
"""
from .accounting import PrivacyLedger, PrivacyAccountingError
from .schema import Schema, Band, SchemaError, DataValidationError
from .release import Release, release_statistics, ReleaseError
from .generate import (Generator, GenerationError, ContextTruncationError,
                       EmptyContentError, FatalAPIError, ModelRefusalError, RateLimitedError)
from .models import check_model, ModelCapabilityError, validated_models
from .autoconfig import derive, DerivedConfig
from .select import select_to_release, inclusion_weights, pool_coverage, release_subbin_values
from .bound import transmission_bound, bound_with_controls, BoundReport, BoundResult, BoundError
from .prompts import PromptIntegrityError
from .report import RunRecord, ResultTable, record, show, guard, install_guard, register_refusal
from .evaluate import evaluate

# the package's deliberate refusals, printed in the standard layout by `guard()` (README §10)
register_refusal(PrivacyAccountingError, SchemaError, DataValidationError, ReleaseError,
                 GenerationError, RateLimitedError, ModelCapabilityError, BoundError,
                 PromptIntegrityError)

__version__ = "0.4.0"
__all__ = ["transmission_bound", "bound_with_controls", "BoundReport", "BoundResult", "BoundError",
           "PrivacyLedger", "PrivacyAccountingError", "Schema", "Band", "SchemaError",
           "DataValidationError", "Release", "release_statistics", "ReleaseError",
           "Generator", "GenerationError", "ContextTruncationError", "EmptyContentError",
           "FatalAPIError", "ModelRefusalError", "RateLimitedError",
           "check_model", "ModelCapabilityError", "validated_models",
           "derive", "DerivedConfig", "select_to_release", "inclusion_weights", "pool_coverage", "release_subbin_values",
           "RunRecord", "ResultTable", "record", "show", "evaluate",
           "guard", "install_guard", "register_refusal", "PromptIntegrityError",
           "__version__"]
