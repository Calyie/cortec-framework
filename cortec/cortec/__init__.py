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

__version__ = "0.3.0"
__all__ = ["transmission_bound", "bound_with_controls", "BoundReport", "BoundResult", "BoundError",
           "PrivacyLedger", "PrivacyAccountingError", "Schema", "Band", "SchemaError",
           "DataValidationError", "Release", "release_statistics", "ReleaseError",
           "Generator", "GenerationError", "ContextTruncationError", "EmptyContentError",
           "FatalAPIError", "ModelRefusalError", "RateLimitedError",
           "check_model", "ModelCapabilityError", "validated_models",
           "derive", "DerivedConfig", "select_to_release", "inclusion_weights", "pool_coverage", "release_subbin_values",
           "__version__"]
