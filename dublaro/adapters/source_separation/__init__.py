"""Source separation adapter interfaces."""

from dublaro.adapters.source_separation.base import (
    SourceSeparationAdapter,
    SourceSeparationOptions,
    SourceSeparationResult,
)
from dublaro.adapters.source_separation.fake import FakeSourceSeparationAdapter

__all__ = [
    "FakeSourceSeparationAdapter",
    "SourceSeparationAdapter",
    "SourceSeparationOptions",
    "SourceSeparationResult",
]
