"""Source separation adapter interfaces."""

from dublaro.adapters.source_separation.base import (
    SourceSeparationAdapter,
    SourceSeparationOptions,
    SourceSeparationResult,
)
from dublaro.adapters.source_separation.demucs import DemucsSourceSeparationAdapter
from dublaro.adapters.source_separation.fake import FakeSourceSeparationAdapter

__all__ = [
    "DemucsSourceSeparationAdapter",
    "FakeSourceSeparationAdapter",
    "SourceSeparationAdapter",
    "SourceSeparationOptions",
    "SourceSeparationResult",
]
