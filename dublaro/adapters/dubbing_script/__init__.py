"""Dubbing script adapter interfaces."""

from dublaro.adapters.dubbing_script.base import (
    DubbingScriptAdapter,
    DubbingScriptOptions,
    DubbingScriptResult,
)
from dublaro.adapters.dubbing_script.ollama import OllamaDubbingScriptAdapter

__all__ = [
    "DubbingScriptAdapter",
    "DubbingScriptOptions",
    "DubbingScriptResult",
    "OllamaDubbingScriptAdapter",
]
