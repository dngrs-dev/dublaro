"""Text adaptation adapter interfaces."""

from dublaro.adapters.text_adapter.base import TextAdaptationOptions, TextAdapter
from dublaro.adapters.text_adapter.fake import FakeTextAdapter
from dublaro.adapters.text_adapter.ollama import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
    OllamaTextAdapter,
    check_ollama_model_available,
)
from dublaro.adapters.text_adapter.rules import RuleBasedTextAdapter

__all__ = [
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_TEMPERATURE",
    "DEFAULT_OLLAMA_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_URL",
    "FakeTextAdapter",
    "OllamaTextAdapter",
    "RuleBasedTextAdapter",
    "TextAdaptationOptions",
    "TextAdapter",
    "check_ollama_model_available",
]
