"""Translation adapter interfaces."""

from dublaro.adapters.translation.argos import ArgosTranslationAdapter
from dublaro.adapters.translation.base import TranslationAdapter, TranslationOptions
from dublaro.adapters.translation.fake import FakeTranslationAdapter
from dublaro.adapters.translation.ollama import (
    DEFAULT_OLLAMA_TRANSLATION_MODEL,
    DEFAULT_OLLAMA_TRANSLATION_TEMPERATURE,
    DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_TRANSLATION_URL,
    OllamaTranslationAdapter,
)

__all__ = [
    "DEFAULT_OLLAMA_TRANSLATION_MODEL",
    "DEFAULT_OLLAMA_TRANSLATION_TEMPERATURE",
    "DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_TRANSLATION_URL",
    "ArgosTranslationAdapter",
    "FakeTranslationAdapter",
    "OllamaTranslationAdapter",
    "TranslationAdapter",
    "TranslationOptions",
]
