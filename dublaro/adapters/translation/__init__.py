"""Translation adapter interfaces."""

from dublaro.adapters.translation.base import TranslationAdapter, TranslationOptions
from dublaro.adapters.translation.fake import FakeTranslationAdapter

__all__ = ["FakeTranslationAdapter", "TranslationAdapter", "TranslationOptions"]
