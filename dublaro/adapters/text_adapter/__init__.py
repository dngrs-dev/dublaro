"""Text adaptation adapter interfaces."""

from dublaro.adapters.text_adapter.base import TextAdaptationOptions, TextAdapter
from dublaro.adapters.text_adapter.fake import FakeTextAdapter
from dublaro.adapters.text_adapter.rules import RuleBasedTextAdapter

__all__ = [
    "FakeTextAdapter",
    "RuleBasedTextAdapter",
    "TextAdaptationOptions",
    "TextAdapter",
]
