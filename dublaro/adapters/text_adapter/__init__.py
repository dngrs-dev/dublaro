"""Text adaptation adapter interfaces."""

from dublaro.adapters.text_adapter.base import TextAdaptationOptions, TextAdapter
from dublaro.adapters.text_adapter.fake import FakeTextAdapter

__all__ = [
    "FakeTextAdapter",
    "TextAdaptationOptions",
    "TextAdapter",
]
