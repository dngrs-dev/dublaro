from typing import Protocol

from pydantic import BaseModel, Field

from dublaro.schemas import Segment


class TextAdaptationOptions(BaseModel):
    source_language: str | None = None
    target_language: str = Field(min_length=2)
    max_chars_per_second: float = Field(default=16.0, gt=0)
    preserve_meaning: bool = True


class TextAdapter(Protocol):
    name: str

    def adapt_segment(
        self,
        segment: Segment,
        options: TextAdaptationOptions,
    ) -> str:
        """Adapt translated text for spoken dubbing."""
        ...
