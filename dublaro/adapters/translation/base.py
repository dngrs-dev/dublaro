from typing import Protocol

from pydantic import BaseModel, Field


class TranslationOptions(BaseModel):
    source_language: str | None = None
    target_language: str = Field(min_length=2)
    preserve_timing: bool = True


class TranslationAdapter(Protocol):
    name: str

    def translate_text(self, text: str, options: TranslationOptions) -> str:
        """Translate one text segment."""
        ...
