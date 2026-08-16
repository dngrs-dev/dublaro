from typing import Protocol

from pydantic import BaseModel, Field

from dublaro.schemas import Segment


class DubbingScriptOptions(BaseModel):
    source_language: str | None = None
    target_language: str = Field(min_length=2)
    max_chars_per_second: float = Field(default=16.0, gt=0)
    preserve_meaning: bool = True


class DubbingScriptResult(BaseModel):
    translated_text: str = ""
    adapted_text: str = ""
    reason: str | None = None


class DubbingScriptAdapter(Protocol):
    name: str

    def generate_segment_script(
        self,
        segment: Segment,
        options: DubbingScriptOptions,
    ) -> DubbingScriptResult:
        """Generate translated and speakable dubbed text in one step."""
        ...
