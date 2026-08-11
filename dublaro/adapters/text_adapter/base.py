from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from dublaro.schemas import Segment


class TextAdaptationOptions(BaseModel):
    source_language: str | None = None
    target_language: str = Field(min_length=2)
    max_chars_per_second: float = Field(default=16.0, gt=0)
    preserve_meaning: bool = True


class TextTimingRepairOptions(BaseModel):
    source_language: str | None = None
    target_language: str = Field(min_length=2)
    target_duration_seconds: float = Field(gt=0)
    current_audio_duration_seconds: float = Field(gt=0)
    max_audio_duration_seconds: float = Field(gt=0)
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
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


@runtime_checkable
class TimingRepairTextAdapter(Protocol):
    name: str

    def repair_segment_timing(
        self,
        segment: Segment,
        options: TextTimingRepairOptions,
    ) -> str:
        """Rewrite spoken text after generated audio is known to be too long."""
        ...
