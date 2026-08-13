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


class TextAdapterResult(BaseModel):
    text: str = ""
    reason: str | None = None


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
class StructuredTextAdapter(Protocol):
    def adapt_segment_result(
        self,
        segment: Segment,
        options: TextAdaptationOptions,
    ) -> TextAdapterResult:
        """Adapt text and return auditable model metadata."""
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


@runtime_checkable
class StructuredTimingRepairTextAdapter(Protocol):
    def repair_segment_timing_result(
        self,
        segment: Segment,
        options: TextTimingRepairOptions,
    ) -> TextAdapterResult:
        """Repair timing and return auditable model metadata."""
        ...


def adapt_segment_with_result(
    adapter: TextAdapter,
    segment: Segment,
    options: TextAdaptationOptions,
) -> TextAdapterResult:
    if isinstance(adapter, StructuredTextAdapter):
        return adapter.adapt_segment_result(segment, options)

    return TextAdapterResult(text=adapter.adapt_segment(segment, options))


def repair_segment_timing_with_result(
    adapter: TimingRepairTextAdapter,
    segment: Segment,
    options: TextTimingRepairOptions,
) -> TextAdapterResult:
    if isinstance(adapter, StructuredTimingRepairTextAdapter):
        return adapter.repair_segment_timing_result(segment, options)

    return TextAdapterResult(text=adapter.repair_segment_timing(segment, options))
