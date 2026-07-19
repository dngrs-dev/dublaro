from pathlib import Path

from dublaro.adapters.text_adapter.base import TextAdaptationOptions, TextAdapter
from dublaro.schemas import Transcript


def adapt_transcript_text(
    transcript: Transcript,
    adapter: TextAdapter,
    *,
    target_language: str | None = None,
    source_language: str | None = None,
    max_chars_per_second: float = 16.0,
) -> Transcript:
    resolved_target_language = (
        target_language or transcript.target_language or transcript.source_language
    )
    resolved_source_language = source_language or transcript.source_language

    options = TextAdaptationOptions(
        source_language=resolved_source_language,
        target_language=resolved_target_language,
        max_chars_per_second=max_chars_per_second,
    )

    adapted = transcript.model_copy(deep=True)
    adapted.metadata = {
        **adapted.metadata,
        "text_adapter": adapter.name,
        "text_adapter_max_chars_per_second": str(max_chars_per_second),
    }

    for segment in adapted.segments:
        segment.adapted_text = adapter.adapt_segment(segment, options)

    return adapted


def default_adapted_transcript_path(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(
        f"{transcript_file.stem}.adapted{transcript_file.suffix}"
    )
