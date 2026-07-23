from pathlib import Path

from dublaro.adapters.translation.base import TranslationAdapter, TranslationOptions
from dublaro.pipeline.units import group_segments_for_translation, merge_segment_group
from dublaro.schemas import Transcript


def translate_transcript(
    transcript: Transcript,
    adapter: TranslationAdapter,
    *,
    target_language: str,
    source_language: str | None = None,
    group_segments: bool = True,
    max_group_pause_seconds: float = 0.8,
    max_group_duration_seconds: float = 12.0,
) -> Transcript:
    resolved_source_language = source_language or transcript.source_language

    options = TranslationOptions(
        source_language=resolved_source_language,
        target_language=target_language,
    )

    translated = transcript.model_copy(deep=True)
    translated.target_language = target_language

    if group_segments:
        groups = group_segments_for_translation(
            transcript,
            max_pause_seconds=max_group_pause_seconds,
            max_duration_seconds=max_group_duration_seconds,
        )
        translated.segments = [merge_segment_group(group) for group in groups]

    translated.metadata = {
        **translated.metadata,
        "translation_adapter": adapter.name,
        "translation_source_language": resolved_source_language,
        "translation_target_language": target_language,
        "translation_group_segments": str(group_segments),
        "translation_max_group_pause_seconds": str(max_group_pause_seconds),
        "translation_max_group_duration_seconds": str(max_group_duration_seconds),
    }

    for segment in translated.segments:
        segment.source_language = segment.source_language or resolved_source_language
        segment.target_language = target_language
        segment.translated_text = adapter.translate_text(
            segment.source_text,
            options,
        )

    return translated


def default_translated_transcript_path(
    transcript_path: str | Path,
    target_language: str,
) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(
        f"{transcript_file.stem}.{target_language}{transcript_file.suffix}"
    )
