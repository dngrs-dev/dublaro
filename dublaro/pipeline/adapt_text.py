from pathlib import Path

from dublaro.adapters.text_adapter.base import TextAdaptationOptions, TextAdapter
from dublaro.schemas import Segment, Transcript


def adapt_transcript_text(
    transcript: Transcript,
    adapter: TextAdapter,
    *,
    target_language: str | None = None,
    source_language: str | None = None,
    max_chars_per_second: float = 16.0,
    preserve_meaning: bool = True,
) -> Transcript:
    resolved_target_language = (
        target_language or transcript.target_language or transcript.source_language
    )
    resolved_source_language = source_language or transcript.source_language

    options = TextAdaptationOptions(
        source_language=resolved_source_language,
        target_language=resolved_target_language,
        max_chars_per_second=max_chars_per_second,
        preserve_meaning=preserve_meaning,
    )

    adapted = transcript.model_copy(deep=True)
    adapted.metadata = {
        **adapted.metadata,
        "text_adapter": adapter.name,
        "text_adapter_max_chars_per_second": str(max_chars_per_second),
        "text_adapter_preserve_meaning": str(preserve_meaning).lower(),
    }

    for segment in adapted.segments:
        original_text = _adaptation_source_text(segment)
        segment.adapted_text = adapter.adapt_segment(segment, options)
        _write_adaptation_metadata(segment, original_text, options)

    return adapted


def default_adapted_transcript_path(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(
        f"{transcript_file.stem}.adapted{transcript_file.suffix}"
    )


def _adaptation_source_text(segment: Segment) -> str:
    return _normalize_spacing(segment.translated_text or segment.source_text)


def _write_adaptation_metadata(
    segment: Segment,
    original_text: str,
    options: TextAdaptationOptions,
) -> None:
    adapted_text = _normalize_spacing(segment.adapted_text)
    budget = _character_budget(segment, options)

    metadata = {
        "adaptation_original_char_count": str(len(original_text)),
        "adaptation_adapted_char_count": str(len(adapted_text)),
        "adaptation_preserve_meaning": str(options.preserve_meaning).lower(),
        "adaptation_over_budget": "false",
    }

    if segment.duration > 0:
        metadata["adaptation_required_chars_per_second"] = (
            f"{len(adapted_text) / segment.duration:.2f}"
        )

    if budget is None:
        metadata["adaptation_status"] = "no_timing_budget"
    else:
        over_budget = len(adapted_text) > budget
        metadata["adaptation_char_budget"] = str(budget)
        metadata["adaptation_over_budget"] = str(over_budget).lower()

        if not over_budget:
            metadata["adaptation_status"] = "fits"
        elif options.preserve_meaning:
            metadata["adaptation_status"] = "over_budget_preserved"
        else:
            metadata["adaptation_status"] = "over_budget"

    segment.metadata = {
        **segment.metadata,
        **metadata,
    }


def _character_budget(
    segment: Segment,
    options: TextAdaptationOptions,
) -> int | None:
    if segment.duration <= 0:
        return None

    return max(8, int(segment.duration * options.max_chars_per_second))


def _normalize_spacing(text: str) -> str:
    return " ".join(text.split())
