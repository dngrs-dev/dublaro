from dataclasses import dataclass

from dublaro.adapters.dubbing_script import (
    DubbingScriptAdapter,
    DubbingScriptOptions,
    DubbingScriptResult,
)
from dublaro.pipeline.units import group_segments_for_translation, merge_segment_group
from dublaro.schemas import Segment, Transcript


@dataclass(frozen=True)
class DubbingScriptTranscripts:
    translated: Transcript
    adapted: Transcript


def generate_dubbing_script_transcripts(
    transcript: Transcript,
    *,
    adapter: DubbingScriptAdapter,
    target_language: str,
    source_language: str | None = None,
    group_segments: bool = True,
    max_group_pause_seconds: float = 0.8,
    max_group_duration_seconds: float = 12.0,
    max_sentence_group_duration_seconds: float = 24.0,
    max_chars_per_second: float = 16.0,
    preserve_meaning: bool = True,
) -> DubbingScriptTranscripts:
    resolved_source_language = source_language or transcript.source_language

    options = DubbingScriptOptions(
        source_language=resolved_source_language,
        target_language=target_language,
        max_chars_per_second=max_chars_per_second,
        preserve_meaning=preserve_meaning,
    )

    script_segments = _script_segments(
        transcript,
        group_segments=group_segments,
        max_group_pause_seconds=max_group_pause_seconds,
        max_group_duration_seconds=max_group_duration_seconds,
        max_sentence_group_duration_seconds=max_sentence_group_duration_seconds,
    )

    translated = transcript.model_copy(deep=True)
    translated.target_language = target_language
    translated.segments = [segment.model_copy(deep=True) for segment in script_segments]
    translated.metadata = {
        **translated.metadata,
        "text_workflow": "llm-dubbing",
        "translation_adapter": adapter.name,
        "dubbing_script_adapter": adapter.name,
        "translation_source_language": resolved_source_language,
        "translation_target_language": target_language,
        "translation_group_segments": str(group_segments),
        "translation_max_group_pause_seconds": str(max_group_pause_seconds),
        "translation_max_group_duration_seconds": str(max_group_duration_seconds),
        "translation_max_sentence_group_duration_seconds": str(
            max_sentence_group_duration_seconds
        ),
    }

    adapted = translated.model_copy(deep=True)
    adapted.metadata = {
        **adapted.metadata,
        "text_adapter": adapter.name,
        "text_adapter_max_chars_per_second": str(max_chars_per_second),
        "text_adapter_preserve_meaning": str(preserve_meaning).lower(),
    }

    for translated_segment, adapted_segment in zip(
        translated.segments,
        adapted.segments,
        strict=True,
    ):
        result = adapter.generate_segment_script(adapted_segment, options)
        _apply_dubbing_script_result(
            translated_segment,
            adapted_segment,
            result,
            options,
        )

    return DubbingScriptTranscripts(translated=translated, adapted=adapted)


def _script_segments(
    transcript: Transcript,
    *,
    group_segments: bool,
    max_group_pause_seconds: float,
    max_group_duration_seconds: float,
    max_sentence_group_duration_seconds: float,
) -> list[Segment]:
    if not group_segments:
        return [segment.model_copy(deep=True) for segment in transcript.segments]

    groups = group_segments_for_translation(
        transcript,
        max_pause_seconds=max_group_pause_seconds,
        max_duration_seconds=max_group_duration_seconds,
        max_sentence_duration_seconds=max_sentence_group_duration_seconds,
    )
    return [merge_segment_group(group) for group in groups]


def _apply_dubbing_script_result(
    translated_segment: Segment,
    adapted_segment: Segment,
    result: DubbingScriptResult,
    options: DubbingScriptOptions,
) -> None:
    translated_text = _normalize_spacing(result.translated_text)
    adapted_text = _normalize_spacing(result.adapted_text)

    if not translated_text and adapted_text:
        translated_text = adapted_text
    if not adapted_text and translated_text:
        adapted_text = translated_text

    if _normalize_spacing(adapted_segment.source_text) and (
        not translated_text or not adapted_text
    ):
        raise ValueError(
            f"Dubbing script adapter returned empty text for {adapted_segment.id}."
        )

    for segment in (translated_segment, adapted_segment):
        segment.source_language = segment.source_language or options.source_language
        segment.target_language = options.target_language

    translated_segment.translated_text = translated_text
    translated_segment.adapted_text = ""

    adapted_segment.translated_text = translated_text
    adapted_segment.adapted_text = adapted_text
    _write_dubbing_metadata(adapted_segment, result, options)


def _write_dubbing_metadata(
    segment: Segment,
    result: DubbingScriptResult,
    options: DubbingScriptOptions,
) -> None:
    translated_text = _normalize_spacing(result.translated_text)
    adapted_text = _normalize_spacing(result.adapted_text)
    budget = _character_budget(segment, options)

    metadata = {
        "text_workflow": "llm-dubbing",
        "adaptation_original_char_count": str(len(translated_text)),
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
        metadata["adaptation_status"] = "fits" if not over_budget else "over_budget"

    if result.reason:
        metadata["dubbing_script_reason"] = _normalize_spacing(result.reason)
        metadata["adaptation_reason"] = _normalize_spacing(result.reason)

    segment.metadata = {
        **segment.metadata,
        **metadata,
    }


def _character_budget(
    segment: Segment,
    options: DubbingScriptOptions,
) -> int | None:
    if segment.duration <= 0:
        return None

    return max(8, int(segment.duration * options.max_chars_per_second))


def _normalize_spacing(text: str) -> str:
    return " ".join(text.split())
