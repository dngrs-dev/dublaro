from dublaro.adapters.translation.base import TranslationAdapter, TranslationOptions
from dublaro.schemas import Transcript


def translate_transcript(
    transcript: Transcript,
    adapter: TranslationAdapter,
    *,
    target_language: str,
    source_language: str | None = None,
) -> Transcript:
    resolved_source_language = source_language or transcript.source_language

    options = TranslationOptions(
        source_language=resolved_source_language,
        target_language=target_language,
    )

    translated = transcript.model_copy(deep=True)
    translated.target_language = target_language
    translated.metadata = {
        **translated.metadata,
        "translation_adapter": adapter.name,
        "translation_source_language": resolved_source_language,
        "translation_target_language": target_language,
    }

    for segment in translated.segments:
        segment.source_language = segment.source_language or resolved_source_language
        segment.target_language = target_language
        segment.translated_text = adapter.translate_text(
            segment.source_text,
            options,
        )

    return translated
