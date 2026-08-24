from pathlib import Path

from dublaro.adapters.asr import TranscriptionOptions
from dublaro.adapters.diarization import DiarizationOptions
from dublaro.pipeline.adapt_text import adapt_transcript_text
from dublaro.pipeline.diarize import diarize_transcript
from dublaro.pipeline.dub.context import DubRunContext
from dublaro.pipeline.dub.progress import (
    progress_skipped as _progress_skipped,
)
from dublaro.pipeline.dub.progress import (
    progress_stage as _progress_stage,
)
from dublaro.pipeline.dub.results import TextWorkflowResult
from dublaro.pipeline.dubbing_script import generate_dubbing_script_transcripts
from dublaro.pipeline.resume import load_reusable_transcript
from dublaro.pipeline.transcribe import save_transcript, transcribe_audio
from dublaro.pipeline.translate import translate_transcript
from dublaro.schemas import Transcript


def _transcribe_source_audio(
    context: DubRunContext,
    extracted_audio_path: Path,
) -> Transcript:
    source_transcript_path = context.artifact_paths.source_transcript_path

    source_transcript = (
        load_reusable_transcript(source_transcript_path)
        if context.options.resume
        else None
    )

    if source_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "transcribe",
            f"Using existing source transcript: {source_transcript_path}.",
        )
        return source_transcript

    with _progress_stage(
        context.progress_callback,
        "transcribe",
        "Transcribing source audio.",
    ):
        source_transcript = transcribe_audio(
            extracted_audio_path,
            adapter=context.adapters.asr,
            options=TranscriptionOptions(
                source_language=context.options.source_language,
            ),
        )
        save_transcript(source_transcript, source_transcript_path)
        return source_transcript


def _diarize_source_transcript(
    context: DubRunContext,
    extracted_audio_path: Path,
    source_transcript: Transcript,
) -> Transcript:
    if not context.options.diarize:
        return source_transcript

    if context.adapters.diarization is None:
        raise ValueError("diarization_adapter is required when diarize is True.")

    diarized_path = context.artifact_paths.diarized_transcript_path
    reusable = (
        load_reusable_transcript(diarized_path) if context.options.resume else None
    )

    if reusable is not None:
        _progress_skipped(
            context.progress_callback,
            "diarize",
            f"Using existing diarized transcript: {diarized_path}.",
        )
        return reusable

    with _progress_stage(
        context.progress_callback,
        "diarize",
        "Assigning speakers to transcript segments.",
    ):
        diarized = diarize_transcript(
            extracted_audio_path,
            source_transcript,
            adapter=context.adapters.diarization,
            options=DiarizationOptions(
                min_speakers=context.options.diarization_min_speakers,
                max_speakers=context.options.diarization_max_speakers,
            ),
        )
        save_transcript(diarized, diarized_path)
        return diarized


def _translate_source_transcript(
    context: DubRunContext,
    source_transcript: Transcript,
) -> Transcript:
    translated_transcript_path = context.artifact_paths.translated_transcript_path

    translated_transcript = (
        load_reusable_transcript(translated_transcript_path)
        if context.options.resume
        else None
    )

    if translated_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "translate",
            f"Using existing translated transcript: {translated_transcript_path}.",
        )
        return translated_transcript

    with _progress_stage(
        context.progress_callback,
        "translate",
        f"Translating transcript to {context.options.target_language}.",
    ):
        translated_transcript = translate_transcript(
            source_transcript,
            adapter=context.adapters.translation,
            target_language=context.options.target_language,
            source_language=context.options.source_language,
            group_segments=context.options.translation_group_segments,
            max_group_pause_seconds=(
                context.options.max_translation_group_pause_seconds
            ),
            max_group_duration_seconds=(
                context.options.max_translation_group_duration_seconds
            ),
            max_sentence_group_duration_seconds=(
                context.options.max_translation_sentence_group_duration_seconds
            ),
        )
        save_transcript(translated_transcript, translated_transcript_path)
        return translated_transcript


def _adapt_translated_text(
    context: DubRunContext,
    translated_transcript: Transcript,
) -> Transcript:
    adapted_transcript_path = context.artifact_paths.adapted_transcript_path

    adapted_transcript = (
        load_reusable_transcript(adapted_transcript_path)
        if context.options.resume
        else None
    )

    if adapted_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "adapt_text",
            f"Using existing adapted transcript: {adapted_transcript_path}.",
        )
        return adapted_transcript

    with _progress_stage(
        context.progress_callback,
        "adapt_text",
        "Adapting translated text for timing.",
    ):
        adapted_transcript = adapt_transcript_text(
            translated_transcript,
            adapter=context.adapters.text_adapter,
            target_language=context.options.target_language,
            source_language=context.options.source_language,
        )
        save_transcript(adapted_transcript, adapted_transcript_path)
        return adapted_transcript


def _prepare_text_for_dubbing(
    context: DubRunContext,
    source_transcript: Transcript,
) -> TextWorkflowResult:
    if context.options.text_workflow == "translate-then-adapt":
        translated = _translate_source_transcript(context, source_transcript)
        adapted = _adapt_translated_text(context, translated)
        return TextWorkflowResult(
            translated_transcript=translated,
            adapted_transcript=adapted,
        )

    if context.options.text_workflow == "llm-dubbing":
        return _generate_llm_dubbing_script(context, source_transcript)

    raise ValueError(f"Unsupported text workflow: {context.options.text_workflow}")


def _generate_llm_dubbing_script(
    context: DubRunContext,
    source_transcript: Transcript,
) -> TextWorkflowResult:
    if context.adapters.dubbing_script is None:
        raise ValueError(
            "dubbing_script_adapter is required when text_workflow is llm-dubbing."
        )

    translated_path = context.artifact_paths.translated_transcript_path
    adapted_path = context.artifact_paths.adapted_transcript_path

    if context.options.resume:
        reusable_translated = load_reusable_transcript(translated_path)
        reusable_adapted = load_reusable_transcript(adapted_path)

        if reusable_translated is not None and reusable_adapted is not None:
            _progress_skipped(
                context.progress_callback,
                "dubbing_script",
                f"Using existing LLM dubbing script: {adapted_path}.",
            )
            return TextWorkflowResult(
                translated_transcript=reusable_translated,
                adapted_transcript=reusable_adapted,
            )

    with _progress_stage(
        context.progress_callback,
        "dubbing_script",
        "Generating LLM dubbing script.",
    ):
        result = generate_dubbing_script_transcripts(
            source_transcript,
            adapter=context.adapters.dubbing_script,
            target_language=context.options.target_language,
            source_language=context.options.source_language,
            group_segments=context.options.translation_group_segments,
            max_group_pause_seconds=(
                context.options.max_translation_group_pause_seconds
            ),
            max_group_duration_seconds=(
                context.options.max_translation_group_duration_seconds
            ),
            max_sentence_group_duration_seconds=(
                context.options.max_translation_sentence_group_duration_seconds
            ),
        )
        save_transcript(result.translated, translated_path)
        save_transcript(result.adapted, adapted_path)

        return TextWorkflowResult(
            translated_transcript=result.translated,
            adapted_transcript=result.adapted,
        )
