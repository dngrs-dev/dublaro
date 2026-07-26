from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dublaro.adapters.asr import TranscriptionOptions
from dublaro.adapters.diarization import DiarizationOptions
from dublaro.audio.ffmpeg import extract_audio_from_video
from dublaro.pipeline.adapt_text import adapt_transcript_text
from dublaro.pipeline.align import build_speech_timeline
from dublaro.pipeline.diarize import diarize_transcript
from dublaro.pipeline.dub_plan import (
    DubAdapters,
    DubArtifactPaths,
    DubOptions,
    DubPaths,
)
from dublaro.pipeline.export import export_dubbed_video
from dublaro.pipeline.fit_speech import fit_generated_speech_to_segments
from dublaro.pipeline.manifest import (
    DubbingArtifactsManifest,
    DubbingOptionsManifest,
    build_dubbing_manifest,
    save_manifest,
)
from dublaro.pipeline.mix import mix_original_audio_with_dubbed_speech
from dublaro.pipeline.resume import (
    load_reusable_synthesized_transcript,
    load_reusable_transcript,
    reusable_file,
)
from dublaro.pipeline.subtitles import save_srt
from dublaro.pipeline.synthesize import synthesize_transcript_speech
from dublaro.pipeline.transcribe import save_transcript, transcribe_audio
from dublaro.pipeline.translate import translate_transcript
from dublaro.schemas import Transcript

DubbingProgressStep = Literal[
    "extract_audio",
    "transcribe",
    "diarize",
    "translate",
    "adapt_text",
    "synthesize",
    "fit_speech",
    "align_speech",
    "mix_original_audio",
    "export_video",
    "export_srt",
    "write_manifest",
]

DubbingProgressStatus = Literal["started", "finished", "failed", "skipped"]

DubbingProgressCallback = Callable[
    [DubbingProgressStep, DubbingProgressStatus, str],
    None,
]


@dataclass(frozen=True)
class DubRunContext:
    paths: DubPaths
    options: DubOptions
    adapters: DubAdapters
    artifact_paths: DubArtifactPaths
    progress_callback: DubbingProgressCallback | None = None


@dataclass(frozen=True)
class SpeechTimingResult:
    transcript: Transcript
    fitted_transcript_path: Path | None = None
    fitted_speech_dir: Path | None = None


@dataclass(frozen=True)
class ExportAudioResult:
    audio_path: Path
    mix_original_audio_path: Path | None = None
    mixed_audio_path: Path | None = None


@dataclass(frozen=True)
class ManifestInputs:
    started_at: datetime
    extracted_audio_path: Path
    source_transcript_path: Path
    diarized_transcript_path: Path | None
    translated_transcript_path: Path
    adapted_transcript_path: Path
    synthesized_transcript_path: Path
    speech_dir: Path
    speech_track_path: Path
    dubbed_video_path: Path
    fitted_transcript_path: Path | None
    fitted_speech_dir: Path | None
    mix_original_audio_path: Path | None
    mixed_audio_path: Path | None
    srt_path: Path | None
    source_transcript: Transcript
    translated_transcript: Transcript
    adapted_transcript: Transcript
    synthesized_transcript: Transcript
    speech_timeline_transcript: Transcript


@contextmanager
def _progress_stage(
    callback: DubbingProgressCallback | None,
    step: DubbingProgressStep,
    message: str,
) -> Iterator[None]:
    if callback is None:
        yield
        return

    callback(step, "started", message)

    try:
        yield
    except Exception:
        callback(step, "failed", message)
        raise

    callback(step, "finished", message)


def _progress_skipped(
    callback: DubbingProgressCallback | None,
    step: DubbingProgressStep,
    message: str,
) -> None:
    if callback is not None:
        callback(step, "skipped", message)


def _extract_audio(context: DubRunContext) -> Path:
    extracted_audio_path = context.artifact_paths.extracted_audio_path

    if context.options.resume and reusable_file(extracted_audio_path):
        _progress_skipped(
            context.progress_callback,
            "extract_audio",
            f"Using existing extracted audio: {extracted_audio_path}.",
        )
        return extracted_audio_path

    with _progress_stage(
        context.progress_callback,
        "extract_audio",
        "Extracting audio from video.",
    ):
        return extract_audio_from_video(
            context.paths.video_path,
            extracted_audio_path,
            sample_rate=context.options.asr_sample_rate,
            channels=1,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )


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


def _synthesize_speech(
    context: DubRunContext,
    adapted_transcript: Transcript,
) -> Transcript:
    synthesized_transcript_path = context.artifact_paths.synthesized_transcript_path

    synthesized_transcript = (
        load_reusable_synthesized_transcript(synthesized_transcript_path)
        if context.options.resume
        else None
    )

    if synthesized_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "synthesize",
            f"Using existing synthesized speech: {synthesized_transcript_path}.",
        )
        return synthesized_transcript

    with _progress_stage(
        context.progress_callback,
        "synthesize",
        "Synthesizing speech clips.",
    ):
        synthesized_transcript = synthesize_transcript_speech(
            adapted_transcript,
            adapter=context.adapters.tts,
            output_dir=context.artifact_paths.speech_dir,
            language=context.options.target_language,
            sample_rate=context.options.speech_sample_rate,
            speaker_voices=context.adapters.speaker_voices,
        )
        save_transcript(synthesized_transcript, synthesized_transcript_path)
        return synthesized_transcript


def _fit_speech_to_timing(
    context: DubRunContext,
    synthesized_transcript: Transcript,
) -> SpeechTimingResult:
    if not context.options.fit_speech:
        return SpeechTimingResult(transcript=synthesized_transcript)

    fitted_transcript_path = context.artifact_paths.fitted_transcript_path
    fitted_speech_dir = context.artifact_paths.fitted_speech_dir

    reusable_fitted_transcript = (
        load_reusable_synthesized_transcript(fitted_transcript_path)
        if context.options.resume
        else None
    )

    if reusable_fitted_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "fit_speech",
            f"Using existing fitted transcript: {fitted_transcript_path}.",
        )
        return SpeechTimingResult(
            transcript=reusable_fitted_transcript,
            fitted_transcript_path=fitted_transcript_path,
            fitted_speech_dir=fitted_speech_dir,
        )

    with _progress_stage(
        context.progress_callback,
        "fit_speech",
        "Fitting overlong speech clips to segment timing.",
    ):
        fitted_transcript = fit_generated_speech_to_segments(
            synthesized_transcript,
            output_dir=fitted_speech_dir,
            max_speedup=context.options.max_speech_speedup,
            min_overrun_seconds=context.options.min_speech_overrun_seconds,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )
        save_transcript(fitted_transcript, fitted_transcript_path)

        return SpeechTimingResult(
            transcript=fitted_transcript,
            fitted_transcript_path=fitted_transcript_path,
            fitted_speech_dir=fitted_speech_dir,
        )


def _align_speech_track(
    context: DubRunContext,
    speech_timeline_transcript: Transcript,
) -> Path:
    speech_track_path = context.artifact_paths.speech_track_path

    if context.options.resume and reusable_file(speech_track_path):
        _progress_skipped(
            context.progress_callback,
            "align_speech",
            f"Using existing speech track: {speech_track_path}.",
        )
        return speech_track_path

    with _progress_stage(
        context.progress_callback,
        "align_speech",
        "Building timed speech track.",
    ):
        return build_speech_timeline(
            speech_timeline_transcript,
            output_path=speech_track_path,
            sample_rate=context.options.speech_sample_rate,
        )


def _prepare_audio_for_export(
    context: DubRunContext,
    speech_timeline_transcript: Transcript,
    speech_track_path: Path,
) -> ExportAudioResult:
    if not context.options.mix_original_audio:
        return ExportAudioResult(audio_path=speech_track_path)

    original_mix_path = context.artifact_paths.mix_original_audio_path
    mixed_path = context.artifact_paths.mixed_audio_path

    if context.options.resume and reusable_file(mixed_path):
        _progress_skipped(
            context.progress_callback,
            "mix_original_audio",
            f"Using existing mixed audio: {mixed_path}.",
        )
        return ExportAudioResult(
            audio_path=mixed_path,
            mix_original_audio_path=original_mix_path,
            mixed_audio_path=mixed_path,
        )

    with _progress_stage(
        context.progress_callback,
        "mix_original_audio",
        "Mixing dubbed speech over original audio.",
    ):
        mix_original_audio_path = original_mix_path

        if context.options.resume and reusable_file(original_mix_path):
            _progress_skipped(
                context.progress_callback,
                "mix_original_audio",
                f"Using existing original mix audio: {original_mix_path}.",
            )
        else:
            mix_original_audio_path = extract_audio_from_video(
                context.paths.video_path,
                original_mix_path,
                sample_rate=context.options.speech_sample_rate,
                channels=1,
                overwrite=context.options.overwrite,
                executable=context.options.ffmpeg_executable,
            )

        mixed_audio_path = mix_original_audio_with_dubbed_speech(
            speech_timeline_transcript,
            original_audio_path=mix_original_audio_path,
            speech_track_path=speech_track_path,
            output_path=mixed_path,
            original_gain=context.options.original_audio_gain,
            ducking_gain=context.options.ducking_gain,
            speech_gain=context.options.speech_gain,
            ducking_margin_seconds=context.options.ducking_margin_seconds,
            ducking_fade_seconds=context.options.ducking_fade_seconds,
        )

        return ExportAudioResult(
            audio_path=mixed_audio_path,
            mix_original_audio_path=mix_original_audio_path,
            mixed_audio_path=mixed_audio_path,
        )


def _export_video(
    context: DubRunContext,
    audio_for_export_path: Path,
) -> Path:
    with _progress_stage(
        context.progress_callback,
        "export_video",
        "Exporting dubbed video.",
    ):
        return export_dubbed_video(
            context.paths.video_path,
            audio_for_export_path,
            context.paths.output_path,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )


def _export_srt(
    context: DubRunContext,
    speech_timeline_transcript: Transcript,
) -> Path | None:
    if not context.options.export_srt:
        return None

    with _progress_stage(
        context.progress_callback,
        "export_srt",
        "Exporting SRT subtitles.",
    ):
        return save_srt(
            speech_timeline_transcript,
            context.artifact_paths.srt_path,
            text_mode=context.options.srt_text_mode,
        )


def _write_manifest(
    context: DubRunContext,
    inputs: ManifestInputs,
) -> Path | None:
    if not context.options.write_manifest:
        return None

    run_options = context.options
    artifacts = context.artifact_paths
    adapters = context.adapters
    resolved_manifest_path = artifacts.manifest_path

    with _progress_stage(
        context.progress_callback,
        "write_manifest",
        "Writing run manifest.",
    ):
        manifest = build_dubbing_manifest(
            started_at=inputs.started_at,
            finished_at=datetime.now(UTC),
            input_video_path=context.paths.video_path,
            output_video_path=inputs.dubbed_video_path,
            source_language=inputs.source_transcript.source_language,
            target_language=run_options.target_language,
            asr_adapter=adapters.asr,
            diarization_adapter=adapters.diarization,
            translation_adapter=adapters.translation,
            text_adapter=adapters.text_adapter,
            tts_adapter=adapters.tts,
            options=DubbingOptionsManifest(
                asr_sample_rate=run_options.asr_sample_rate,
                diarize=run_options.diarize,
                diarization_min_speakers=run_options.diarization_min_speakers,
                diarization_max_speakers=run_options.diarization_max_speakers,
                speech_sample_rate=run_options.speech_sample_rate,
                fit_speech=run_options.fit_speech,
                max_speech_speedup=run_options.max_speech_speedup,
                min_speech_overrun_seconds=run_options.min_speech_overrun_seconds,
                mix_original_audio=run_options.mix_original_audio,
                original_audio_gain=run_options.original_audio_gain,
                ducking_gain=run_options.ducking_gain,
                speech_gain=run_options.speech_gain,
                ducking_margin_seconds=run_options.ducking_margin_seconds,
                ducking_fade_seconds=run_options.ducking_fade_seconds,
                translation_group_segments=run_options.translation_group_segments,
                max_translation_group_pause_seconds=(
                    run_options.max_translation_group_pause_seconds
                ),
                max_translation_group_duration_seconds=(
                    run_options.max_translation_group_duration_seconds
                ),
                export_srt=run_options.export_srt,
                srt_text_mode=run_options.srt_text_mode,
                ffmpeg_executable=run_options.ffmpeg_executable,
                resume=run_options.resume,
                overwrite=run_options.overwrite,
            ),
            artifacts=DubbingArtifactsManifest(
                workspace_dir=str(context.paths.workspace_dir),
                extracted_audio_path=str(inputs.extracted_audio_path),
                source_transcript_path=str(inputs.source_transcript_path),
                diarized_transcript_path=(
                    str(inputs.diarized_transcript_path)
                    if inputs.diarized_transcript_path is not None
                    else None
                ),
                translated_transcript_path=str(inputs.translated_transcript_path),
                adapted_transcript_path=str(inputs.adapted_transcript_path),
                synthesized_transcript_path=str(inputs.synthesized_transcript_path),
                speech_dir=str(inputs.speech_dir),
                speech_track_path=str(inputs.speech_track_path),
                dubbed_video_path=str(inputs.dubbed_video_path),
                fitted_transcript_path=(
                    str(inputs.fitted_transcript_path)
                    if inputs.fitted_transcript_path is not None
                    else None
                ),
                fitted_speech_dir=(
                    str(inputs.fitted_speech_dir)
                    if inputs.fitted_speech_dir is not None
                    else None
                ),
                mix_original_audio_path=(
                    str(inputs.mix_original_audio_path)
                    if inputs.mix_original_audio_path is not None
                    else None
                ),
                mixed_audio_path=(
                    str(inputs.mixed_audio_path)
                    if inputs.mixed_audio_path is not None
                    else None
                ),
                srt_path=str(inputs.srt_path) if inputs.srt_path is not None else None,
                manifest_path=str(resolved_manifest_path),
            ),
            metadata={
                "source_segment_count": str(len(inputs.source_transcript.segments)),
                "translated_segment_count": str(
                    len(inputs.translated_transcript.segments)
                ),
                "adapted_segment_count": str(len(inputs.adapted_transcript.segments)),
                "synthesized_segment_count": str(
                    len(inputs.synthesized_transcript.segments)
                ),
                "final_speech_segment_count": str(
                    len(inputs.speech_timeline_transcript.segments)
                ),
            },
        )

        return save_manifest(manifest, resolved_manifest_path)
