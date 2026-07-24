from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dublaro.adapters.asr import AsrAdapter, TranscriptionOptions
from dublaro.adapters.text_adapter import TextAdapter
from dublaro.adapters.translation import TranslationAdapter
from dublaro.adapters.tts import TtsAdapter
from dublaro.audio.ffmpeg import extract_audio_from_video
from dublaro.pipeline.adapt_text import adapt_transcript_text
from dublaro.pipeline.align import build_speech_timeline
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
from dublaro.pipeline.subtitles import SrtTextMode, save_srt
from dublaro.pipeline.synthesize import synthesize_transcript_speech
from dublaro.pipeline.transcribe import save_transcript, transcribe_audio
from dublaro.pipeline.translate import translate_transcript
from dublaro.schemas import Transcript


@dataclass(frozen=True)
class DubbingArtifacts:
    workspace_dir: Path
    extracted_audio_path: Path
    source_transcript_path: Path
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
    manifest_path: Path | None


DubbingProgressStep = Literal[
    "extract_audio",
    "transcribe",
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


def dub_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    source_language: str | None,
    target_language: str,
    workspace_dir: str | Path,
    asr_adapter: AsrAdapter,
    translation_adapter: TranslationAdapter,
    text_adapter: TextAdapter,
    tts_adapter: TtsAdapter,
    asr_sample_rate: int = 16_000,
    speech_sample_rate: int = 24_000,
    fit_speech: bool = False,
    max_speech_speedup: float = 1.35,
    min_speech_overrun_seconds: float = 0.05,
    mix_original_audio: bool = False,
    original_audio_gain: float = 1.0,
    ducking_gain: float = 0.25,
    speech_gain: float = 1.0,
    ducking_margin_seconds: float = 0.05,
    ducking_fade_seconds: float = 0.05,
    translation_group_segments: bool = True,
    max_translation_group_pause_seconds: float = 0.8,
    max_translation_group_duration_seconds: float = 12.0,
    export_srt: bool = False,
    srt_output_path: str | Path | None = None,
    srt_text_mode: SrtTextMode = "adapted",
    write_manifest: bool = True,
    manifest_output_path: str | Path | None = None,
    progress_callback: DubbingProgressCallback | None = None,
    ffmpeg_executable: str = "ffmpeg",
    resume: bool = False,
    overwrite: bool = False,
) -> DubbingArtifacts:
    paths = DubPaths.build(
        video_path=video_path,
        output_path=output_path,
        workspace_dir=workspace_dir,
    )
    options = DubOptions(
        source_language=source_language,
        target_language=target_language,
        asr_sample_rate=asr_sample_rate,
        speech_sample_rate=speech_sample_rate,
        fit_speech=fit_speech,
        max_speech_speedup=max_speech_speedup,
        min_speech_overrun_seconds=min_speech_overrun_seconds,
        mix_original_audio=mix_original_audio,
        original_audio_gain=original_audio_gain,
        ducking_gain=ducking_gain,
        speech_gain=speech_gain,
        ducking_margin_seconds=ducking_margin_seconds,
        ducking_fade_seconds=ducking_fade_seconds,
        translation_group_segments=translation_group_segments,
        max_translation_group_pause_seconds=max_translation_group_pause_seconds,
        max_translation_group_duration_seconds=max_translation_group_duration_seconds,
        export_srt=export_srt,
        srt_output_path=Path(srt_output_path) if srt_output_path is not None else None,
        srt_text_mode=srt_text_mode,
        write_manifest=write_manifest,
        manifest_output_path=(
            Path(manifest_output_path) if manifest_output_path is not None else None
        ),
        ffmpeg_executable=ffmpeg_executable,
        resume=resume,
        overwrite=overwrite,
    )
    adapters = DubAdapters(
        asr=asr_adapter,
        translation=translation_adapter,
        text_adapter=text_adapter,
        tts=tts_adapter,
    )

    context = DubRunContext(
        paths=paths,
        options=options,
        adapters=adapters,
        artifact_paths=paths.artifacts(options),
        progress_callback=progress_callback,
    )

    return _run_dub_video(context)


def _run_dub_video(context: DubRunContext) -> DubbingArtifacts:
    started_at = datetime.now(UTC)

    paths = context.paths
    options = context.options
    adapters = context.adapters
    artifact_paths = context.artifact_paths
    progress_callback = context.progress_callback

    video_file = paths.video_path
    output_file = paths.output_path
    workspace = paths.workspace_dir
    workspace.mkdir(parents=True, exist_ok=True)

    target_language = options.target_language
    asr_sample_rate = options.asr_sample_rate
    speech_sample_rate = options.speech_sample_rate
    fit_speech = options.fit_speech
    max_speech_speedup = options.max_speech_speedup
    min_speech_overrun_seconds = options.min_speech_overrun_seconds
    mix_original_audio = options.mix_original_audio
    original_audio_gain = options.original_audio_gain
    ducking_gain = options.ducking_gain
    speech_gain = options.speech_gain
    ducking_margin_seconds = options.ducking_margin_seconds
    ducking_fade_seconds = options.ducking_fade_seconds
    translation_group_segments = options.translation_group_segments
    max_translation_group_pause_seconds = options.max_translation_group_pause_seconds
    max_translation_group_duration_seconds = (
        options.max_translation_group_duration_seconds
    )
    export_srt = options.export_srt
    srt_text_mode = options.srt_text_mode
    write_manifest = options.write_manifest
    ffmpeg_executable = options.ffmpeg_executable
    resume = options.resume
    overwrite = options.overwrite

    asr_adapter = adapters.asr
    translation_adapter = adapters.translation
    text_adapter = adapters.text_adapter
    tts_adapter = adapters.tts

    extracted_audio_path = _extract_audio(context)
    source_transcript_path = artifact_paths.source_transcript_path
    translated_transcript_path = artifact_paths.translated_transcript_path
    adapted_transcript_path = artifact_paths.adapted_transcript_path
    synthesized_transcript_path = artifact_paths.synthesized_transcript_path
    speech_dir = artifact_paths.speech_dir
    speech_track_path = artifact_paths.speech_track_path

    source_transcript = _transcribe_source_audio(context, extracted_audio_path)

    translated_transcript = _translate_source_transcript(context, source_transcript)

    adapted_transcript = _adapt_translated_text(context, translated_transcript)

    synthesized_transcript = _synthesize_speech(context, adapted_transcript)

    speech_timing = _fit_speech_to_timing(context, synthesized_transcript)
    speech_timeline_transcript = speech_timing.transcript
    fitted_transcript_path = speech_timing.fitted_transcript_path
    fitted_speech_dir = speech_timing.fitted_speech_dir

    speech_track_path = _align_speech_track(context, speech_timeline_transcript)

    export_audio = _prepare_audio_for_export(
        context,
        speech_timeline_transcript,
        speech_track_path,
    )
    audio_for_export_path = export_audio.audio_path
    mix_original_audio_path = export_audio.mix_original_audio_path
    mixed_audio_path = export_audio.mixed_audio_path

    with _progress_stage(
        progress_callback,
        "export_video",
        "Exporting dubbed video.",
    ):
        dubbed_video_path = export_dubbed_video(
            video_file,
            audio_for_export_path,
            output_file,
            overwrite=overwrite,
            executable=ffmpeg_executable,
        )

    srt_path: Path | None = None

    if export_srt:
        resolved_srt_path = artifact_paths.srt_path

        with _progress_stage(
            progress_callback,
            "export_srt",
            "Exporting SRT subtitles.",
        ):
            srt_path = save_srt(
                speech_timeline_transcript,
                resolved_srt_path,
                text_mode=srt_text_mode,
            )

    manifest_path: Path | None = None

    if write_manifest:
        resolved_manifest_path = artifact_paths.manifest_path

        with _progress_stage(
            progress_callback,
            "write_manifest",
            "Writing run manifest.",
        ):
            finished_at = datetime.now(UTC)

            manifest = build_dubbing_manifest(
                started_at=started_at,
                finished_at=finished_at,
                input_video_path=video_file,
                output_video_path=dubbed_video_path,
                source_language=source_transcript.source_language,
                target_language=target_language,
                asr_adapter=asr_adapter,
                translation_adapter=translation_adapter,
                text_adapter=text_adapter,
                tts_adapter=tts_adapter,
                options=DubbingOptionsManifest(
                    asr_sample_rate=asr_sample_rate,
                    speech_sample_rate=speech_sample_rate,
                    fit_speech=fit_speech,
                    max_speech_speedup=max_speech_speedup,
                    min_speech_overrun_seconds=min_speech_overrun_seconds,
                    mix_original_audio=mix_original_audio,
                    original_audio_gain=original_audio_gain,
                    ducking_gain=ducking_gain,
                    speech_gain=speech_gain,
                    ducking_margin_seconds=ducking_margin_seconds,
                    ducking_fade_seconds=ducking_fade_seconds,
                    translation_group_segments=translation_group_segments,
                    max_translation_group_pause_seconds=max_translation_group_pause_seconds,
                    max_translation_group_duration_seconds=max_translation_group_duration_seconds,
                    export_srt=export_srt,
                    srt_text_mode=srt_text_mode,
                    ffmpeg_executable=ffmpeg_executable,
                    resume=resume,
                    overwrite=overwrite,
                ),
                artifacts=DubbingArtifactsManifest(
                    workspace_dir=str(workspace),
                    extracted_audio_path=str(extracted_audio_path),
                    source_transcript_path=str(source_transcript_path),
                    translated_transcript_path=str(translated_transcript_path),
                    adapted_transcript_path=str(adapted_transcript_path),
                    synthesized_transcript_path=str(synthesized_transcript_path),
                    speech_dir=str(speech_dir),
                    speech_track_path=str(speech_track_path),
                    dubbed_video_path=str(dubbed_video_path),
                    fitted_transcript_path=(
                        str(fitted_transcript_path)
                        if fitted_transcript_path is not None
                        else None
                    ),
                    fitted_speech_dir=(
                        str(fitted_speech_dir)
                        if fitted_speech_dir is not None
                        else None
                    ),
                    mix_original_audio_path=(
                        str(mix_original_audio_path)
                        if mix_original_audio_path is not None
                        else None
                    ),
                    mixed_audio_path=(
                        str(mixed_audio_path) if mixed_audio_path is not None else None
                    ),
                    srt_path=str(srt_path) if srt_path is not None else None,
                    manifest_path=str(resolved_manifest_path),
                ),
                metadata={
                    "source_segment_count": str(len(source_transcript.segments)),
                    "translated_segment_count": str(
                        len(translated_transcript.segments)
                    ),
                    "adapted_segment_count": str(len(adapted_transcript.segments)),
                    "synthesized_segment_count": str(
                        len(synthesized_transcript.segments)
                    ),
                    "final_speech_segment_count": str(
                        len(speech_timeline_transcript.segments)
                    ),
                },
            )

            manifest_path = save_manifest(manifest, resolved_manifest_path)

    return DubbingArtifacts(
        workspace_dir=workspace,
        extracted_audio_path=extracted_audio_path,
        source_transcript_path=source_transcript_path,
        translated_transcript_path=translated_transcript_path,
        adapted_transcript_path=adapted_transcript_path,
        synthesized_transcript_path=synthesized_transcript_path,
        speech_dir=speech_dir,
        speech_track_path=speech_track_path,
        dubbed_video_path=dubbed_video_path,
        fitted_transcript_path=fitted_transcript_path,
        fitted_speech_dir=fitted_speech_dir,
        mix_original_audio_path=mix_original_audio_path,
        mixed_audio_path=mixed_audio_path,
        srt_path=srt_path,
        manifest_path=manifest_path,
    )


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
