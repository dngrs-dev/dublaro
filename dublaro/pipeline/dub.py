from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dublaro.adapters.asr import AsrAdapter
from dublaro.adapters.diarization import DiarizationAdapter
from dublaro.adapters.dubbing_script import DubbingScriptAdapter
from dublaro.adapters.source_separation import SourceSeparationAdapter
from dublaro.adapters.text_adapter import TextAdapter
from dublaro.adapters.translation import TranslationAdapter
from dublaro.adapters.tts import TtsAdapter
from dublaro.pipeline.dub_plan import (
    BackgroundMode,
    DubAdapters,
    DubOptions,
    DubPaths,
    TextWorkflowMode,
)
from dublaro.pipeline.dub_stages import (
    DubbingProgressCallback,
    DubbingProgressStatus,
    DubbingProgressStep,
    DubRunContext,
    ManifestInputs,
    _align_speech_track,
    _diarize_source_transcript,
    _export_video,
    _extract_audio,
    _fit_speech_to_timing,
    _fit_video_to_speech,
    _prepare_audio_for_export,
    _prepare_subtitles_for_export,
    _prepare_text_for_dubbing,
    _repair_speech_timing,
    _synthesize_speech,
    _transcribe_source_audio,
    _write_manifest,
)
from dublaro.pipeline.subtitles import SrtTextMode, SubtitleEmbedMode
from dublaro.pipeline.voices import SpeakerVoice

__all__ = [
    "DubbingArtifacts",
    "DubbingProgressCallback",
    "DubbingProgressStatus",
    "DubbingProgressStep",
    "dub_video",
]


@dataclass(frozen=True)
class DubbingArtifacts:
    workspace_dir: Path
    extracted_audio_path: Path
    source_transcript_path: Path
    diarized_transcript_path: Path | None
    translated_transcript_path: Path
    adapted_transcript_path: Path
    synthesized_transcript_path: Path
    speech_dir: Path
    timing_repaired_transcript_path: Path | None
    timing_repaired_speech_dir: Path | None
    speech_track_path: Path
    dubbed_video_path: Path
    fitted_transcript_path: Path | None
    fitted_speech_dir: Path | None
    video_fitted_transcript_path: Path | None
    fitted_video_path: Path | None
    video_fitted_original_audio_path: Path | None
    separated_background_audio_path: Path | None
    separated_voice_audio_path: Path | None
    video_fitted_background_audio_path: Path | None
    mix_original_audio_path: Path | None
    mixed_audio_path: Path | None
    srt_path: Path | None
    embedded_srt_path: Path | None
    manifest_path: Path | None


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
    dubbing_script_adapter: DubbingScriptAdapter | None = None,
    speaker_voices: Mapping[str, SpeakerVoice] | None = None,
    diarization_adapter: DiarizationAdapter | None = None,
    source_separation_adapter: SourceSeparationAdapter | None = None,
    background_mode: BackgroundMode = "speech-only",
    diarize: bool = False,
    diarization_min_speakers: int | None = None,
    diarization_max_speakers: int | None = None,
    asr_sample_rate: int = 16_000,
    speech_sample_rate: int = 24_000,
    repair_timing: bool = False,
    max_timing_repair_attempts: int = 2,
    timing_repair_target_speedup: float = 1.15,
    fit_speech: bool = False,
    max_speech_speedup: float = 1.35,
    min_speech_overrun_seconds: float = 0.05,
    fit_video: bool = False,
    max_video_slowdown: float = 1.5,
    mix_original_audio: bool = False,
    original_audio_gain: float = 1.0,
    ducking_gain: float = 0.25,
    speech_gain: float = 1.0,
    ducking_margin_seconds: float = 0.05,
    ducking_fade_seconds: float = 0.05,
    text_workflow: TextWorkflowMode = "translate-then-adapt",
    translation_group_segments: bool = True,
    max_translation_group_pause_seconds: float = 0.8,
    max_translation_group_duration_seconds: float = 12.0,
    max_translation_sentence_group_duration_seconds: float = 24.0,
    export_srt: bool = False,
    srt_output_path: str | Path | None = None,
    srt_text_mode: SrtTextMode = "adapted",
    subtitle_embed: SubtitleEmbedMode = "none",
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
    resolved_background_mode: BackgroundMode = (
        "ducked"
        if mix_original_audio and background_mode == "speech-only"
        else background_mode
    )
    resolved_mix_original_audio = (
        mix_original_audio or resolved_background_mode != "speech-only"
    )
    options = DubOptions(
        source_language=source_language,
        target_language=target_language,
        asr_sample_rate=asr_sample_rate,
        diarize=diarize,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        speech_sample_rate=speech_sample_rate,
        repair_timing=repair_timing,
        max_timing_repair_attempts=max_timing_repair_attempts,
        timing_repair_target_speedup=timing_repair_target_speedup,
        fit_speech=fit_speech,
        max_speech_speedup=max_speech_speedup,
        min_speech_overrun_seconds=min_speech_overrun_seconds,
        fit_video=fit_video,
        max_video_slowdown=max_video_slowdown,
        background_mode=resolved_background_mode,
        mix_original_audio=resolved_mix_original_audio,
        original_audio_gain=original_audio_gain,
        ducking_gain=ducking_gain,
        speech_gain=speech_gain,
        ducking_margin_seconds=ducking_margin_seconds,
        ducking_fade_seconds=ducking_fade_seconds,
        text_workflow=text_workflow,
        translation_group_segments=translation_group_segments,
        max_translation_group_pause_seconds=max_translation_group_pause_seconds,
        max_translation_group_duration_seconds=max_translation_group_duration_seconds,
        max_translation_sentence_group_duration_seconds=(
            max_translation_sentence_group_duration_seconds
        ),
        export_srt=export_srt,
        srt_output_path=Path(srt_output_path) if srt_output_path is not None else None,
        srt_text_mode=srt_text_mode,
        subtitle_embed=subtitle_embed,
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
        diarization=diarization_adapter,
        translation=translation_adapter,
        text_adapter=text_adapter,
        tts=tts_adapter,
        dubbing_script=dubbing_script_adapter,
        speaker_voices=speaker_voices,
        source_separation=source_separation_adapter,
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

    artifact_paths = context.artifact_paths
    workspace = context.paths.workspace_dir
    workspace.mkdir(parents=True, exist_ok=True)

    extracted_audio_path = _extract_audio(context)
    source_transcript_path = artifact_paths.source_transcript_path
    diarized_transcript_path = (
        artifact_paths.diarized_transcript_path if context.options.diarize else None
    )
    translated_transcript_path = artifact_paths.translated_transcript_path
    adapted_transcript_path = artifact_paths.adapted_transcript_path
    synthesized_transcript_path = artifact_paths.synthesized_transcript_path
    speech_dir = artifact_paths.speech_dir
    timing_repaired_transcript_path = None
    timing_repaired_speech_dir = None
    speech_track_path = artifact_paths.speech_track_path

    source_transcript = _transcribe_source_audio(context, extracted_audio_path)
    source_transcript = _diarize_source_transcript(
        context,
        extracted_audio_path,
        source_transcript,
    )

    text_workflow = _prepare_text_for_dubbing(context, source_transcript)
    translated_transcript = text_workflow.translated_transcript
    adapted_transcript = text_workflow.adapted_transcript

    synthesized_transcript = _synthesize_speech(context, adapted_transcript)

    timing_repair = _repair_speech_timing(context, synthesized_transcript)
    timing_repaired_transcript_path = timing_repair.timing_repaired_transcript_path
    timing_repaired_speech_dir = timing_repair.timing_repaired_speech_dir

    speech_timing = _fit_speech_to_timing(context, timing_repair.transcript)
    speech_timeline_transcript = speech_timing.transcript
    fitted_transcript_path = speech_timing.fitted_transcript_path
    fitted_speech_dir = speech_timing.fitted_speech_dir

    video_fit = _fit_video_to_speech(context, speech_timing.transcript)
    speech_timeline_transcript = video_fit.transcript
    video_for_export_path = video_fit.video_path
    video_fitted_transcript_path = video_fit.video_fitted_transcript_path
    fitted_video_path = video_fit.fitted_video_path

    speech_track_path = _align_speech_track(context, speech_timeline_transcript)

    export_audio = _prepare_audio_for_export(
        context,
        speech_timeline_transcript,
        speech_track_path,
        video_slowdown_factor=video_fit.slowdown_factor,
    )
    audio_for_export_path = export_audio.audio_path
    mix_original_audio_path = export_audio.mix_original_audio_path
    mixed_audio_path = export_audio.mixed_audio_path
    video_fitted_original_audio_path = export_audio.video_fitted_original_audio_path

    separated_background_audio_path = export_audio.separated_background_audio_path
    separated_voice_audio_path = export_audio.separated_voice_audio_path
    video_fitted_background_audio_path = export_audio.video_fitted_background_audio_path

    subtitle_export = _prepare_subtitles_for_export(
        context,
        speech_timeline_transcript,
    )

    dubbed_video_path = _export_video(
        context,
        video_for_export_path,
        audio_for_export_path,
        subtitle_export.embedded_srt_path,
    )

    srt_path = subtitle_export.sidecar_srt_path
    embedded_srt_path = subtitle_export.embedded_srt_path

    manifest_inputs = ManifestInputs(
        started_at=started_at,
        extracted_audio_path=extracted_audio_path,
        source_transcript_path=source_transcript_path,
        diarized_transcript_path=diarized_transcript_path,
        translated_transcript_path=translated_transcript_path,
        adapted_transcript_path=adapted_transcript_path,
        synthesized_transcript_path=synthesized_transcript_path,
        timing_repaired_transcript_path=timing_repaired_transcript_path,
        timing_repaired_speech_dir=timing_repaired_speech_dir,
        speech_dir=speech_dir,
        speech_track_path=speech_track_path,
        dubbed_video_path=dubbed_video_path,
        fitted_transcript_path=fitted_transcript_path,
        fitted_speech_dir=fitted_speech_dir,
        video_fitted_transcript_path=video_fitted_transcript_path,
        fitted_video_path=fitted_video_path,
        video_fitted_original_audio_path=video_fitted_original_audio_path,
        separated_background_audio_path=separated_background_audio_path,
        separated_voice_audio_path=separated_voice_audio_path,
        video_fitted_background_audio_path=video_fitted_background_audio_path,
        mix_original_audio_path=mix_original_audio_path,
        mixed_audio_path=mixed_audio_path,
        srt_path=srt_path,
        embedded_srt_path=embedded_srt_path,
        source_transcript=source_transcript,
        translated_transcript=translated_transcript,
        adapted_transcript=adapted_transcript,
        synthesized_transcript=synthesized_transcript,
        speech_timeline_transcript=speech_timeline_transcript,
    )
    manifest_path = _write_manifest(context, manifest_inputs)

    return DubbingArtifacts(
        workspace_dir=workspace,
        extracted_audio_path=extracted_audio_path,
        source_transcript_path=source_transcript_path,
        diarized_transcript_path=diarized_transcript_path,
        translated_transcript_path=translated_transcript_path,
        adapted_transcript_path=adapted_transcript_path,
        synthesized_transcript_path=synthesized_transcript_path,
        speech_dir=speech_dir,
        timing_repaired_transcript_path=timing_repaired_transcript_path,
        timing_repaired_speech_dir=timing_repaired_speech_dir,
        speech_track_path=speech_track_path,
        dubbed_video_path=dubbed_video_path,
        fitted_transcript_path=fitted_transcript_path,
        fitted_speech_dir=fitted_speech_dir,
        video_fitted_transcript_path=video_fitted_transcript_path,
        fitted_video_path=fitted_video_path,
        video_fitted_original_audio_path=video_fitted_original_audio_path,
        separated_background_audio_path=separated_background_audio_path,
        separated_voice_audio_path=separated_voice_audio_path,
        video_fitted_background_audio_path=video_fitted_background_audio_path,
        mix_original_audio_path=mix_original_audio_path,
        mixed_audio_path=mixed_audio_path,
        srt_path=srt_path,
        embedded_srt_path=embedded_srt_path,
        manifest_path=manifest_path,
    )
