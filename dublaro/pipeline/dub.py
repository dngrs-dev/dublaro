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
from dublaro.pipeline.export import export_dubbed_video
from dublaro.pipeline.fit_speech import fit_generated_speech_to_segments
from dublaro.pipeline.manifest import (
    DubbingArtifactsManifest,
    DubbingOptionsManifest,
    build_dubbing_manifest,
    save_manifest,
)
from dublaro.pipeline.mix import mix_original_audio_with_dubbed_speech
from dublaro.pipeline.subtitles import SrtTextMode, save_srt
from dublaro.pipeline.synthesize import synthesize_transcript_speech
from dublaro.pipeline.transcribe import save_transcript, transcribe_audio
from dublaro.pipeline.translate import translate_transcript


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

DubbingProgressStatus = Literal["started", "finished", "failed"]

DubbingProgressCallback = Callable[
    [DubbingProgressStep, DubbingProgressStatus, str],
    None,
]


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
    overwrite: bool = False,
) -> DubbingArtifacts:
    started_at = datetime.now(UTC)

    if manifest_output_path is not None and not write_manifest:
        raise ValueError(
            "manifest_output_path cannot be used when write_manifest is False."
        )
    video_file = Path(video_path)
    output_file = Path(output_path)
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    source_label = source_language or "auto"
    stem = video_file.stem

    extracted_audio_path = workspace / f"{stem}.audio.wav"
    source_transcript_path = workspace / f"{stem}.{source_label}.json"
    translated_transcript_path = workspace / f"{stem}.{target_language}.json"
    adapted_transcript_path = workspace / f"{stem}.{target_language}.adapted.json"
    synthesized_transcript_path = (
        workspace / f"{stem}.{target_language}.synthesized.json"
    )
    speech_dir = workspace / f"{stem}.{target_language}.speech"
    speech_track_path = workspace / f"{stem}.{target_language}.speech-track.wav"

    with _progress_stage(
        progress_callback,
        "extract_audio",
        "Extracting audio from video.",
    ):
        extracted_audio_path = extract_audio_from_video(
            video_file,
            extracted_audio_path,
            sample_rate=asr_sample_rate,
            channels=1,
            overwrite=overwrite,
            executable=ffmpeg_executable,
        )

    with _progress_stage(
        progress_callback,
        "transcribe",
        "Transcribing source audio.",
    ):
        source_transcript = transcribe_audio(
            extracted_audio_path,
            adapter=asr_adapter,
            options=TranscriptionOptions(source_language=source_language),
        )
        save_transcript(source_transcript, source_transcript_path)

    with _progress_stage(
        progress_callback,
        "translate",
        f"Translating transcript to {target_language}.",
    ):
        translated_transcript = translate_transcript(
            source_transcript,
            adapter=translation_adapter,
            target_language=target_language,
            source_language=source_language,
            group_segments=translation_group_segments,
            max_group_pause_seconds=max_translation_group_pause_seconds,
            max_group_duration_seconds=max_translation_group_duration_seconds,
        )
        save_transcript(translated_transcript, translated_transcript_path)

    with _progress_stage(
        progress_callback,
        "adapt_text",
        "Adapting translated text for timing.",
    ):
        adapted_transcript = adapt_transcript_text(
            translated_transcript,
            adapter=text_adapter,
            target_language=target_language,
            source_language=source_language,
        )
        save_transcript(adapted_transcript, adapted_transcript_path)

    with _progress_stage(
        progress_callback,
        "synthesize",
        "Synthesizing speech clips.",
    ):
        synthesized_transcript = synthesize_transcript_speech(
            adapted_transcript,
            adapter=tts_adapter,
            output_dir=speech_dir,
            language=target_language,
            sample_rate=speech_sample_rate,
        )
        save_transcript(synthesized_transcript, synthesized_transcript_path)

    fitted_transcript_path: Path | None = None
    fitted_speech_dir: Path | None = None
    speech_timeline_transcript = synthesized_transcript

    if fit_speech:
        fitted_transcript_path = workspace / f"{stem}.{target_language}.fitted.json"
        fitted_speech_dir = workspace / f"{stem}.{target_language}.fitted-speech"

        with _progress_stage(
            progress_callback,
            "fit_speech",
            "Fitting overlong speech clips to segment timing.",
        ):
            speech_timeline_transcript = fit_generated_speech_to_segments(
                synthesized_transcript,
                output_dir=fitted_speech_dir,
                max_speedup=max_speech_speedup,
                min_overrun_seconds=min_speech_overrun_seconds,
                overwrite=overwrite,
                executable=ffmpeg_executable,
            )
            save_transcript(speech_timeline_transcript, fitted_transcript_path)

    with _progress_stage(
        progress_callback,
        "align_speech",
        "Building timed speech track.",
    ):
        speech_track_path = build_speech_timeline(
            speech_timeline_transcript,
            output_path=speech_track_path,
            sample_rate=speech_sample_rate,
        )

    audio_for_export_path = speech_track_path
    mix_original_audio_path: Path | None = None
    mixed_audio_path: Path | None = None

    if mix_original_audio:
        mix_original_audio_path = workspace / f"{stem}.original-mix.wav"
        mixed_audio_path = workspace / f"{stem}.{target_language}.mixed.wav"

        with _progress_stage(
            progress_callback,
            "mix_original_audio",
            "Mixing dubbed speech over original audio.",
        ):
            mix_original_audio_path = extract_audio_from_video(
                video_file,
                mix_original_audio_path,
                sample_rate=speech_sample_rate,
                channels=1,
                overwrite=overwrite,
                executable=ffmpeg_executable,
            )

            mixed_audio_path = mix_original_audio_with_dubbed_speech(
                speech_timeline_transcript,
                original_audio_path=mix_original_audio_path,
                speech_track_path=speech_track_path,
                output_path=mixed_audio_path,
                original_gain=original_audio_gain,
                ducking_gain=ducking_gain,
                speech_gain=speech_gain,
                ducking_margin_seconds=ducking_margin_seconds,
                ducking_fade_seconds=ducking_fade_seconds,
            )

            audio_for_export_path = mixed_audio_path

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
        resolved_srt_path = (
            Path(srt_output_path)
            if srt_output_path is not None
            else output_file.with_suffix(".srt")
        )

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
        resolved_manifest_path = (
            Path(manifest_output_path)
            if manifest_output_path is not None
            else workspace / f"{stem}.{target_language}.manifest.json"
        )

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
