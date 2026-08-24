from dataclasses import dataclass
from pathlib import Path

from dublaro.audio.ffmpeg import (
    change_audio_tempo,
    extract_audio_from_video,
    normalize_audio_loudness,
)
from dublaro.pipeline.dub.context import DubRunContext
from dublaro.pipeline.dub.progress import (
    progress_skipped as _progress_skipped,
)
from dublaro.pipeline.dub.progress import (
    progress_stage as _progress_stage,
)
from dublaro.pipeline.dub.results import AudioNormalizationResult, ExportAudioResult
from dublaro.pipeline.mix import mix_original_audio_with_dubbed_speech
from dublaro.pipeline.resume import reusable_file
from dublaro.pipeline.separate import separate_background_audio
from dublaro.schemas import Transcript


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


@dataclass(frozen=True)
class BackgroundAudioResult:
    audio_path: Path
    ducking_gain: float
    mix_original_audio_path: Path
    separated_background_audio_path: Path | None = None
    separated_voice_audio_path: Path | None = None
    video_fitted_original_audio_path: Path | None = None
    video_fitted_background_audio_path: Path | None = None


def _prepare_background_audio_for_mix(
    context: DubRunContext,
    *,
    video_slowdown_factor: float,
) -> BackgroundAudioResult | None:
    if context.options.background_mode == "speech-only":
        return None

    original_mix_path = context.artifact_paths.mix_original_audio_path

    if context.options.resume and reusable_file(original_mix_path):
        _progress_skipped(
            context.progress_callback,
            "mix_original_audio",
            f"Using existing original mix audio: {original_mix_path}.",
        )
    else:
        extract_audio_from_video(
            context.paths.video_path,
            original_mix_path,
            sample_rate=context.options.speech_sample_rate,
            channels=1,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )

    background_audio_path = original_mix_path
    separated_background_audio_path = None
    separated_voice_audio_path = None

    if context.options.background_mode == "separated":
        if context.adapters.source_separation is None:
            raise ValueError(
                "source_separation_adapter is required when background_mode is separated."
            )

        separated_background_audio_path = (
            context.artifact_paths.separated_background_audio_path
        )
        separated_voice_audio_path = context.artifact_paths.separated_voice_audio_path

        if context.options.resume and reusable_file(separated_background_audio_path):
            _progress_skipped(
                context.progress_callback,
                "separate_background",
                f"Using existing separated background: {separated_background_audio_path}.",
            )
            background_audio_path = separated_background_audio_path
        else:
            with _progress_stage(
                context.progress_callback,
                "separate_background",
                "Separating original voice from background audio.",
            ):
                separated = separate_background_audio(
                    original_mix_path,
                    adapter=context.adapters.source_separation,
                    background_output_path=separated_background_audio_path,
                    voice_output_path=separated_voice_audio_path,
                    sample_rate=context.options.speech_sample_rate,
                    overwrite=context.options.overwrite,
                )
                background_audio_path = separated.background_audio_path
                separated_background_audio_path = separated.background_audio_path
                separated_voice_audio_path = separated.voice_audio_path

    ducking_gain = (
        context.options.ducking_gain
        if context.options.background_mode == "ducked"
        else context.options.original_audio_gain
    )

    video_fitted_original_audio_path = None
    video_fitted_background_audio_path = None

    if video_slowdown_factor > 1.0:
        video_fitted_path = (
            context.artifact_paths.video_fitted_background_audio_path
            if context.options.background_mode == "separated"
            else context.artifact_paths.video_fitted_original_audio_path
        )

        if context.options.resume and reusable_file(video_fitted_path):
            _progress_skipped(
                context.progress_callback,
                "mix_original_audio",
                f"Using existing video-fitted background audio: {video_fitted_path}.",
            )
            background_audio_path = video_fitted_path
        else:
            background_audio_path = change_audio_tempo(
                background_audio_path,
                video_fitted_path,
                tempo_factor=1.0 / video_slowdown_factor,
                sample_rate=context.options.speech_sample_rate,
                overwrite=context.options.overwrite,
                executable=context.options.ffmpeg_executable,
            )

        if context.options.background_mode == "separated":
            video_fitted_background_audio_path = video_fitted_path
        else:
            video_fitted_original_audio_path = video_fitted_path

    return BackgroundAudioResult(
        audio_path=background_audio_path,
        ducking_gain=ducking_gain,
        mix_original_audio_path=original_mix_path,
        separated_background_audio_path=separated_background_audio_path,
        separated_voice_audio_path=separated_voice_audio_path,
        video_fitted_original_audio_path=video_fitted_original_audio_path,
        video_fitted_background_audio_path=video_fitted_background_audio_path,
    )


def _prepare_audio_for_export(
    context: DubRunContext,
    speech_timeline_transcript: Transcript,
    speech_track_path: Path,
    *,
    video_slowdown_factor: float = 1.0,
) -> ExportAudioResult:
    if context.options.background_mode == "speech-only":
        return ExportAudioResult(audio_path=speech_track_path)

    mixed_path = context.artifact_paths.mixed_audio_path

    if context.options.resume and reusable_file(mixed_path):
        _progress_skipped(
            context.progress_callback,
            "mix_original_audio",
            f"Using existing mixed audio: {mixed_path}.",
        )

        def existing(path: Path) -> Path | None:
            return path if reusable_file(path) else None

        return ExportAudioResult(
            audio_path=mixed_path,
            mix_original_audio_path=existing(
                context.artifact_paths.mix_original_audio_path
            ),
            mixed_audio_path=mixed_path,
            video_fitted_original_audio_path=existing(
                context.artifact_paths.video_fitted_original_audio_path
            ),
            separated_background_audio_path=(
                existing(context.artifact_paths.separated_background_audio_path)
                if context.options.background_mode == "separated"
                else None
            ),
            separated_voice_audio_path=(
                existing(context.artifact_paths.separated_voice_audio_path)
                if context.options.background_mode == "separated"
                else None
            ),
            video_fitted_background_audio_path=(
                existing(context.artifact_paths.video_fitted_background_audio_path)
                if context.options.background_mode == "separated"
                else None
            ),
        )

    background_audio = _prepare_background_audio_for_mix(
        context,
        video_slowdown_factor=video_slowdown_factor,
    )
    if background_audio is None:
        return ExportAudioResult(audio_path=speech_track_path)

    with _progress_stage(
        context.progress_callback,
        "mix_original_audio",
        "Mixing dubbed speech over background audio.",
    ):
        mixed_audio_path = mix_original_audio_with_dubbed_speech(
            speech_timeline_transcript,
            original_audio_path=background_audio.audio_path,
            speech_track_path=speech_track_path,
            output_path=mixed_path,
            original_gain=context.options.original_audio_gain,
            ducking_gain=background_audio.ducking_gain,
            speech_gain=context.options.speech_gain,
            ducking_margin_seconds=context.options.ducking_margin_seconds,
            ducking_fade_seconds=context.options.ducking_fade_seconds,
        )

        return ExportAudioResult(
            audio_path=mixed_audio_path,
            mix_original_audio_path=background_audio.mix_original_audio_path,
            mixed_audio_path=mixed_audio_path,
            video_fitted_original_audio_path=(
                background_audio.video_fitted_original_audio_path
            ),
            separated_background_audio_path=(
                background_audio.separated_background_audio_path
            ),
            separated_voice_audio_path=background_audio.separated_voice_audio_path,
            video_fitted_background_audio_path=(
                background_audio.video_fitted_background_audio_path
            ),
        )


def _normalize_audio_for_export(
    context: DubRunContext,
    audio_path: Path,
) -> AudioNormalizationResult:
    if not context.options.normalize_final_audio:
        return AudioNormalizationResult(audio_path=audio_path)

    normalized_path = context.artifact_paths.normalized_audio_path

    if context.options.resume and reusable_file(normalized_path):
        _progress_skipped(
            context.progress_callback,
            "normalize_audio",
            f"Using existing normalized audio: {normalized_path}.",
        )
        return AudioNormalizationResult(
            audio_path=normalized_path,
            normalized_audio_path=normalized_path,
        )

    with _progress_stage(
        context.progress_callback,
        "normalize_audio",
        "Normalizing final audio loudness.",
    ):
        saved_path = normalize_audio_loudness(
            audio_path,
            normalized_path,
            target_lufs=context.options.target_final_lufs,
            true_peak=context.options.final_true_peak,
            loudness_range=context.options.final_loudness_range,
            sample_rate=context.options.speech_sample_rate,
            channels=1,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )

    return AudioNormalizationResult(
        audio_path=saved_path,
        normalized_audio_path=saved_path,
    )
