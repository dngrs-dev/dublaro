from dataclasses import dataclass
from pathlib import Path

from dublaro.cli.services.dub_runner import validate_resolved_dub_settings
from dublaro.cli_config import (
    DubCliOverrides,
    ResolvedDubSettings,
    resolve_dub_settings,
)
from dublaro.config import load_config
from dublaro.pipeline.dub_plan import BackgroundMode, TextWorkflowMode
from dublaro.pipeline.subtitles import SrtTextMode, SubtitleEmbedMode


@dataclass(frozen=True)
class DubCommandOverrides:
    source_language: str | None = None
    target_language: str | None = None
    text_workflow: str | None = None
    background_mode: str | None = None
    output_path: Path | None = None
    output_dir: Path | None = None
    workspace_dir: Path | None = None
    resume_enabled: bool | None = None
    overwrite: bool | None = None
    preflight_enabled: bool | None = None
    ffmpeg_executable: str | None = None
    asr_sample_rate: int | None = None
    speech_sample_rate: int | None = None
    asr_backend: str | None = None
    model_size: str | None = None
    device: str | None = None
    compute_type: str | None = None
    diarize_enabled: bool | None = None
    diarization_backend: str | None = None
    diarization_model_id: str | None = None
    diarization_device: str | None = None
    diarization_token_env_var: str | None = None
    diarization_min_speakers: int | None = None
    diarization_max_speakers: int | None = None
    translation_backend: str | None = None
    install_package: bool | None = None
    translation_ollama_model: str | None = None
    translation_ollama_url: str | None = None
    translation_ollama_timeout_seconds: float | None = None
    translation_ollama_temperature: float | None = None
    translation_group_segments: bool | None = None
    max_translation_group_pause_seconds: float | None = None
    max_translation_group_duration_seconds: float | None = None
    max_translation_sentence_group_duration_seconds: float | None = None
    text_adapter_backend: str | None = None
    ollama_model: str | None = None
    ollama_url: str | None = None
    ollama_timeout_seconds: float | None = None
    ollama_temperature: float | None = None
    tts_backend: str | None = None
    piper_model_path: Path | None = None
    piper_config_path: Path | None = None
    piper_executable: str | None = None
    piper_speaker: int | None = None
    repair_timing_enabled: bool | None = None
    max_timing_repair_attempts: int | None = None
    timing_repair_target_speedup: float | None = None
    fit_speech_enabled: bool | None = None
    max_speech_speedup: float | None = None
    min_speech_overrun_seconds: float | None = None
    fit_video_enabled: bool | None = None
    max_video_slowdown: float | None = None
    mix_original_audio_enabled: bool | None = None
    source_separation_backend: str | None = None
    demucs_executable: str | None = None
    demucs_model: str | None = None
    demucs_device: str | None = None
    original_audio_gain: float | None = None
    ducking_gain: float | None = None
    speech_gain: float | None = None
    ducking_margin_seconds: float | None = None
    ducking_fade_seconds: float | None = None
    normalize_final_audio: bool | None = None
    target_final_lufs: float | None = None
    final_true_peak: float | None = None
    final_loudness_range: float | None = None
    export_srt_enabled: bool | None = None
    srt_output_path: Path | None = None
    srt_text_mode: str | None = None
    subtitle_embed: str | None = None
    write_manifest_enabled: bool | None = None
    manifest_output_path: Path | None = None

    def to_cli_overrides(self) -> DubCliOverrides:
        return DubCliOverrides(
            source_language=self.source_language,
            target_language=self.target_language,
            text_workflow=self.text_workflow,
            background_mode=self.background_mode,
            output_path=self.output_path,
            output_dir=self.output_dir,
            workspace_dir=self.workspace_dir,
            resume=self.resume_enabled,
            overwrite=self.overwrite,
            preflight=self.preflight_enabled,
            ffmpeg_executable=self.ffmpeg_executable,
            asr_sample_rate=self.asr_sample_rate,
            speech_sample_rate=self.speech_sample_rate,
            asr_backend=self.asr_backend,
            model_size=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            diarize=self.diarize_enabled,
            diarization_backend=self.diarization_backend,
            diarization_model_id=self.diarization_model_id,
            diarization_device=self.diarization_device,
            diarization_token_env_var=self.diarization_token_env_var,
            diarization_min_speakers=self.diarization_min_speakers,
            diarization_max_speakers=self.diarization_max_speakers,
            translation_backend=self.translation_backend,
            install_package=self.install_package,
            translation_ollama_model=self.translation_ollama_model,
            translation_ollama_url=self.translation_ollama_url,
            translation_ollama_timeout_seconds=self.translation_ollama_timeout_seconds,
            translation_ollama_temperature=self.translation_ollama_temperature,
            translation_group_segments=self.translation_group_segments,
            max_translation_group_pause_seconds=(
                self.max_translation_group_pause_seconds
            ),
            max_translation_group_duration_seconds=(
                self.max_translation_group_duration_seconds
            ),
            max_translation_sentence_group_duration_seconds=(
                self.max_translation_sentence_group_duration_seconds
            ),
            text_adapter_backend=self.text_adapter_backend,
            ollama_model=self.ollama_model,
            ollama_url=self.ollama_url,
            ollama_timeout_seconds=self.ollama_timeout_seconds,
            ollama_temperature=self.ollama_temperature,
            tts_backend=self.tts_backend,
            piper_model_path=self.piper_model_path,
            piper_config_path=self.piper_config_path,
            piper_executable=self.piper_executable,
            piper_speaker=self.piper_speaker,
            repair_timing=self.repair_timing_enabled,
            max_timing_repair_attempts=self.max_timing_repair_attempts,
            timing_repair_target_speedup=self.timing_repair_target_speedup,
            fit_speech=self.fit_speech_enabled,
            max_speech_speedup=self.max_speech_speedup,
            min_speech_overrun_seconds=self.min_speech_overrun_seconds,
            fit_video=self.fit_video_enabled,
            max_video_slowdown=self.max_video_slowdown,
            mix_original_audio=self.mix_original_audio_enabled,
            source_separation_backend=self.source_separation_backend,
            demucs_executable=self.demucs_executable,
            demucs_model=self.demucs_model,
            demucs_device=self.demucs_device,
            original_audio_gain=self.original_audio_gain,
            ducking_gain=self.ducking_gain,
            speech_gain=self.speech_gain,
            ducking_margin_seconds=self.ducking_margin_seconds,
            ducking_fade_seconds=self.ducking_fade_seconds,
            normalize_final_audio=self.normalize_final_audio,
            target_final_lufs=self.target_final_lufs,
            final_true_peak=self.final_true_peak,
            final_loudness_range=self.final_loudness_range,
            export_srt=self.export_srt_enabled,
            srt_output_path=self.srt_output_path,
            srt_text_mode=self.srt_text_mode,
            subtitle_embed=self.subtitle_embed,
            write_manifest=self.write_manifest_enabled,
            manifest_output_path=self.manifest_output_path,
        )


@dataclass(frozen=True)
class ResolvedDubCommandSettings:
    settings: ResolvedDubSettings
    text_workflow: TextWorkflowMode
    background_mode: BackgroundMode
    srt_text_mode: SrtTextMode
    subtitle_embed: SubtitleEmbedMode


def resolve_dub_command_settings(
    *,
    video_path: Path,
    config_path: Path | None,
    overrides: DubCommandOverrides,
) -> ResolvedDubCommandSettings:
    loaded_config = load_config(config_path)
    settings = resolve_dub_settings(
        video_path=video_path,
        loaded_config=loaded_config,
        overrides=overrides.to_cli_overrides(),
    )
    (
        parsed_text_workflow,
        parsed_background_mode,
        parsed_srt_text_mode,
        parsed_subtitle_embed,
    ) = validate_resolved_dub_settings(settings)

    return ResolvedDubCommandSettings(
        settings=settings,
        text_workflow=parsed_text_workflow,
        background_mode=parsed_background_mode,
        srt_text_mode=parsed_srt_text_mode,
        subtitle_embed=parsed_subtitle_embed,
    )
