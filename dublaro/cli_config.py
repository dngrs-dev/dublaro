from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dublaro.config import LoadedConfig, resolve_config_path
from dublaro.pipeline.export import default_dubbed_video_path

T = TypeVar("T")


@dataclass(frozen=True)
class DubCliOverrides:
    source_language: str | None = None
    target_language: str | None = None
    output_path: Path | None = None
    workspace_dir: Path | None = None
    resume: bool | None = None
    overwrite: bool | None = None
    preflight: bool | None = None
    ffmpeg_executable: str | None = None
    asr_sample_rate: int | None = None
    speech_sample_rate: int | None = None
    asr_backend: str | None = None
    model_size: str | None = None
    device: str | None = None
    compute_type: str | None = None
    translation_backend: str | None = None
    install_package: bool | None = None
    translation_group_segments: bool | None = None
    max_translation_group_pause_seconds: float | None = None
    max_translation_group_duration_seconds: float | None = None
    text_adapter_backend: str | None = None
    tts_backend: str | None = None
    piper_model_path: Path | None = None
    piper_config_path: Path | None = None
    piper_executable: str | None = None
    piper_speaker: int | None = None
    fit_speech: bool | None = None
    max_speech_speedup: float | None = None
    min_speech_overrun_seconds: float | None = None
    mix_original_audio: bool | None = None
    original_audio_gain: float | None = None
    ducking_gain: float | None = None
    speech_gain: float | None = None
    ducking_margin_seconds: float | None = None
    ducking_fade_seconds: float | None = None
    export_srt: bool | None = None
    srt_output_path: Path | None = None
    srt_text_mode: str | None = None
    write_manifest: bool | None = None
    manifest_output_path: Path | None = None


@dataclass(frozen=True)
class ResolvedDubSettings:
    source_language: str | None
    target_language: str
    output_path: Path
    workspace_dir: Path
    resume: bool
    overwrite: bool
    preflight: bool
    ffmpeg_executable: str
    asr_sample_rate: int
    speech_sample_rate: int
    asr_backend: str
    model_size: str
    device: str
    compute_type: str
    translation_backend: str
    install_package: bool
    translation_group_segments: bool
    max_translation_group_pause_seconds: float
    max_translation_group_duration_seconds: float
    text_adapter_backend: str
    tts_backend: str
    piper_model_path: Path | None
    piper_config_path: Path | None
    piper_executable: str
    piper_speaker: int | None
    fit_speech: bool
    max_speech_speedup: float
    min_speech_overrun_seconds: float
    mix_original_audio: bool
    original_audio_gain: float
    ducking_gain: float
    speech_gain: float
    ducking_margin_seconds: float
    ducking_fade_seconds: float
    export_srt: bool
    srt_output_path: Path | None
    srt_text_mode: str
    write_manifest: bool
    manifest_output_path: Path | None


def _select(cli_value: T | None, config_value: T | None, default: T) -> T:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _select_optional(cli_value: T | None, config_value: T | None) -> T | None:
    return cli_value if cli_value is not None else config_value


def _select_path(
    cli_value: Path | None,
    config_value: Path | None,
    base_dir: Path | None,
    default: Path | None = None,
) -> Path | None:
    if cli_value is not None:
        return cli_value

    resolved_config_path = resolve_config_path(config_value, base_dir)
    return resolved_config_path or default


def _required_text(
    option_name: str,
    config_name: str,
    cli_value: str | None,
    config_value: str | None,
) -> str:
    value = _select_optional(cli_value, config_value)
    if value is None:
        raise ValueError(f"{option_name} is required when {config_name} is not set.")
    return value


def resolve_dub_settings(
    *,
    video_path: Path,
    loaded_config: LoadedConfig,
    overrides: DubCliOverrides,
) -> ResolvedDubSettings:
    config = loaded_config.config.dub
    base_dir = loaded_config.base_dir

    target_language = _required_text(
        "--to",
        "dub.target_language",
        overrides.target_language,
        config.target_language,
    )

    default_output_path = default_dubbed_video_path(video_path, target_language)
    output_path = (
        _select_path(
            overrides.output_path,
            config.output_path,
            base_dir,
            default_output_path,
        )
        or default_output_path
    )

    default_workspace_dir = Path(".dublaro") / video_path.stem
    workspace_dir = (
        _select_path(
            overrides.workspace_dir,
            config.workspace_dir,
            base_dir,
            default_workspace_dir,
        )
        or default_workspace_dir
    )

    return ResolvedDubSettings(
        source_language=_select_optional(
            overrides.source_language,
            config.source_language,
        ),
        target_language=target_language,
        output_path=output_path,
        workspace_dir=workspace_dir,
        resume=_select(overrides.resume, config.resume, False),
        overwrite=_select(overrides.overwrite, config.overwrite, False),
        preflight=_select(overrides.preflight, config.preflight, True),
        ffmpeg_executable=_select(
            overrides.ffmpeg_executable,
            config.ffmpeg_executable,
            "ffmpeg",
        ),
        asr_sample_rate=_select(
            overrides.asr_sample_rate, config.asr_sample_rate, 16000
        ),
        speech_sample_rate=_select(
            overrides.speech_sample_rate,
            config.speech_sample_rate,
            24000,
        ),
        asr_backend=_select(overrides.asr_backend, config.asr.backend, "fake"),
        model_size=_select(overrides.model_size, config.asr.model_size, "small"),
        device=_select(overrides.device, config.asr.device, "cpu"),
        compute_type=_select(overrides.compute_type, config.asr.compute_type, "int8"),
        translation_backend=_select(
            overrides.translation_backend,
            config.translation.backend,
            "fake",
        ),
        install_package=_select(
            overrides.install_package,
            config.translation.install_package,
            False,
        ),
        translation_group_segments=_select(
            overrides.translation_group_segments,
            config.translation.group_segments,
            True,
        ),
        max_translation_group_pause_seconds=_select(
            overrides.max_translation_group_pause_seconds,
            config.translation.max_group_pause_seconds,
            0.8,
        ),
        max_translation_group_duration_seconds=_select(
            overrides.max_translation_group_duration_seconds,
            config.translation.max_group_duration_seconds,
            12.0,
        ),
        text_adapter_backend=_select(
            overrides.text_adapter_backend,
            config.text_adapter.backend,
            "rules",
        ),
        tts_backend=_select(overrides.tts_backend, config.tts.backend, "fake"),
        piper_model_path=_select_path(
            overrides.piper_model_path,
            config.tts.piper_model_path,
            base_dir,
        ),
        piper_config_path=_select_path(
            overrides.piper_config_path,
            config.tts.piper_config_path,
            base_dir,
        ),
        piper_executable=_select(
            overrides.piper_executable,
            config.tts.piper_executable,
            "piper",
        ),
        piper_speaker=_select_optional(
            overrides.piper_speaker, config.tts.piper_speaker
        ),
        fit_speech=_select(overrides.fit_speech, config.fit_speech.enabled, False),
        max_speech_speedup=_select(
            overrides.max_speech_speedup,
            config.fit_speech.max_speedup,
            1.35,
        ),
        min_speech_overrun_seconds=_select(
            overrides.min_speech_overrun_seconds,
            config.fit_speech.min_overrun_seconds,
            0.05,
        ),
        mix_original_audio=_select(
            overrides.mix_original_audio, config.mix.enabled, False
        ),
        original_audio_gain=_select(
            overrides.original_audio_gain,
            config.mix.original_audio_gain,
            1.0,
        ),
        ducking_gain=_select(overrides.ducking_gain, config.mix.ducking_gain, 0.25),
        speech_gain=_select(overrides.speech_gain, config.mix.speech_gain, 1.0),
        ducking_margin_seconds=_select(
            overrides.ducking_margin_seconds,
            config.mix.ducking_margin_seconds,
            0.05,
        ),
        ducking_fade_seconds=_select(
            overrides.ducking_fade_seconds,
            config.mix.ducking_fade_seconds,
            0.05,
        ),
        export_srt=_select(overrides.export_srt, config.srt.export, False),
        srt_output_path=_select_path(
            overrides.srt_output_path,
            config.srt.output_path,
            base_dir,
        ),
        srt_text_mode=_select(overrides.srt_text_mode, config.srt.text_mode, "adapted"),
        write_manifest=_select(overrides.write_manifest, config.manifest.write, True),
        manifest_output_path=_select_path(
            overrides.manifest_output_path,
            config.manifest.output_path,
            base_dir,
        ),
    )
