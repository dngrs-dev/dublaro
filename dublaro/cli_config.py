from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dublaro.adapters.diarization import DEFAULT_PYANNOTE_MODEL_ID
from dublaro.adapters.text_adapter import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
)
from dublaro.adapters.tts.piper import default_piper_config_path, read_piper_sample_rate
from dublaro.config import LoadedConfig, VoiceConfig, resolve_config_path
from dublaro.pipeline.export import (
    default_dubbed_video_path,
    default_dubbed_video_path_in_dir,
)

T = TypeVar("T")


@dataclass(frozen=True)
class DubCliOverrides:
    source_language: str | None = None
    target_language: str | None = None
    output_path: Path | None = None
    output_dir: Path | None = None
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
    diarize: bool | None = None
    diarization_backend: str | None = None
    diarization_model_id: str | None = None
    diarization_device: str | None = None
    diarization_token_env_var: str | None = None
    diarization_min_speakers: int | None = None
    diarization_max_speakers: int | None = None
    translation_backend: str | None = None
    install_package: bool | None = None
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
    repair_timing: bool | None = None
    max_timing_repair_attempts: int | None = None
    timing_repair_target_speedup: float | None = None
    fit_speech: bool | None = None
    max_speech_speedup: float | None = None
    min_speech_overrun_seconds: float | None = None
    fit_video: bool | None = None
    max_video_slowdown: float | None = None
    mix_original_audio: bool | None = None
    original_audio_gain: float | None = None
    ducking_gain: float | None = None
    speech_gain: float | None = None
    ducking_margin_seconds: float | None = None
    ducking_fade_seconds: float | None = None
    export_srt: bool | None = None
    srt_output_path: Path | None = None
    srt_text_mode: str | None = None
    subtitle_embed: str | None = None
    write_manifest: bool | None = None
    manifest_output_path: Path | None = None


@dataclass(frozen=True)
class ResolvedVoiceProfileSettings:
    speaker_id: str
    display_name: str | None
    language: str | None
    tts_backend: str
    piper_model_path: Path | None
    piper_config_path: Path | None
    piper_executable: str
    piper_speaker: int | None
    metadata: dict[str, str]


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
    diarize: bool
    diarization_backend: str
    diarization_model_id: str
    diarization_device: str | None
    diarization_token_env_var: str | None
    diarization_min_speakers: int | None
    diarization_max_speakers: int | None
    translation_backend: str
    install_package: bool
    translation_group_segments: bool
    max_translation_group_pause_seconds: float
    max_translation_group_duration_seconds: float
    max_translation_sentence_group_duration_seconds: float
    text_adapter_backend: str
    ollama_model: str
    ollama_url: str
    ollama_timeout_seconds: float
    ollama_temperature: float
    tts_backend: str
    piper_model_path: Path | None
    piper_config_path: Path | None
    piper_executable: str
    piper_speaker: int | None
    voice_profiles: dict[str, ResolvedVoiceProfileSettings]
    repair_timing: bool
    max_timing_repair_attempts: int
    timing_repair_target_speedup: float
    fit_speech: bool
    max_speech_speedup: float
    min_speech_overrun_seconds: float
    fit_video: bool
    max_video_slowdown: float
    mix_original_audio: bool
    original_audio_gain: float
    ducking_gain: float
    speech_gain: float
    ducking_margin_seconds: float
    ducking_fade_seconds: float
    export_srt: bool
    srt_output_path: Path | None
    srt_text_mode: str
    subtitle_embed: str
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


def _validate_speaker_range(
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    if min_speakers is not None and min_speakers < 1:
        raise ValueError("--min-speakers must be >= 1.")

    if max_speakers is not None and max_speakers < 1:
        raise ValueError("--max-speakers must be >= 1.")

    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise ValueError("--min-speakers cannot be greater than --max-speakers.")


def _resolve_output_path(
    *,
    video_path: Path,
    target_language: str,
    cli_output_path: Path | None,
    cli_output_dir: Path | None,
    config_output_path: Path | None,
    config_output_dir: Path | None,
    base_dir: Path | None,
) -> Path:
    if cli_output_path is not None and cli_output_dir is not None:
        raise ValueError("--output cannot be used with --output-dir.")

    if cli_output_path is not None:
        return cli_output_path

    if cli_output_dir is not None:
        return default_dubbed_video_path_in_dir(
            video_path,
            target_language,
            cli_output_dir,
        )

    resolved_config_output_path = resolve_config_path(config_output_path, base_dir)
    if resolved_config_output_path is not None:
        return resolved_config_output_path

    resolved_config_output_dir = resolve_config_path(config_output_dir, base_dir)
    if resolved_config_output_dir is not None:
        return default_dubbed_video_path_in_dir(
            video_path,
            target_language,
            resolved_config_output_dir,
        )

    return default_dubbed_video_path(video_path, target_language)


def _resolve_voice_profiles(
    config_profiles: dict[str, VoiceConfig],
    *,
    base_dir: Path | None,
    default_tts_backend: str,
    default_piper_model_path: Path | None,
    default_piper_config_path: Path | None,
    default_piper_executable: str,
    default_piper_speaker: int | None,
) -> dict[str, ResolvedVoiceProfileSettings]:
    resolved: dict[str, ResolvedVoiceProfileSettings] = {}

    for speaker_id, profile in config_profiles.items():
        resolved[speaker_id] = ResolvedVoiceProfileSettings(
            speaker_id=speaker_id,
            display_name=profile.display_name,
            language=profile.language,
            tts_backend=profile.tts_backend or default_tts_backend,
            piper_model_path=(
                resolve_config_path(profile.piper_model_path, base_dir)
                or default_piper_model_path
            ),
            piper_config_path=(
                resolve_config_path(profile.piper_config_path, base_dir)
                or default_piper_config_path
            ),
            piper_executable=profile.piper_executable or default_piper_executable,
            piper_speaker=(
                profile.piper_speaker
                if profile.piper_speaker is not None
                else default_piper_speaker
            ),
            metadata=dict(profile.metadata),
        )

    return resolved


def _piper_sample_rate(
    model_path: Path | None,
    config_path: Path | None,
) -> int | None:
    metadata_path = config_path

    if metadata_path is None and model_path is not None:
        metadata_path = default_piper_config_path(model_path)

    if metadata_path is None:
        return None

    return read_piper_sample_rate(metadata_path)


def _resolve_speech_sample_rate(
    *,
    cli_value: int | None,
    config_value: int | None,
    tts_backend: str,
    piper_model_path: Path | None,
    piper_config_path: Path | None,
    voice_profiles: dict[str, ResolvedVoiceProfileSettings] | None = None,
    default: int = 24_000,
) -> int:
    detected_rates: set[int] = set()

    if tts_backend == "piper":
        rate = _piper_sample_rate(piper_model_path, piper_config_path)
        if rate is not None:
            detected_rates.add(rate)

    for profile in (voice_profiles or {}).values():
        if profile.tts_backend != "piper":
            continue

        rate = _piper_sample_rate(profile.piper_model_path, profile.piper_config_path)
        if rate is not None:
            detected_rates.add(rate)

    if len(detected_rates) > 1:
        rates = ", ".join(str(rate) for rate in sorted(detected_rates))
        raise ValueError(
            "Piper voices use different sample rates "
            f"({rates}). Use voices with the same sample rate in one dub run."
        )

    configured_value = _select_optional(cli_value, config_value)
    if configured_value is not None:
        if detected_rates and configured_value not in detected_rates:
            detected_rate = next(iter(detected_rates))
            raise ValueError(
                "Configured speech sample rate does not match Piper voice sample rate: "
                f"{configured_value} != {detected_rate}."
            )
        return configured_value

    if detected_rates:
        return next(iter(detected_rates))

    return default


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

    output_path = _resolve_output_path(
        video_path=video_path,
        target_language=target_language,
        cli_output_path=overrides.output_path,
        cli_output_dir=overrides.output_dir,
        config_output_path=config.output_path,
        config_output_dir=config.output_dir,
        base_dir=base_dir,
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

    tts_backend = _select(overrides.tts_backend, config.tts.backend, "fake")

    piper_model_path = _select_path(
        overrides.piper_model_path,
        config.tts.piper_model_path,
        base_dir,
    )

    piper_config_path = _select_path(
        overrides.piper_config_path,
        config.tts.piper_config_path,
        base_dir,
    )

    piper_executable = _select(
        overrides.piper_executable,
        config.tts.piper_executable,
        "piper",
    )
    piper_speaker = _select_optional(
        overrides.piper_speaker,
        config.tts.piper_speaker,
    )

    voice_profiles = _resolve_voice_profiles(
        loaded_config.config.voices,
        base_dir=base_dir,
        default_tts_backend=tts_backend,
        default_piper_model_path=piper_model_path,
        default_piper_config_path=piper_config_path,
        default_piper_executable=piper_executable,
        default_piper_speaker=piper_speaker,
    )

    speech_sample_rate = _resolve_speech_sample_rate(
        cli_value=overrides.speech_sample_rate,
        config_value=config.speech_sample_rate,
        tts_backend=tts_backend,
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        voice_profiles=voice_profiles,
    )

    diarization_min_speakers = _select_optional(
        overrides.diarization_min_speakers,
        config.diarization.min_speakers,
    )
    diarization_max_speakers = _select_optional(
        overrides.diarization_max_speakers,
        config.diarization.max_speakers,
    )
    _validate_speaker_range(diarization_min_speakers, diarization_max_speakers)

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
        speech_sample_rate=speech_sample_rate,
        asr_backend=_select(overrides.asr_backend, config.asr.backend, "fake"),
        model_size=_select(overrides.model_size, config.asr.model_size, "small"),
        device=_select(overrides.device, config.asr.device, "cpu"),
        compute_type=_select(overrides.compute_type, config.asr.compute_type, "int8"),
        diarize=_select(overrides.diarize, config.diarization.enabled, False),
        diarization_backend=_select(
            overrides.diarization_backend,
            config.diarization.backend,
            "fake",
        ),
        diarization_model_id=_select(
            overrides.diarization_model_id,
            config.diarization.model_id,
            DEFAULT_PYANNOTE_MODEL_ID,
        ),
        diarization_device=_select_optional(
            overrides.diarization_device,
            config.diarization.device,
        ),
        diarization_token_env_var=_select_optional(
            overrides.diarization_token_env_var,
            config.diarization.token_env_var,
        ),
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
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
        max_translation_sentence_group_duration_seconds=_select(
            overrides.max_translation_sentence_group_duration_seconds,
            config.translation.max_sentence_group_duration_seconds,
            24.0,
        ),
        text_adapter_backend=_select(
            overrides.text_adapter_backend,
            config.text_adapter.backend,
            "rules",
        ),
        ollama_model=_select(
            overrides.ollama_model,
            config.text_adapter.ollama_model,
            DEFAULT_OLLAMA_MODEL,
        ),
        ollama_url=_select(
            overrides.ollama_url,
            config.text_adapter.ollama_url,
            DEFAULT_OLLAMA_URL,
        ),
        ollama_timeout_seconds=_select(
            overrides.ollama_timeout_seconds,
            config.text_adapter.ollama_timeout_seconds,
            DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        ),
        ollama_temperature=_select(
            overrides.ollama_temperature,
            config.text_adapter.ollama_temperature,
            DEFAULT_OLLAMA_TEMPERATURE,
        ),
        tts_backend=tts_backend,
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        piper_executable=piper_executable,
        piper_speaker=piper_speaker,
        voice_profiles=voice_profiles,
        repair_timing=_select(
            overrides.repair_timing,
            config.timing_repair.enabled,
            False,
        ),
        max_timing_repair_attempts=_select(
            overrides.max_timing_repair_attempts,
            config.timing_repair.max_attempts,
            2,
        ),
        timing_repair_target_speedup=_select(
            overrides.timing_repair_target_speedup,
            config.timing_repair.target_speedup,
            1.15,
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
        fit_video=_select(overrides.fit_video, config.fit_video.enabled, False),
        max_video_slowdown=_select(
            overrides.max_video_slowdown,
            config.fit_video.max_slowdown,
            1.5,
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
        subtitle_embed=_select(overrides.subtitle_embed, config.srt.embed, "none"),
        write_manifest=_select(overrides.write_manifest, config.manifest.write, True),
        manifest_output_path=_select_path(
            overrides.manifest_output_path,
            config.manifest.output_path,
            base_dir,
        ),
    )
