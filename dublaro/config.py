import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from dublaro.pipeline.subtitles import SrtTextMode


class DublaroConfigError(ValueError):
    pass


class AsrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str | None = None
    model_size: str | None = None
    device: str | None = None
    compute_type: str | None = None


class TranslationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str | None = None
    install_package: bool | None = None
    group_segments: bool | None = None
    max_group_pause_seconds: float | None = None
    max_group_duration_seconds: float | None = None


class TextAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str | None = None


class TtsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str | None = None
    piper_model_path: Path | None = None
    piper_config_path: Path | None = None
    piper_executable: str | None = None
    piper_speaker: int | None = None


class FitSpeechConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    max_speedup: float | None = None
    min_overrun_seconds: float | None = None


class MixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    original_audio_gain: float | None = None
    ducking_gain: float | None = None
    speech_gain: float | None = None
    ducking_margin_seconds: float | None = None
    ducking_fade_seconds: float | None = None


class SrtConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    export: bool | None = None
    output_path: Path | None = None
    text_mode: SrtTextMode | None = None


class ManifestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    write: bool | None = None
    output_path: Path | None = None


class DubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    asr: AsrConfig = AsrConfig()
    translation: TranslationConfig = TranslationConfig()
    text_adapter: TextAdapterConfig = TextAdapterConfig()
    tts: TtsConfig = TtsConfig()
    fit_speech: FitSpeechConfig = FitSpeechConfig()
    mix: MixConfig = MixConfig()
    srt: SrtConfig = SrtConfig()
    manifest: ManifestConfig = ManifestConfig()


class DublaroConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dub: DubConfig = DubConfig()


@dataclass(frozen=True)
class LoadedConfig:
    config: DublaroConfig
    path: Path | None
    base_dir: Path | None


def load_config(config_path: Path | None) -> LoadedConfig:
    if config_path is None:
        return LoadedConfig(
            config=DublaroConfig(),
            path=None,
            base_dir=None,
        )

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DublaroConfigError(
            f"Config file does not exist: {config_path}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise DublaroConfigError(f"Invalid TOML in {config_path}: {error}") from error

    try:
        config = DublaroConfig.model_validate(data)
    except ValidationError as error:
        raise DublaroConfigError(
            f"Invalid config in {config_path}:\n{error}"
        ) from error

    return LoadedConfig(
        config=config,
        path=config_path,
        base_dir=config_path.parent,
    )


def resolve_config_path(path: Path | None, base_dir: Path | None) -> Path | None:
    if path is None:
        return None

    if path.is_absolute() or base_dir is None:
        return path

    return base_dir / path
