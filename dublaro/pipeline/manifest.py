from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dublaro import __version__
from dublaro.pipeline.voices import SpeakerVoice

JsonScalar = str | int | float | bool | None

_ADAPTER_SETTING_NAMES = (
    "model_size",
    "device",
    "compute_type",
    "model_id",
    "device",
    "token_env_var",
    "auto_install",
    "model_path",
    "config_path",
    "executable",
    "speaker",
    "model_sample_rate",
)


class LanguageManifest(BaseModel):
    source: str | None = None
    target: str


class AdapterManifest(BaseModel):
    name: str
    settings: dict[str, JsonScalar] = Field(default_factory=dict)


class VoiceProfileManifest(BaseModel):
    speaker_id: str
    display_name: str | None = None
    language: str | None = None
    tts_backend: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SpeakerVoiceManifest(BaseModel):
    profile: VoiceProfileManifest
    adapter: AdapterManifest


class DubbingAdaptersManifest(BaseModel):
    asr: AdapterManifest
    diarization: AdapterManifest | None = None
    translation: AdapterManifest
    text_adapter: AdapterManifest
    tts: AdapterManifest
    speaker_voices: dict[str, SpeakerVoiceManifest] = Field(default_factory=dict)


class DubbingOptionsManifest(BaseModel):
    asr_sample_rate: int
    diarize: bool
    diarization_min_speakers: int | None = None
    diarization_max_speakers: int | None = None
    speech_sample_rate: int
    resume: bool
    fit_speech: bool
    max_speech_speedup: float
    min_speech_overrun_seconds: float
    mix_original_audio: bool
    original_audio_gain: float
    ducking_gain: float
    speech_gain: float
    ducking_margin_seconds: float
    ducking_fade_seconds: float
    translation_group_segments: bool
    max_translation_group_pause_seconds: float
    max_translation_group_duration_seconds: float
    export_srt: bool
    srt_text_mode: str
    ffmpeg_executable: str
    overwrite: bool


class DubbingArtifactsManifest(BaseModel):
    workspace_dir: str
    extracted_audio_path: str
    source_transcript_path: str
    diarized_transcript_path: str | None = None
    translated_transcript_path: str
    adapted_transcript_path: str
    synthesized_transcript_path: str
    speech_dir: str
    speech_track_path: str
    dubbed_video_path: str
    fitted_transcript_path: str | None = None
    fitted_speech_dir: str | None = None
    mix_original_audio_path: str | None = None
    mixed_audio_path: str | None = None
    srt_path: str | None = None
    manifest_path: str | None = None


class DubbingManifest(BaseModel):
    schema_version: int = 1
    dublaro_version: str = __version__
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    input_video_path: str
    output_video_path: str
    language: LanguageManifest
    adapters: DubbingAdaptersManifest
    options: DubbingOptionsManifest
    artifacts: DubbingArtifactsManifest
    metadata: dict[str, str] = Field(default_factory=dict)


def build_dubbing_manifest(
    *,
    started_at: datetime,
    finished_at: datetime,
    input_video_path: str | Path,
    output_video_path: str | Path,
    source_language: str | None,
    target_language: str,
    asr_adapter: object,
    diarization_adapter: object | None = None,
    translation_adapter: object,
    text_adapter: object,
    tts_adapter: object,
    speaker_voices: Mapping[str, SpeakerVoice] | None = None,
    options: DubbingOptionsManifest,
    artifacts: DubbingArtifactsManifest,
    metadata: dict[str, str] | None = None,
) -> DubbingManifest:
    return DubbingManifest(
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
        input_video_path=_path_to_string(input_video_path),
        output_video_path=_path_to_string(output_video_path),
        language=LanguageManifest(
            source=source_language,
            target=target_language,
        ),
        adapters=DubbingAdaptersManifest(
            asr=describe_adapter(asr_adapter),
            diarization=(
                describe_adapter(diarization_adapter)
                if diarization_adapter is not None
                else None
            ),
            translation=describe_adapter(translation_adapter),
            text_adapter=describe_adapter(text_adapter),
            tts=describe_adapter(tts_adapter),
            speaker_voices={
                speaker_id: describe_speaker_voice(speaker_voice)
                for speaker_id, speaker_voice in (speaker_voices or {}).items()
            },
        ),
        options=options,
        artifacts=artifacts,
        metadata=metadata or {},
    )


def save_manifest(
    manifest: DubbingManifest,
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_file


def describe_speaker_voice(speaker_voice: SpeakerVoice) -> SpeakerVoiceManifest:
    profile = speaker_voice.profile

    return SpeakerVoiceManifest(
        profile=VoiceProfileManifest(
            speaker_id=profile.speaker_id,
            display_name=profile.display_name,
            language=profile.language,
            tts_backend=profile.tts_backend,
            metadata=dict(profile.metadata),
        ),
        adapter=describe_adapter(speaker_voice.adapter),
    )


def describe_adapter(adapter: object) -> AdapterManifest:
    settings: dict[str, JsonScalar] = {}

    for name in _ADAPTER_SETTING_NAMES:
        if not hasattr(adapter, name):
            continue

        settings[name] = _to_json_scalar(getattr(adapter, name))

    return AdapterManifest(
        name=str(getattr(adapter, "name", adapter.__class__.__name__)),
        settings=settings,
    )


def _path_to_string(path: str | Path) -> str:
    return str(Path(path))


def _to_json_scalar(value: Any) -> JsonScalar:
    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)
