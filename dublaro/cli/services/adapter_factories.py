from pathlib import Path

import typer

from dublaro.adapters.asr import AsrAdapter, FakeAsrAdapter
from dublaro.adapters.diarization import (
    DiarizationAdapter,
    FakeDiarizationAdapter,
    PyannoteDiarizationAdapter,
)
from dublaro.adapters.text_adapter import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
    FakeTextAdapter,
    OllamaTextAdapter,
    RuleBasedTextAdapter,
    TextAdapter,
)
from dublaro.adapters.translation import (
    ArgosTranslationAdapter,
    FakeTranslationAdapter,
    TranslationAdapter,
)
from dublaro.adapters.tts import FakeTtsAdapter, PiperTtsAdapter, TtsAdapter
from dublaro.cli_config import ResolvedVoiceProfileSettings
from dublaro.pipeline.preflight import SpeakerVoicePreflightSettings
from dublaro.pipeline.voices import SpeakerVoice
from dublaro.schemas import VoiceProfile


def create_asr_adapter(
    backend: str,
    *,
    model_size: str,
    device: str,
    compute_type: str,
) -> AsrAdapter:
    if backend == "fake":
        return FakeAsrAdapter()

    if backend == "faster-whisper":
        from dublaro.adapters.asr.faster_whisper import FastWhisperAsrAdapter

        return FastWhisperAsrAdapter(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )

    raise typer.BadParameter("ASR backend must be 'fake' or 'faster-whisper'.")


def create_diarization_adapter(
    backend: str,
    *,
    model_id: str,
    device: str | None = None,
    token_env_var: str | None = None,
) -> DiarizationAdapter:
    if backend == "fake":
        return FakeDiarizationAdapter()

    if backend == "pyannote":
        return PyannoteDiarizationAdapter(
            model_id=model_id,
            device=device,
            token_env_var=token_env_var,
        )

    raise typer.BadParameter("Diarization backend must be 'fake' or 'pyannote'.")


def create_translation_adapter(
    backend: str,
    *,
    auto_install: bool = False,
) -> TranslationAdapter:
    if backend == "fake":
        return FakeTranslationAdapter()

    if backend == "argos":
        return ArgosTranslationAdapter(auto_install=auto_install)

    raise typer.BadParameter("Translation backend must be 'fake' or 'argos'.")


def create_text_adapter(
    backend: str,
    *,
    ollama_model: str | None = None,
    ollama_url: str | None = None,
    ollama_timeout_seconds: float | None = None,
    ollama_temperature: float | None = None,
) -> TextAdapter:
    if backend == "fake":
        return FakeTextAdapter()

    if backend == "rules":
        return RuleBasedTextAdapter()

    if backend == "ollama":
        try:
            return OllamaTextAdapter(
                model=ollama_model or DEFAULT_OLLAMA_MODEL,
                url=ollama_url or DEFAULT_OLLAMA_URL,
                timeout_seconds=(
                    ollama_timeout_seconds
                    if ollama_timeout_seconds is not None
                    else DEFAULT_OLLAMA_TIMEOUT_SECONDS
                ),
                temperature=(
                    ollama_temperature
                    if ollama_temperature is not None
                    else DEFAULT_OLLAMA_TEMPERATURE
                ),
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error

    raise typer.BadParameter("Text adapter must be 'fake', 'rules', or 'ollama'.")


def create_tts_adapter(
    backend: str,
    *,
    piper_model_path: Path | None = None,
    piper_config_path: Path | None = None,
    piper_executable: str = "piper",
    piper_speaker: int | None = None,
) -> TtsAdapter:
    if backend == "fake":
        return FakeTtsAdapter()

    if backend == "piper":
        if piper_model_path is None:
            raise typer.BadParameter("--piper-model is required when --tts piper.")

        return PiperTtsAdapter(
            piper_model_path,
            config_path=piper_config_path,
            executable=piper_executable,
            speaker=piper_speaker,
        )

    raise typer.BadParameter("TTS backend must be 'fake' or 'piper'.")


def create_speaker_voices(
    profiles: dict[str, ResolvedVoiceProfileSettings],
) -> dict[str, SpeakerVoice] | None:
    if not profiles:
        return None

    speaker_voices: dict[str, SpeakerVoice] = {}

    for speaker_id, profile in profiles.items():
        adapter = create_tts_adapter(
            profile.tts_backend,
            piper_model_path=profile.piper_model_path,
            piper_config_path=profile.piper_config_path,
            piper_executable=profile.piper_executable,
            piper_speaker=profile.piper_speaker,
        )

        speaker_voices[speaker_id] = SpeakerVoice(
            VoiceProfile(
                speaker_id=speaker_id,
                display_name=profile.display_name,
                language=profile.language,
                tts_backend=profile.tts_backend,
                metadata=profile.metadata,
            ),
            adapter,
        )

    return speaker_voices


def create_speaker_voice_preflight_settings(
    profiles: dict[str, ResolvedVoiceProfileSettings],
) -> dict[str, SpeakerVoicePreflightSettings]:
    return {
        speaker_id: SpeakerVoicePreflightSettings(
            tts_backend=profile.tts_backend,
            piper_model_path=profile.piper_model_path,
            piper_config_path=profile.piper_config_path,
            piper_executable=profile.piper_executable,
        )
        for speaker_id, profile in profiles.items()
    }
