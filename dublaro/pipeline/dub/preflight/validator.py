from collections.abc import Mapping
from pathlib import Path

from dublaro.adapters.text_adapter import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
)
from dublaro.adapters.translation import (
    DEFAULT_OLLAMA_TRANSLATION_MODEL,
    DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_TRANSLATION_URL,
)
from dublaro.pipeline.dub.preflight.checks import (
    _check_argos,
    _check_ffmpeg,
    _check_input_file,
    _check_not_same_path,
    _check_ollama,
    _check_output_path,
    _check_piper,
    _check_source_separation,
    _check_workspace_path,
)
from dublaro.pipeline.dub.preflight.models import (
    DubPreflightReport,
    PreflightIssue,
    PreflightScope,
    SpeakerVoicePreflightSettings,
)


def validate_dub_preflight(
    *,
    video_path: str | Path,
    output_path: str | Path,
    workspace_dir: str | Path,
    overwrite: bool,
    ffmpeg_executable: str,
    asr_backend: str,
    translation_backend: str,
    source_language: str | None,
    target_language: str,
    install_translation_package: bool,
    translation_ollama_model: str = DEFAULT_OLLAMA_TRANSLATION_MODEL,
    translation_ollama_url: str = DEFAULT_OLLAMA_TRANSLATION_URL,
    translation_ollama_timeout_seconds: float = DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS,
    text_adapter_backend: str = "rules",
    background_mode: str = "speech-only",
    source_separation_backend: str = "fake",
    demucs_executable: str = "demucs",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_timeout_seconds: float = DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS,
    tts_backend: str,
    piper_model_path: str | Path | None = None,
    piper_config_path: str | Path | None = None,
    piper_executable: str = "piper",
    speaker_voices: Mapping[str, SpeakerVoicePreflightSettings] | None = None,
    export_srt: bool = False,
    srt_output_path: str | Path | None = None,
    write_manifest: bool = True,
    manifest_output_path: str | Path | None = None,
    resume: bool = False,
    scope: PreflightScope | None = None,
) -> DubPreflightReport:
    issues: list[PreflightIssue] = []

    video_file = Path(video_path)
    output_file = Path(output_path)
    workspace = Path(workspace_dir)
    scope = scope or PreflightScope()

    _check_input_file(issues, "Input video", video_file)
    _check_workspace_path(issues, workspace)
    if scope.export_video:
        _check_output_path(issues, "Output video", output_file, overwrite)
        _check_not_same_path(issues, video_file, output_file)

    if export_srt and scope.export_srt:
        srt_file = (
            Path(srt_output_path)
            if srt_output_path is not None
            else output_file.with_suffix(".srt")
        )
        _check_output_path(
            issues, "SRT output", srt_file, overwrite, allow_existing=resume
        )

    if write_manifest and scope.write_manifest and manifest_output_path is not None:
        _check_output_path(
            issues,
            "Manifest output",
            Path(manifest_output_path),
            overwrite,
            allow_existing=resume,
        )

    _check_ffmpeg(issues, ffmpeg_executable)
    if scope.mix_audio:
        _check_source_separation(
            issues,
            background_mode=background_mode,
            source_separation_backend=source_separation_backend,
            demucs_executable=demucs_executable,
        )

    if scope.synthesize_speech:
        checked_piper_executables: set[str] = set()
        _check_piper(
            issues,
            tts_backend=tts_backend,
            piper_model_path=piper_model_path,
            piper_config_path=piper_config_path,
            piper_executable=piper_executable,
            checked_executables=checked_piper_executables,
        )

        for speaker_id, speaker_voice in (speaker_voices or {}).items():
            _check_piper(
                issues,
                tts_backend=speaker_voice.tts_backend,
                piper_model_path=speaker_voice.piper_model_path,
                piper_config_path=speaker_voice.piper_config_path,
                piper_executable=speaker_voice.piper_executable,
                label=f'Speaker voice "{speaker_id}" Piper',
                missing_model_message=(
                    f'Piper model is required for speaker voice "{speaker_id}" '
                    "when its TTS backend is piper."
                ),
                checked_executables=checked_piper_executables,
            )

    if scope.translate_text:
        _check_argos(
            issues,
            asr_backend=asr_backend,
            translation_backend=translation_backend,
            source_language=source_language,
            target_language=target_language,
            auto_install=install_translation_package,
        )

    checked_ollama_targets: set[tuple[str, str]] = set()

    _check_ollama(
        issues,
        enabled=scope.translate_text and translation_backend == "ollama",
        label="Translation Ollama",
        ollama_model=translation_ollama_model,
        ollama_url=translation_ollama_url,
        ollama_timeout_seconds=translation_ollama_timeout_seconds,
        checked_targets=checked_ollama_targets,
    )

    _check_ollama(
        issues,
        enabled=scope.adapt_text and text_adapter_backend == "ollama",
        label="Text adapter Ollama",
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        ollama_timeout_seconds=ollama_timeout_seconds,
        checked_targets=checked_ollama_targets,
    )

    return DubPreflightReport(tuple(issues))
