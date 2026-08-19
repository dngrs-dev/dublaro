import importlib
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dublaro.adapters.text_adapter import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
    check_ollama_model_available,
)
from dublaro.adapters.translation import (
    DEFAULT_OLLAMA_TRANSLATION_MODEL,
    DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_TRANSLATION_URL,
)

PreflightSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class PreflightIssue:
    severity: PreflightSeverity
    code: str
    message: str
    hint: str | None = None


@dataclass(frozen=True)
class SpeakerVoicePreflightSettings:
    tts_backend: str
    piper_model_path: str | Path | None = None
    piper_config_path: str | Path | None = None
    piper_executable: str = "piper"


@dataclass(frozen=True)
class DubPreflightReport:
    issues: tuple[PreflightIssue, ...]

    @property
    def errors(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def passed(self) -> bool:
        return not self.has_errors


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
) -> DubPreflightReport:
    issues: list[PreflightIssue] = []

    video_file = Path(video_path)
    output_file = Path(output_path)
    workspace = Path(workspace_dir)

    _check_input_file(issues, "Input video", video_file)
    _check_workspace_path(issues, workspace)
    _check_output_path(issues, "Output video", output_file, overwrite)
    _check_not_same_path(issues, video_file, output_file)

    if export_srt:
        srt_file = (
            Path(srt_output_path)
            if srt_output_path is not None
            else output_file.with_suffix(".srt")
        )
        _check_output_path(
            issues, "SRT output", srt_file, overwrite, allow_existing=resume
        )

    if write_manifest and manifest_output_path is not None:
        _check_output_path(
            issues,
            "Manifest output",
            Path(manifest_output_path),
            overwrite,
            allow_existing=resume,
        )

    _check_ffmpeg(issues, ffmpeg_executable)
    _check_source_separation(
        issues,
        background_mode=background_mode,
        source_separation_backend=source_separation_backend,
        demucs_executable=demucs_executable,
    )
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
        enabled=translation_backend == "ollama",
        label="Translation Ollama",
        ollama_model=translation_ollama_model,
        ollama_url=translation_ollama_url,
        ollama_timeout_seconds=translation_ollama_timeout_seconds,
        checked_targets=checked_ollama_targets,
    )

    _check_ollama(
        issues,
        enabled=text_adapter_backend == "ollama",
        label="Text adapter Ollama",
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        ollama_timeout_seconds=ollama_timeout_seconds,
        checked_targets=checked_ollama_targets,
    )

    return DubPreflightReport(tuple(issues))


def _check_input_file(
    issues: list[PreflightIssue],
    label: str,
    path: Path,
) -> None:
    if not path.exists():
        _add_error(
            issues,
            "input_missing",
            f"{label} does not exist: {path}",
            "Check the path and try again.",
        )
        return

    if not path.is_file():
        _add_error(
            issues,
            "input_not_file",
            f"{label} is not a file: {path}",
        )


def _check_workspace_path(
    issues: list[PreflightIssue],
    workspace: Path,
) -> None:
    if workspace.exists() and not workspace.is_dir():
        _add_error(
            issues,
            "workspace_not_directory",
            f"Workspace path exists but is not a directory: {workspace}",
            "Choose another --workspace path.",
        )


def _check_output_path(
    issues: list[PreflightIssue],
    label: str,
    path: Path,
    overwrite: bool,
    allow_existing: bool = False,
) -> None:
    parent = path.parent

    if parent.exists() and not parent.is_dir():
        _add_error(
            issues,
            "output_parent_not_directory",
            f"{label} parent is not a directory: {parent}",
        )

    if not path.exists():
        return

    if path.is_dir():
        _add_error(
            issues,
            "output_is_directory",
            f"{label} path is a directory: {path}",
            "Choose a file path.",
        )
        return

    if not overwrite and not allow_existing:
        _add_error(
            issues,
            "output_exists",
            f"{label} already exists: {path}",
            "Use --overwrite to replace it.",
        )


def _check_not_same_path(
    issues: list[PreflightIssue],
    input_path: Path,
    output_path: Path,
) -> None:
    if input_path.resolve() == output_path.resolve():
        _add_error(
            issues,
            "output_same_as_input",
            "Output video path cannot be the same as input video path.",
            "Choose a different --output path.",
        )


def _check_ffmpeg(
    issues: list[PreflightIssue],
    executable: str,
) -> None:
    resolved = shutil.which(executable)

    if resolved is None:
        _add_error(
            issues,
            "ffmpeg_missing",
            f"ffmpeg executable was not found: {executable}",
            "Install ffmpeg or pass --ffmpeg with the full executable path.",
        )
        return

    try:
        result = subprocess.run(
            [resolved, "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        _add_error(
            issues,
            "ffmpeg_not_runnable",
            f"ffmpeg could not be executed: {error}",
        )
        return

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        _add_error(
            issues,
            "ffmpeg_failed",
            f"ffmpeg exists but failed to run: {details}",
        )


def _check_source_separation(
    issues: list[PreflightIssue],
    *,
    background_mode: str,
    source_separation_backend: str,
    demucs_executable: str,
) -> None:
    if background_mode != "separated":
        return

    if source_separation_backend == "fake":
        return

    if source_separation_backend != "demucs":
        _add_error(
            issues,
            "source_separation_backend_unknown",
            f"Unknown source separation backend: {source_separation_backend}",
            "Use --source-separation fake or --source-separation demucs.",
        )
        return

    if shutil.which(demucs_executable) is None:
        _add_error(
            issues,
            "demucs_executable_missing",
            f"Demucs executable was not found: {demucs_executable}",
            'Install it with: pip install -e ".[source-separation]"',
        )


def _check_piper(
    issues: list[PreflightIssue],
    *,
    tts_backend: str,
    piper_model_path: str | Path | None,
    piper_config_path: str | Path | None,
    piper_executable: str,
    label: str = "Piper",
    missing_model_message: str = "--piper-model is required when --tts piper.",
    checked_executables: set[str] | None = None,
) -> None:
    if tts_backend != "piper":
        return

    if piper_model_path is None:
        _add_error(
            issues,
            "piper_model_missing",
            missing_model_message,
        )
    else:
        _check_input_file(issues, f"{label} model", Path(piper_model_path))

    if piper_config_path is not None:
        _check_input_file(issues, f"{label} config", Path(piper_config_path))

    if checked_executables is not None:
        if piper_executable in checked_executables:
            return

        checked_executables.add(piper_executable)

    if shutil.which(piper_executable) is None:
        _add_error(
            issues,
            "piper_executable_missing",
            f"{label} executable was not found: {piper_executable}",
            "Install Piper or pass --piper-executable with the full path.",
        )


def _check_argos(
    issues: list[PreflightIssue],
    *,
    asr_backend: str,
    translation_backend: str,
    source_language: str | None,
    target_language: str,
    auto_install: bool,
) -> None:
    if translation_backend != "argos":
        return

    translate_module = _load_argos_translate_module(issues)
    if translate_module is None:
        return

    if source_language is None:
        if asr_backend == "fake":
            _add_error(
                issues,
                "argos_source_language_missing",
                "Argos translation needs --from when fake ASR is used.",
                "Pass --from with the source language, for example --from en.",
            )
            return

        _add_warning(
            issues,
            "argos_package_not_verified",
            "Argos package could not be verified because source language is auto-detected.",
            "Pass --from to verify the exact Argos language package before running.",
        )
        return

    if auto_install:
        return

    try:
        translation = translate_module.get_translation_from_codes(
            source_language,
            target_language,
        )
    except Exception:
        translation = None

    if translation is None:
        _add_error(
            issues,
            "argos_package_missing",
            f"No installed Argos package found for {source_language}->{target_language}.",
            "Run with --install-package or install the Argos package manually.",
        )


def _load_argos_translate_module(
    issues: list[PreflightIssue],
) -> Any | None:
    try:
        importlib.import_module("argostranslate.package")
        return importlib.import_module("argostranslate.translate")
    except ImportError:
        _add_error(
            issues,
            "argos_missing",
            "argostranslate is not installed.",
            'Install it with: pip install -e ".[translation]"',
        )
        return None


def _check_ollama(
    issues: list[PreflightIssue],
    *,
    enabled: bool,
    label: str,
    ollama_model: str,
    ollama_url: str,
    ollama_timeout_seconds: float,
    checked_targets: set[tuple[str, str]] | None = None,
) -> None:
    if not enabled:
        return

    target = (ollama_url, ollama_model)
    if checked_targets is not None:
        if target in checked_targets:
            return

        checked_targets.add(target)

    timeout_seconds = min(
        ollama_timeout_seconds,
        DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS,
    )

    try:
        model_available = check_ollama_model_available(
            model=ollama_model,
            url=ollama_url,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as error:
        _add_error(
            issues,
            "ollama_unavailable",
            f"{label} is not available: {error}",
            "Start Ollama or pass the correct Ollama server URL.",
        )
        return

    if not model_available:
        _add_error(
            issues,
            "ollama_model_missing",
            f"{label} model is not available: {ollama_model}",
            f"Run: ollama pull {ollama_model}",
        )


def _add_error(
    issues: list[PreflightIssue],
    code: str,
    message: str,
    hint: str | None = None,
) -> None:
    issues.append(PreflightIssue("error", code, message, hint))


def _add_warning(
    issues: list[PreflightIssue],
    code: str,
    message: str,
    hint: str | None = None,
) -> None:
    issues.append(PreflightIssue("warning", code, message, hint))
