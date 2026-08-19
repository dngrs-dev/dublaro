import importlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dublaro.adapters.tts.piper import default_piper_config_path
from dublaro.cli.reports.preview import preview_speaker_voice
from dublaro.config import (
    DublaroConfigError,
    LoadedConfig,
    load_config,
    resolve_config_path,
)

DoctorStatus = Literal["ok", "warning", "error", "skipped"]


@dataclass(frozen=True)
class DoctorCheck:
    category: str
    name: str
    status: DoctorStatus
    message: str
    hint: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def has_errors(self) -> bool:
        return any(check.status == "error" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.status == "warning" for check in self.checks)


def build_doctor_report(
    *,
    config_path: Path | None = None,
    ffmpeg_executable: str | None = None,
    piper_executable: str | None = None,
    source_language: str | None = None,
    target_language: str | None = None,
) -> DoctorReport:
    checks: list[DoctorCheck] = []
    loaded_config = _load_doctor_config(checks, config_path)

    ffmpeg = ffmpeg_executable or _config_ffmpeg_executable(loaded_config) or "ffmpeg"
    _check_executable(
        checks,
        category="tools",
        name="ffmpeg",
        executable=ffmpeg,
        version_args=("-version",),
        missing_hint="Install ffmpeg or pass --ffmpeg with the full executable path.",
    )

    if loaded_config is None:
        _check_explicit_piper_executable(checks, piper_executable)
        _check_huggingface_cache(checks)
        return DoctorReport(tuple(checks))

    _check_piper_configuration(
        checks,
        loaded_config,
        piper_executable=piper_executable,
    )
    _check_hf_token(checks, loaded_config)
    _check_argos_package(
        checks,
        loaded_config,
        source_language=source_language,
        target_language=target_language,
    )
    _check_source_separation_configuration(checks, loaded_config)
    _check_cache_paths(checks, loaded_config)

    return DoctorReport(tuple(checks))


def _load_doctor_config(
    checks: list[DoctorCheck],
    config_path: Path | None,
) -> LoadedConfig | None:
    resolved_config_path = config_path or _default_config_path()

    if resolved_config_path is None:
        checks.append(
            DoctorCheck(
                category="config",
                name="config",
                status="skipped",
                message="No config file loaded; checking default tool names.",
                hint="Pass --config or create dublaro.toml to check configured voices.",
            )
        )
        return load_config(None)

    try:
        loaded_config = load_config(resolved_config_path)
    except DublaroConfigError as error:
        checks.append(
            DoctorCheck(
                category="config",
                name="config",
                status="error",
                message=str(error),
                hint="Fix the TOML file and run doctor again.",
            )
        )
        return None

    checks.append(
        DoctorCheck(
            category="config",
            name="config",
            status="ok",
            message=f"Loaded config: {resolved_config_path}",
        )
    )
    return loaded_config


def _default_config_path() -> Path | None:
    path = Path("dublaro.toml")
    return path if path.is_file() else None


def _config_ffmpeg_executable(loaded_config: LoadedConfig | None) -> str | None:
    if loaded_config is None:
        return None

    return loaded_config.config.dub.ffmpeg_executable


def _check_explicit_piper_executable(
    checks: list[DoctorCheck],
    piper_executable: str | None,
) -> None:
    if piper_executable is None:
        checks.append(
            DoctorCheck(
                category="piper",
                name="piper",
                status="skipped",
                message="No Piper executable check requested.",
                hint="Pass --piper-executable or configure Piper voices.",
            )
        )
        return

    _check_executable(
        checks,
        category="piper",
        name="Piper executable",
        executable=piper_executable,
        version_args=None,
        missing_hint="Install Piper or pass --piper-executable with the full path.",
    )


def _check_piper_configuration(
    checks: list[DoctorCheck],
    loaded_config: LoadedConfig,
    *,
    piper_executable: str | None,
) -> None:
    checked_executables: set[str] = set()
    checked_piper_count = 0

    tts_config = loaded_config.config.dub.tts
    fallback_tts_backend = tts_config.backend or "fake"

    if fallback_tts_backend == "piper":
        checked_piper_count += 1
        model_path = resolve_config_path(
            tts_config.piper_model_path, loaded_config.base_dir
        )
        config_path = resolve_config_path(
            tts_config.piper_config_path, loaded_config.base_dir
        )

        _check_piper_target(
            checks,
            label="Default Piper",
            model_path=model_path,
            config_path=config_path,
            executable=piper_executable or tts_config.piper_executable or "piper",
            checked_executables=checked_executables,
        )

    if loaded_config.config.voices:
        checks.append(
            DoctorCheck(
                category="voices",
                name="speaker voices",
                status="ok",
                message=f"{len(loaded_config.config.voices)} speaker voice profile(s) configured.",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                category="voices",
                name="speaker voices",
                status="skipped",
                message="No speaker voice profiles configured.",
            )
        )

    for speaker_id in sorted(loaded_config.config.voices):
        voice = preview_speaker_voice(speaker_id, loaded_config)
        if voice.tts_backend != "piper":
            continue

        checked_piper_count += 1
        _check_piper_target(
            checks,
            label=f'Speaker "{speaker_id}" Piper',
            model_path=voice.piper_model_path,
            config_path=voice.piper_config_path,
            executable=piper_executable or voice.piper_executable,
            checked_executables=checked_executables,
        )

    if checked_piper_count == 0:
        checks.append(
            DoctorCheck(
                category="piper",
                name="piper",
                status="skipped",
                message="No Piper TTS backend configured.",
            )
        )


def _check_piper_target(
    checks: list[DoctorCheck],
    *,
    label: str,
    model_path: Path | None,
    config_path: Path | None,
    executable: str,
    checked_executables: set[str],
) -> None:
    inferred_config_path = (
        config_path
        if config_path is not None
        else default_piper_config_path(model_path) if model_path is not None else None
    )

    _check_file(checks, category="piper", name=f"{label} model", path=model_path)
    _check_file(
        checks, category="piper", name=f"{label} config", path=inferred_config_path
    )

    if executable in checked_executables:
        return

    checked_executables.add(executable)
    _check_executable(
        checks,
        category="piper",
        name="Piper executable",
        executable=executable,
        version_args=None,
        missing_hint="Install Piper or pass --piper-executable with the full path.",
    )


def _check_hf_token(checks: list[DoctorCheck], loaded_config: LoadedConfig) -> None:
    diarization = loaded_config.config.dub.diarization
    backend = diarization.backend or "fake"

    if diarization.enabled is False or backend != "pyannote":
        checks.append(
            DoctorCheck(
                category="huggingface",
                name="HF token",
                status="skipped",
                message="Pyannote diarization is not enabled.",
            )
        )
        return

    token_names = (
        (diarization.token_env_var,)
        if diarization.token_env_var is not None
        else ("HF_TOKEN", "HUGGINGFACE_TOKEN")
    )
    found_token_name = next((name for name in token_names if os.getenv(name)), None)

    if found_token_name is not None:
        checks.append(
            DoctorCheck(
                category="huggingface",
                name="HF token",
                status="ok",
                message=f"Token found in {found_token_name}.",
            )
        )
        return

    checks.append(
        DoctorCheck(
            category="huggingface",
            name="HF token",
            status="warning",
            message="No Hugging Face token environment variable was found.",
            hint="Set HF_TOKEN if pyannote needs to download gated models.",
        )
    )


def _check_argos_package(
    checks: list[DoctorCheck],
    loaded_config: LoadedConfig,
    *,
    source_language: str | None,
    target_language: str | None,
) -> None:
    dub_config = loaded_config.config.dub
    translation = dub_config.translation

    if (translation.backend or "fake") != "argos":
        checks.append(
            DoctorCheck(
                category="argos",
                name="Argos package",
                status="skipped",
                message="Argos translation is not configured.",
            )
        )
        return

    source = source_language or dub_config.source_language
    target = target_language or dub_config.target_language

    try:
        importlib.import_module("argostranslate.package")
        translate_module = importlib.import_module("argostranslate.translate")
    except ImportError:
        checks.append(
            DoctorCheck(
                category="argos",
                name="Argos package",
                status="error",
                message="argostranslate is not installed.",
                hint='Install it with: pip install -e ".[translation]"',
            )
        )
        return

    if source is None or target is None:
        checks.append(
            DoctorCheck(
                category="argos",
                name="Argos package",
                status="warning",
                message="Argos is installed, but the language pair could not be checked.",
                hint="Set dub.source_language and dub.target_language or pass --from and --to.",
            )
        )
        return

    package = _get_argos_translation(translate_module, source, target)
    if package is not None:
        checks.append(
            DoctorCheck(
                category="argos",
                name="Argos package",
                status="ok",
                message=f"Installed translation package found: {source}->{target}.",
            )
        )
        return

    status: DoctorStatus = "warning" if translation.install_package else "error"
    checks.append(
        DoctorCheck(
            category="argos",
            name="Argos package",
            status=status,
            message=f"No installed Argos package found for {source}->{target}.",
            hint="Enable install_package or install the Argos package manually.",
        )
    )


def _check_source_separation_configuration(
    checks: list[DoctorCheck],
    loaded_config: LoadedConfig,
) -> None:
    dub_config = loaded_config.config.dub
    background_mode = dub_config.background_mode

    if background_mode is None:
        background_mode = (
            "ducked" if (dub_config.mix.enabled or False) else "speech-only"
        )

    backend = dub_config.source_separation.backend or "fake"

    if background_mode != "separated":
        checks.append(
            DoctorCheck(
                category="source-separation",
                name="source separation",
                status="skipped",
                message="Separated background mode is not configured.",
            )
        )
        return

    if backend == "fake":
        checks.append(
            DoctorCheck(
                category="source-separation",
                name="source separation",
                status="skipped",
                message="Fake source separation does not require external tools.",
            )
        )
        return

    if backend != "demucs":
        checks.append(
            DoctorCheck(
                category="source-separation",
                name="source separation",
                status="error",
                message=f"Unknown source separation backend: {backend}",
                hint='Use backend = "fake" or backend = "demucs".',
            )
        )
        return

    _check_executable(
        checks,
        category="source-separation",
        name="Demucs executable",
        executable=dub_config.source_separation.demucs_executable or "demucs",
        version_args=None,
        missing_hint='Install it with: pip install -e ".[source-separation]"',
    )


def _check_cache_paths(checks: list[DoctorCheck], loaded_config: LoadedConfig) -> None:
    workspace = resolve_config_path(
        loaded_config.config.dub.workspace_dir, loaded_config.base_dir
    ) or Path(".dublaro")
    _check_directory(
        checks,
        category="cache",
        name="Dublaro workspace",
        path=workspace,
    )
    _check_huggingface_cache(checks)


def _check_huggingface_cache(checks: list[DoctorCheck]) -> None:
    _check_directory(
        checks,
        category="cache",
        name="Hugging Face cache",
        path=_huggingface_cache_path(),
    )


def _huggingface_cache_path() -> Path:
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _check_file(
    checks: list[DoctorCheck],
    *,
    category: str,
    name: str,
    path: Path | None,
) -> None:
    if path is None:
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="error",
                message=f"{name} is not configured.",
            )
        )
        return

    if not path.exists():
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="error",
                message=f"{name} does not exist: {path}",
            )
        )
        return

    if not path.is_file():
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="error",
                message=f"{name} is not a file: {path}",
            )
        )
        return

    checks.append(
        DoctorCheck(
            category=category,
            name=name,
            status="ok",
            message=f"Found file: {path}",
        )
    )


def _check_directory(
    checks: list[DoctorCheck],
    *,
    category: str,
    name: str,
    path: Path,
) -> None:
    if path.exists() and not path.is_dir():
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="error",
                message=f"{name} exists but is not a directory: {path}",
            )
        )
        return

    if path.exists():
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="ok",
                message=f"Directory exists: {path}",
            )
        )
        return

    checks.append(
        DoctorCheck(
            category=category,
            name=name,
            status="warning",
            message=f"Directory does not exist yet: {path}",
            hint="It will be created when Dublaro needs it.",
        )
    )


def _check_executable(
    checks: list[DoctorCheck],
    *,
    category: str,
    name: str,
    executable: str,
    version_args: tuple[str, ...] | None,
    missing_hint: str,
) -> None:
    resolved = shutil.which(executable)
    if resolved is None:
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="error",
                message=f"{name} was not found: {executable}",
                hint=missing_hint,
            )
        )
        return

    if version_args is None:
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="ok",
                message=f"Executable found: {resolved}",
            )
        )
        return

    try:
        result = subprocess.run(
            [resolved, *version_args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="error",
                message=f"{name} could not be executed: {error}",
            )
        )
        return

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        checks.append(
            DoctorCheck(
                category=category,
                name=name,
                status="error",
                message=f"{name} failed to run: {details}",
            )
        )
        return

    version = _first_output_line(result.stdout or result.stderr)
    checks.append(
        DoctorCheck(
            category=category,
            name=name,
            status="ok",
            message=f"Executable works: {resolved}"
            + (f" ({version})" if version else ""),
        )
    )


def _get_argos_translation(
    translate_module: Any,
    source_language: str,
    target_language: str,
) -> Any | None:
    try:
        return translate_module.get_translation_from_codes(
            source_language,
            target_language,
        )
    except Exception:
        return None


def _first_output_line(output: str) -> str:
    return next((line.strip() for line in output.splitlines() if line.strip()), "")
