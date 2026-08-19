import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from dublaro.adapters.source_separation.base import (
    SourceSeparationOptions,
    SourceSeparationResult,
)


class DemucsSourceSeparationAdapter:
    name = "demucs"

    def __init__(
        self,
        *,
        executable: str = "demucs",
        model: str = "htdemucs",
        device: str | None = None,
        ffmpeg_executable: str = "ffmpeg",
    ) -> None:
        self.executable = executable
        self.model = model
        self.device = device
        self.ffmpeg_executable = ffmpeg_executable

    def separate_sources(
        self,
        audio_path: str | Path,
        *,
        background_output_path: str | Path,
        voice_output_path: str | Path,
        options: SourceSeparationOptions,
        overwrite: bool = False,
    ) -> SourceSeparationResult:
        source = Path(audio_path)
        background = Path(background_output_path)
        voice = Path(voice_output_path)

        if not source.exists():
            raise FileNotFoundError(f"Audio does not exist: {source}")

        _ensure_output_path(background, overwrite=overwrite)
        _ensure_output_path(voice, overwrite=overwrite)

        with TemporaryDirectory(prefix="dublaro-demucs-") as temp_dir:
            output_root = Path(temp_dir)

            command = [
                self.executable,
                "--two-stems",
                "vocals",
                "-n",
                self.model,
                "--out",
                str(output_root),
                str(source),
            ]

            if self.device is not None:
                command[1:1] = ["--device", self.device]

            _run_command(command, tool_name="Demucs")

            stem_dir = _find_demucs_stem_dir(
                output_root,
                audio_path=source,
                model=self.model,
            )

            no_vocals_path = stem_dir / "no_vocals.wav"
            vocals_path = stem_dir / "vocals.wav"

            if not no_vocals_path.exists():
                raise FileNotFoundError(
                    f"Demucs did not produce background stem: {no_vocals_path}"
                )

            if not vocals_path.exists():
                raise FileNotFoundError(
                    f"Demucs did not produce voice stem: {vocals_path}"
                )

            _convert_wav(
                no_vocals_path,
                background,
                sample_rate=options.sample_rate,
                overwrite=overwrite,
                executable=self.ffmpeg_executable,
            )
            _convert_wav(
                vocals_path,
                voice,
                sample_rate=options.sample_rate,
                overwrite=overwrite,
                executable=self.ffmpeg_executable,
            )

        return SourceSeparationResult(
            background_audio_path=background,
            voice_audio_path=voice,
        )


def _ensure_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)


def _find_demucs_stem_dir(
    output_root: Path,
    *,
    audio_path: Path,
    model: str,
) -> Path:
    expected = output_root / model / audio_path.stem

    if expected.exists():
        return expected

    matches = sorted(output_root.rglob("vocals.wav"))

    if len(matches) == 1:
        return matches[0].parent

    raise FileNotFoundError(
        f"Demucs did not produce expected stems under: {output_root}"
    )


def _convert_wav(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    overwrite: bool,
    executable: str,
) -> None:
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    _run_command(command, tool_name="FFmpeg")


def _run_command(command: list[str], *, tool_name: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{tool_name} executable was not found: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        message = f"{tool_name} failed"

        if details:
            message = f"{message}: {details}"

        raise RuntimeError(message) from error
