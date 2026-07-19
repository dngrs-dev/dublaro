import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path


class FFmpegError(RuntimeError):
    """Raised when ffmpeg fails."""


class FFmpegNotFoundError(FFmpegError):
    """Raised when ffmpeg cannot be found."""


def find_ffmpeg(executable: str = "ffmpeg") -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FFmpegNotFoundError(
            "ffmpeg was not found. Install ffmpeg and make sure it is available "
            "in your PATH."
        )
    return resolved


def run_ffmpeg(
    args: Sequence[str | Path],
    *,
    executable: str = "ffmpeg",
) -> subprocess.CompletedProcess[str]:
    command = [find_ffmpeg(executable), *[str(arg) for arg in args]]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg failed without stderr output"
        raise FFmpegError(stderr)

    return result


def extract_audio_from_video(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")

    if output_path is None:
        output_file = input_file.with_name(f"{input_file.stem}.wav")
    else:
        output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            "Use overwrite=True to replace it."
        )

    overwrite_flag = "-y" if overwrite else "-n"

    run_ffmpeg(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            overwrite_flag,
            "-i",
            input_file,
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            output_file,
        ],
        executable=executable,
    )

    return output_file


def replace_video_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    video_file = Path(video_path)
    audio_file = Path(audio_path)
    output_file = Path(output_path)

    if not video_file.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_file}")

    if not video_file.is_file():
        raise ValueError(f"Video path is not a file: {video_file}")

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            "Use overwrite=True to replace it."
        )

    overwrite_flag = "-y" if overwrite else "-n"

    run_ffmpeg(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            overwrite_flag,
            "-i",
            video_file,
            "-i",
            audio_file,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_file,
        ],
        executable=executable,
    )

    return output_file
