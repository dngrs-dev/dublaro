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
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [find_ffmpeg(executable), *[str(arg) for arg in args]]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
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


def replace_video_audio_with_soft_subtitles(
    video_path: str | Path,
    audio_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path,
    *,
    subtitle_language: str | None = None,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    video_file, audio_file, output_file = _validate_video_audio_output(
        video_path, audio_path, output_path, overwrite
    )
    subtitle_file = _check_subtitle_file(subtitle_path)
    overwrite_flag = "-y" if overwrite else "-n"

    args: list[str | Path] = [
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite_flag,
        "-i",
        video_file,
        "-i",
        audio_file,
        "-i",
        subtitle_file,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-c:s",
        _soft_subtitle_codec(output_file),
        "-shortest",
    ]

    if subtitle_language:
        args.extend(["-metadata:s:s:0", f"language={subtitle_language}"])

    args.append(output_file)
    run_ffmpeg(args, executable=executable)
    return output_file


def replace_video_audio_with_hard_subtitles(
    video_path: str | Path,
    audio_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    video_file, audio_file, output_file = _validate_video_audio_output(
        video_path, audio_path, output_path, overwrite
    )
    subtitle_file = _check_subtitle_file(subtitle_path)
    subtitle_dir = subtitle_file.parent.resolve()
    overwrite_flag = "-y" if overwrite else "-n"

    run_ffmpeg(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            overwrite_flag,
            "-i",
            video_file.resolve(),
            "-i",
            audio_file.resolve(),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            _subtitles_filter_argument(subtitle_file.name),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            output_file.resolve(),
        ],
        executable=executable,
        cwd=subtitle_dir,
    )
    return output_file


def _validate_video_audio_output(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
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

    return video_file, audio_file, output_file


def _check_subtitle_file(path: str | Path) -> Path:
    subtitle_file = Path(path)

    if not subtitle_file.exists():
        raise FileNotFoundError(f"Subtitle file does not exist: {subtitle_file}")
    if not subtitle_file.is_file():
        raise ValueError(f"Subtitle path is not a file: {subtitle_file}")

    return subtitle_file


def _soft_subtitle_codec(output_path: Path) -> str:
    if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        return "mov_text"
    return "srt"


def _subtitles_filter_argument(filename: str) -> str:
    return f"subtitles=filename={_escape_filter_value(filename)}"


def _escape_filter_value(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(";", "\\;")
    )
    return f"'{escaped}'"


def replace_video_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    video_file, audio_file, output_file = _validate_video_audio_output(
        video_path, audio_path, output_path, overwrite
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


def slow_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    slowdown_factor: float,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video file does not exist: {input_file}")

    if not input_file.is_file():
        raise ValueError(f"Input video path is not a file: {input_file}")

    if slowdown_factor < 1.0:
        raise ValueError("slowdown_factor must be >= 1.0")

    if input_file.resolve() == output_file.resolve():
        raise ValueError("Cannot slow video in place.")

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
            "-map",
            "0:v:0",
            "-filter:v",
            f"setpts={slowdown_factor:.6g}*PTS",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_file,
        ],
        executable=executable,
    )

    return output_file


def change_audio_tempo(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tempo_factor: float,
    sample_rate: int | None = None,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {input_file}")

    if not input_file.is_file():
        raise ValueError(f"Input audio path is not a file: {input_file}")

    if tempo_factor <= 0:
        raise ValueError("tempo_factor must be > 0")

    if input_file.resolve() == output_file.resolve():
        raise ValueError("Cannot change audio tempo in place.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            "Use overwrite=True to replace it."
        )

    overwrite_flag = "-y" if overwrite else "-n"

    args: list[str | Path] = [
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite_flag,
        "-i",
        input_file,
        "-filter:a",
        build_atempo_filter(tempo_factor),
        "-ac",
        "1",
    ]

    if sample_rate is not None:
        args.extend(["-ar", str(sample_rate)])

    args.extend(["-acodec", "pcm_s16le", output_file])

    run_ffmpeg(args, executable=executable)

    return output_file


def build_atempo_filter(tempo_factor: float) -> str:
    if tempo_factor <= 0:
        raise ValueError("tempo_factor must be > 0")

    factors: list[float] = []
    remaining = tempo_factor

    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    factors.append(remaining)

    return ",".join(f"atempo={factor:.6g}" for factor in factors)
