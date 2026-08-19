from pathlib import Path
from typing import Annotated

import typer

from dublaro.audio.ffmpeg import FFmpegError, normalize_audio_loudness
from dublaro.cli.rendering import console


def normalize_audio(
    input_audio: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input audio file.",
        ),
    ],
    output_audio: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output normalized WAV path.",
        ),
    ] = None,
    target_lufs: Annotated[
        float,
        typer.Option(
            "--target-lufs",
            help="Target integrated loudness in LUFS.",
        ),
    ] = -16.0,
    true_peak: Annotated[
        float,
        typer.Option(
            "--true-peak",
            help="Target true peak in dBTP.",
        ),
    ] = -1.5,
    loudness_range: Annotated[
        float,
        typer.Option(
            "--loudness-range",
            help="Target loudness range.",
        ),
    ] = 11.0,
    sample_rate: Annotated[
        int | None,
        typer.Option(
            "--sample-rate",
            help="Output sample rate.",
        ),
    ] = None,
    channels: Annotated[
        int | None,
        typer.Option(
            "--channels",
            help="Output channel count.",
        ),
    ] = None,
    ffmpeg_executable: Annotated[
        str,
        typer.Option(
            "--ffmpeg",
            help="FFmpeg executable.",
        ),
    ] = "ffmpeg",
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace output file if it already exists.",
        ),
    ] = False,
) -> None:
    """Normalize audio loudness with FFmpeg loudnorm."""
    try:
        output_path = normalize_audio_loudness(
            input_audio,
            output_audio,
            target_lufs=target_lufs,
            true_peak=true_peak,
            loudness_range=loudness_range,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
            executable=ffmpeg_executable,
        )
    except (FFmpegError, FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Normalized audio saved:[/green] {output_path}")
