from pathlib import Path
from typing import Annotated

import typer

from dublaro.audio.ffmpeg import FFmpegError, extract_audio_from_video
from dublaro.cli.rendering import console


def extract_audio(
    input_video: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input video or audio file.",
        ),
    ],
    output_audio: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output WAV path. Defaults to input filename with .wav extension.",
        ),
    ] = None,
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            help="Output audio sample rate.",
        ),
    ] = 16_000,
    channels: Annotated[
        int,
        typer.Option(
            "--channels",
            help="Number of output audio channels.",
        ),
    ] = 1,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace output file if it already exists.",
        ),
    ] = False,
) -> None:
    """Extract clean WAV audio from a video file."""
    try:
        output_path = extract_audio_from_video(
            input_video,
            output_audio,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
    except FFmpegError as error:
        console.print(f"[red]ffmpeg error:[/red] {error}")
        raise typer.Exit(code=1) from error
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Audio extracted:[/green] {output_path}")
