from pathlib import Path
from typing import Annotated

import typer

from dublaro.audio.ffmpeg import (
    FFmpegError,
)
from dublaro.cli.rendering import (
    console,
)
from dublaro.pipeline.export import (
    default_dubbed_video_path,
    export_dubbed_video,
)


def export_video(
    video_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Original input video file.",
        ),
    ],
    speech_track_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Dubbed speech track WAV file.",
        ),
    ],
    target_language: Annotated[
        str,
        typer.Option(
            "--to",
            help="Target language code used for default output naming.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output dubbed video path.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace output file if it already exists.",
        ),
    ] = False,
) -> None:
    """Replace a video's audio with the dubbed speech track."""
    dubbed_output = output_path or default_dubbed_video_path(
        video_path,
        target_language,
    )

    try:
        saved_path = export_dubbed_video(
            video_path,
            speech_track_path,
            dubbed_output,
            overwrite=overwrite,
        )
    except (FFmpegError, FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Dubbed video saved:[/green] {saved_path}")
