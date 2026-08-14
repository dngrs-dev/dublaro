from pathlib import Path
from typing import Annotated

import typer

from dublaro.audio.ffmpeg import (
    FFmpegError,
)
from dublaro.cli.rendering import (
    console,
)
from dublaro.pipeline.fit_speech import (
    default_fitted_speech_output_dir,
    default_fitted_transcript_path,
    fit_generated_speech_to_segments,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
    save_transcript,
)


def fit_speech(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input synthesized transcript JSON file.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for fitted speech clips.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output fitted transcript JSON path.",
        ),
    ] = None,
    max_speedup: Annotated[
        float,
        typer.Option(
            "--max-speedup",
            help="Maximum allowed audio speedup factor.",
        ),
    ] = 1.35,
    min_overrun_seconds: Annotated[
        float,
        typer.Option(
            "--min-overrun",
            help="Only fit clips longer than this tolerance.",
        ),
    ] = 0.05,
    ffmpeg_executable: Annotated[
        str,
        typer.Option(
            "--ffmpeg",
            help="ffmpeg executable name or path.",
        ),
    ] = "ffmpeg",
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace existing fitted audio files.",
        ),
    ] = False,
) -> None:
    """Speed up overlong speech clips so they fit segment timing."""
    fitted_dir = output_dir or default_fitted_speech_output_dir(transcript_path)
    fitted_output = output_path or default_fitted_transcript_path(transcript_path)

    try:
        transcript = load_transcript(transcript_path)
        fitted = fit_generated_speech_to_segments(
            transcript,
            output_dir=fitted_dir,
            max_speedup=max_speedup,
            min_overrun_seconds=min_overrun_seconds,
            overwrite=overwrite,
            executable=ffmpeg_executable,
        )
        saved_path = save_transcript(fitted, fitted_output)
    except (FFmpegError, FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Fitted speech clips:[/green] {fitted_dir}")
    console.print(f"[green]Fitted transcript saved:[/green] {saved_path}")
    console.print(
        "[green]Segments adjusted:[/green] "
        f"{fitted.metadata['speech_fitting_fitted_segments']}"
    )
