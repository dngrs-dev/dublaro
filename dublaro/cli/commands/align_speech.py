from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
)
from dublaro.pipeline.align import (
    build_speech_timeline,
    default_speech_timeline_path,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
)


def align_speech(
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
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output speech track WAV path.",
        ),
    ] = None,
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            help="Expected generated speech sample rate.",
        ),
    ] = 24_000,
    duration: Annotated[
        float | None,
        typer.Option(
            "--duration",
            help="Override output track duration in seconds.",
        ),
    ] = None,
) -> None:
    """Build one timed speech track from generated segment clips."""
    speech_track_output = output_path or default_speech_timeline_path(transcript_path)

    try:
        transcript = load_transcript(transcript_path)
        saved_path = build_speech_timeline(
            transcript,
            output_path=speech_track_output,
            sample_rate=sample_rate,
            duration=duration,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Speech track saved:[/green] {saved_path}")
