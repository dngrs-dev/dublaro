from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.dub_runner import (
    parse_srt_text_mode,
)
from dublaro.cli.rendering import (
    console,
)
from dublaro.pipeline.subtitles import (
    default_srt_path,
    save_srt,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
)


def export_srt(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input transcript JSON file.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output SRT subtitle path.",
        ),
    ] = None,
    text_mode: Annotated[
        str,
        typer.Option(
            "--text",
            help="Subtitle text: auto, source, translated, or adapted.",
        ),
    ] = "auto",
) -> None:
    """Export transcript JSON as SRT subtitles."""
    srt_output = output_path or default_srt_path(transcript_path)

    try:
        transcript = load_transcript(transcript_path)
        saved_path = save_srt(
            transcript,
            srt_output,
            text_mode=parse_srt_text_mode(text_mode),
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]SRT saved:[/green] {saved_path}")
