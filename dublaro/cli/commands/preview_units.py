from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.preview import (
    build_translation_units_preview,
)
from dublaro.cli.rendering import (
    console,
    print_translation_units_preview,
)


def preview_units(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input source transcript JSON file.",
        ),
    ],
    max_group_pause_seconds: Annotated[
        float,
        typer.Option(
            "--max-group-pause",
            help="Maximum pause between segments grouped for translation.",
        ),
    ] = 0.8,
    max_group_duration_seconds: Annotated[
        float,
        typer.Option(
            "--max-group-duration",
            help="Maximum duration for one grouped translation unit.",
        ),
    ] = 12.0,
    max_sentence_group_duration_seconds: Annotated[
        float,
        typer.Option(
            "--max-sentence-group-duration",
            help="Hard maximum duration for one unfinished sentence group.",
        ),
    ] = 24.0,
) -> None:
    """Preview how transcript segments will be grouped before translation."""
    try:
        preview = build_translation_units_preview(
            transcript_path,
            max_group_pause_seconds=max_group_pause_seconds,
            max_group_duration_seconds=max_group_duration_seconds,
            max_sentence_group_duration_seconds=max_sentence_group_duration_seconds,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_translation_units_preview(preview)
