from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
    print_speaker_preview,
)
from dublaro.cli.reports.preview import (
    build_speaker_preview,
)
from dublaro.config import (
    DublaroConfigError,
)


def preview_speakers(
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
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to dublaro.toml config file.",
        ),
    ] = None,
) -> None:
    """Preview detected speakers and configured voice routing."""
    try:
        preview = build_speaker_preview(
            transcript_path,
            config_path=config_path,
        )
    except (DublaroConfigError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_speaker_preview(preview)
