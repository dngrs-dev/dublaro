from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.preview import (
    build_timing_repair_preview_report,
)
from dublaro.cli.rendering import (
    console,
    print_timing_repair_preview_report,
)


def preview_repairs(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input timing-repaired transcript JSON file.",
        ),
    ],
    include_all: Annotated[
        bool,
        typer.Option(
            "--all/--attempted-only",
            help="Show every segment, not only segments attempted by timing repair.",
        ),
    ] = False,
) -> None:
    """Preview timing repair decisions from a repaired transcript."""
    try:
        report = build_timing_repair_preview_report(
            transcript_path,
            include_all=include_all,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_timing_repair_preview_report(report)
