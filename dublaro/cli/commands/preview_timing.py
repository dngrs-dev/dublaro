from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.preview import (
    build_timing_preview_report,
)
from dublaro.cli.rendering import (
    console,
    print_timing_preview_report,
)


def preview_timing(
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
    max_speedup: Annotated[
        float,
        typer.Option(
            "--max-speedup",
            help="Maximum speech speed-up before video fitting is needed.",
        ),
    ] = 1.35,
    min_overrun_seconds: Annotated[
        float,
        typer.Option(
            "--min-overrun",
            help="Ignore audio overruns at or below this duration.",
        ),
    ] = 0.05,
    only_issues: Annotated[
        bool,
        typer.Option(
            "--only-issues/--all",
            help="Show only segments that need attention.",
        ),
    ] = False,
) -> None:
    """Preview segment timing before speech or video fitting."""
    try:
        report = build_timing_preview_report(
            transcript_path,
            max_speedup=max_speedup,
            min_overrun_seconds=min_overrun_seconds,
            only_issues=only_issues,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_timing_preview_report(report)
