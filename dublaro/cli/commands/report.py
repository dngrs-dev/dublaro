from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
    print_dub_quality_report,
)
from dublaro.cli.reports.quality import build_dub_quality_report


def report(
    workspace_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Dublaro workspace directory, for example .dublaro/zoo.",
        ),
    ],
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Manifest path. Defaults to latest *.manifest.json in workspace.",
        ),
    ] = None,
    max_speedup: Annotated[
        float | None,
        typer.Option(
            "--max-speedup",
            help="Override max speech speed-up used for timing diagnostics.",
        ),
    ] = None,
    min_overrun_seconds: Annotated[
        float | None,
        typer.Option(
            "--min-overrun",
            help="Override ignored timing overrun duration in seconds.",
        ),
    ] = None,
    fail_on_issues: Annotated[
        bool,
        typer.Option(
            "--fail-on-issues/--no-fail-on-issues",
            help="Exit with code 1 when report finds missing artifacts or quality issues.",
        ),
    ] = False,
) -> None:
    """Summarize quality, timing, repairs, and artifacts for a dub workspace."""
    try:
        quality_report = build_dub_quality_report(
            workspace_dir,
            manifest_path=manifest_path,
            max_speedup=max_speedup,
            min_overrun_seconds=min_overrun_seconds,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_dub_quality_report(quality_report)

    if fail_on_issues and quality_report.has_issues:
        raise typer.Exit(code=1)
