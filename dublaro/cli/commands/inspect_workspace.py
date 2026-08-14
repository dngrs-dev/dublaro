from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
    print_workspace_inspection_report,
)
from dublaro.cli.workspace import inspect_workspace as inspect_workspace_artifacts


def inspect_workspace(
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
    include_manifest: Annotated[
        bool,
        typer.Option(
            "--manifest/--no-manifest",
            help="Include artifact paths referenced by workspace manifest files.",
        ),
    ] = True,
    include_unknown: Annotated[
        bool,
        typer.Option(
            "--all/--known-only",
            help="Include files and directories not recognized as Dublaro artifacts.",
        ),
    ] = False,
    fail_on_missing: Annotated[
        bool,
        typer.Option(
            "--fail-on-missing/--no-fail-on-missing",
            help="Exit with code 1 when manifest-referenced artifacts are missing.",
        ),
    ] = False,
) -> None:
    """Inspect generated artifacts in a Dublaro workspace."""
    try:
        report = inspect_workspace_artifacts(
            workspace_dir,
            include_manifest=include_manifest,
            include_unknown=include_unknown,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_workspace_inspection_report(report)

    if fail_on_missing and report.missing_count:
        raise typer.Exit(code=1)
