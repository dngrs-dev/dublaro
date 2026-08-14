from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from dublaro.cli.rendering import (
    console,
)
from dublaro.pipeline.timing import (
    analyze_speech_timing,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
)


def check_timing(
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
    max_overrun_seconds: Annotated[
        float,
        typer.Option(
            "--max-overrun",
            help="Allowed seconds beyond the original segment duration.",
        ),
    ] = 0.15,
    max_ratio: Annotated[
        float,
        typer.Option(
            "--max-ratio",
            help="Allowed generated/original duration ratio.",
        ),
    ] = 1.10,
    fail_on_issues: Annotated[
        bool,
        typer.Option(
            "--fail-on-issues/--no-fail-on-issues",
            help="Exit with code 1 when timing issues are found.",
        ),
    ] = False,
) -> None:
    """Check whether generated speech clips fit transcript segment timing."""
    try:
        transcript = load_transcript(transcript_path)
        issues = analyze_speech_timing(
            transcript,
            max_overrun_seconds=max_overrun_seconds,
            max_ratio=max_ratio,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    if not issues:
        console.print("[green]Timing ok:[/green] generated speech fits segments.")
        return

    table = Table(title="Speech Timing Issues")
    table.add_column("Segment")
    table.add_column("Window")
    table.add_column("Target")
    table.add_column("Audio")
    table.add_column("Overrun")
    table.add_column("Ratio")

    for issue in issues:
        table.add_row(
            issue.segment_id,
            f"{issue.start:.2f}-{issue.end:.2f}s",
            f"{issue.target_duration:.2f}s",
            f"{issue.audio_duration:.2f}s",
            f"{issue.overrun_seconds:.2f}s",
            f"{issue.ratio:.2f}x",
        )

    console.print(table)

    if fail_on_issues:
        raise typer.Exit(code=1)
