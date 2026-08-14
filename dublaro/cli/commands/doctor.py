from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.doctor import build_doctor_report
from dublaro.cli.rendering import (
    print_doctor_report,
)


def doctor(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to dublaro.toml config file. Defaults to ./dublaro.toml if present.",
        ),
    ] = None,
    ffmpeg_executable: Annotated[
        str | None,
        typer.Option(
            "--ffmpeg",
            help="ffmpeg executable name or path.",
        ),
    ] = None,
    piper_executable: Annotated[
        str | None,
        typer.Option(
            "--piper-executable",
            help="Piper executable name or path.",
        ),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Source language code used to verify Argos packages.",
        ),
    ] = None,
    target_language: Annotated[
        str | None,
        typer.Option(
            "--to",
            help="Target language code used to verify Argos packages.",
        ),
    ] = None,
) -> None:
    """Check local tools, configured voices, tokens, packages, and cache paths."""
    report = build_doctor_report(
        config_path=config_path,
        ffmpeg_executable=ffmpeg_executable,
        piper_executable=piper_executable,
        source_language=source_language,
        target_language=target_language,
    )

    print_doctor_report(report)

    if report.has_errors:
        raise typer.Exit(code=1)
