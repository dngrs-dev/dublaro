from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.factories import (
    create_translation_adapter,
)
from dublaro.cli.rendering import (
    console,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
    save_transcript,
)
from dublaro.pipeline.translate import (
    default_translated_transcript_path,
    translate_transcript,
)


def translate(
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
    target_language: Annotated[
        str,
        typer.Option(
            "--to",
            help="Target language code, for example pl, uk, es.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output translated transcript JSON path.",
        ),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Override source language code.",
        ),
    ] = None,
    translation_backend: Annotated[
        str,
        typer.Option(
            "--translator",
            help="Translation backend: fake or argos.",
        ),
    ] = "fake",
    install_package: Annotated[
        bool,
        typer.Option(
            "--install-package",
            help="Download and install the Argos language package if missing.",
        ),
    ] = False,
    group_segments: Annotated[
        bool,
        typer.Option(
            "--group-segments/--no-group-segments",
            help="Translate nearby sentence fragments as one natural unit.",
        ),
    ] = True,
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
    """Translate transcript JSON into another language."""
    adapter = create_translation_adapter(
        translation_backend,
        auto_install=install_package,
    )
    translated_output = output_path or default_translated_transcript_path(
        transcript_path,
        target_language,
    )

    try:
        transcript = load_transcript(transcript_path)
        translated = translate_transcript(
            transcript,
            adapter=adapter,
            target_language=target_language,
            source_language=source_language,
            group_segments=group_segments,
            max_group_pause_seconds=max_group_pause_seconds,
            max_group_duration_seconds=max_group_duration_seconds,
            max_sentence_group_duration_seconds=max_sentence_group_duration_seconds,
        )
        saved_path = save_transcript(translated, translated_output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Translated transcript saved:[/green] {saved_path}")
    if translation_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake translation adapter.")
