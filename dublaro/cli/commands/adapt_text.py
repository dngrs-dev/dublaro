from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
)
from dublaro.cli.services.adapter_factories import (
    create_text_adapter,
)
from dublaro.pipeline.adapt_text import (
    adapt_transcript_text,
    default_adapted_transcript_path,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
    save_transcript,
)


def adapt_text(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input translated transcript JSON file.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output adapted transcript JSON path.",
        ),
    ] = None,
    target_language: Annotated[
        str | None,
        typer.Option(
            "--to",
            help="Override target language code.",
        ),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Override source language code.",
        ),
    ] = None,
    text_adapter_backend: Annotated[
        str,
        typer.Option(
            "--text-adapter",
            help="Text adaptation backend: fake, rules, or ollama.",
        ),
    ] = "rules",
    ollama_model: Annotated[
        str | None,
        typer.Option(
            "--ollama-model",
            help="Ollama model used when --text-adapter ollama.",
        ),
    ] = None,
    ollama_url: Annotated[
        str | None,
        typer.Option(
            "--ollama-url",
            help="Ollama server URL used when --text-adapter ollama.",
        ),
    ] = None,
    ollama_timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--ollama-timeout",
            help="Ollama request timeout in seconds.",
        ),
    ] = None,
    ollama_temperature: Annotated[
        float | None,
        typer.Option(
            "--ollama-temperature",
            help="Ollama generation temperature.",
        ),
    ] = None,
    max_chars_per_second: Annotated[
        float,
        typer.Option(
            "--max-chars-per-second",
            help="Target maximum spoken text density.",
        ),
    ] = 16.0,
    preserve_meaning: Annotated[
        bool,
        typer.Option(
            "--preserve-meaning/--allow-trimming",
            help="Keep meaningful text even when it exceeds timing budget.",
        ),
    ] = True,
) -> None:
    """Adapt translated transcript text for dubbing."""
    adapted_output = output_path or default_adapted_transcript_path(transcript_path)

    try:
        adapter = create_text_adapter(
            text_adapter_backend,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
            ollama_timeout_seconds=ollama_timeout_seconds,
            ollama_temperature=ollama_temperature,
        )
        transcript = load_transcript(transcript_path)
        adapted = adapt_transcript_text(
            transcript,
            adapter=adapter,
            target_language=target_language,
            source_language=source_language,
            max_chars_per_second=max_chars_per_second,
            preserve_meaning=preserve_meaning,
        )
        saved_path = save_transcript(adapted, adapted_output)
    except (FileNotFoundError, ValueError, typer.BadParameter) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Adapted transcript saved:[/green] {saved_path}")
    if text_adapter_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake text adapter.")
