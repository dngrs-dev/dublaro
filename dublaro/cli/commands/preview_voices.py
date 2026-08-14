from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.preview import (
    build_voice_samples_preview,
)
from dublaro.cli.rendering import (
    console,
    print_voice_samples_preview,
)
from dublaro.config import (
    DublaroConfigError,
)


def preview_voices(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to dublaro TOML config.",
        ),
    ] = None,
    text: Annotated[
        str,
        typer.Option(
            "--text",
            help="Text to synthesize for each configured speaker voice.",
        ),
    ] = "Hello",
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where generated voice samples will be saved.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            help="Preview language. Defaults to dub.target_language from config.",
        ),
    ] = None,
    sample_rate: Annotated[
        int | None,
        typer.Option(
            "--sample-rate",
            help="Preview sample rate. Defaults to dub speech sample rate.",
        ),
    ] = None,
) -> None:
    """Generate short TTS samples for configured speaker voices."""
    try:
        preview = build_voice_samples_preview(
            config_path=config_path,
            text=text,
            output_dir=output_dir,
            language=language,
            sample_rate=sample_rate,
        )
    except (DublaroConfigError, FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_voice_samples_preview(preview)
