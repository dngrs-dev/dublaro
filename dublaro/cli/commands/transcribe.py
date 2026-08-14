from pathlib import Path
from typing import Annotated

import typer

from dublaro.adapters.asr import TranscriptionOptions
from dublaro.cli.rendering import (
    console,
)
from dublaro.cli.services.adapter_factories import (
    create_asr_adapter,
)
from dublaro.pipeline.transcribe import (
    default_transcript_path,
    save_transcript,
    transcribe_audio,
)


def transcribe(
    audio_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input audio file.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output transcript JSON path.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Source language code, for example en, de, fr.",
        ),
    ] = None,
    asr_backend: Annotated[
        str,
        typer.Option(
            "--asr",
            help="ASR backend: fake or faster-whisper.",
        ),
    ] = "fake",
    model_size: Annotated[
        str,
        typer.Option(
            "--model",
            help="faster-whisper model size, for example tiny, base, small, medium.",
        ),
    ] = "small",
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Inference device: cpu or cuda.",
        ),
    ] = "cpu",
    compute_type: Annotated[
        str,
        typer.Option(
            "--compute-type",
            help="faster-whisper compute type, for example int8 or float16.",
        ),
    ] = "int8",
) -> None:
    """Transcribe an audio file into transcript JSON."""
    adapter = create_asr_adapter(
        asr_backend,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    transcript_output = output_path or default_transcript_path(audio_path)

    try:
        transcript = transcribe_audio(
            audio_path,
            adapter=adapter,
            options=TranscriptionOptions(source_language=language),
        )
        saved_path = save_transcript(transcript, transcript_output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Transcript saved:[/green] {saved_path}")
    if asr_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake ASR adapter.")
