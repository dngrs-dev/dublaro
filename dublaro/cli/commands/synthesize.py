from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
)
from dublaro.cli.services.adapter_factories import (
    create_tts_adapter,
)
from dublaro.pipeline.synthesize import (
    default_speech_output_dir,
    default_synthesized_transcript_path,
    synthesize_transcript_speech,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
    save_transcript,
)


def synthesize(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input adapted transcript JSON file.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for generated speech segment audio files.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output synthesized transcript JSON path.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Override speech synthesis language code.",
        ),
    ] = None,
    tts_backend: Annotated[
        str,
        typer.Option(
            "--tts",
            help="TTS backend: fake.",
        ),
    ] = "fake",
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            help="Generated audio sample rate.",
        ),
    ] = 24_000,
    piper_model_path: Annotated[
        Path | None,
        typer.Option(
            "--piper-model",
            help="Path to Piper .onnx voice model.",
        ),
    ] = None,
    piper_config_path: Annotated[
        Path | None,
        typer.Option(
            "--piper-config",
            help="Path to Piper .onnx.json voice config.",
        ),
    ] = None,
    piper_executable: Annotated[
        str,
        typer.Option(
            "--piper-executable",
            help="Piper executable name or path.",
        ),
    ] = "piper",
    piper_speaker: Annotated[
        int | None,
        typer.Option(
            "--piper-speaker",
            help="Piper speaker id for multi-speaker voices.",
        ),
    ] = None,
) -> None:
    """Generate speech audio files from transcript segments."""
    adapter = create_tts_adapter(
        tts_backend,
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        piper_executable=piper_executable,
        piper_speaker=piper_speaker,
    )
    speech_output_dir = output_dir or default_speech_output_dir(transcript_path)
    synthesized_output = output_path or default_synthesized_transcript_path(
        transcript_path
    )

    try:
        transcript = load_transcript(transcript_path)
        synthesized = synthesize_transcript_speech(
            transcript,
            adapter=adapter,
            output_dir=speech_output_dir,
            language=language,
            sample_rate=sample_rate,
        )
        saved_path = save_transcript(synthesized, synthesized_output)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Speech clips saved:[/green] {speech_output_dir}")
    console.print(f"[green]Synthesized transcript saved:[/green] {saved_path}")
    if tts_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake TTS adapter.")
