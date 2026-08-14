from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
)
from dublaro.pipeline.mix import (
    default_mixed_audio_path,
    mix_original_audio_with_dubbed_speech,
)
from dublaro.pipeline.transcribe import (
    load_transcript,
)


def mix_audio(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input synthesized or fitted transcript JSON file.",
        ),
    ],
    original_audio_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Original video audio extracted as mono WAV.",
        ),
    ],
    speech_track_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Dubbed speech track WAV file.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output mixed audio WAV path.",
        ),
    ] = None,
    original_gain: Annotated[
        float,
        typer.Option(
            "--original-gain",
            help="Original audio volume multiplier outside dubbed speech.",
        ),
    ] = 1.0,
    ducking_gain: Annotated[
        float,
        typer.Option(
            "--ducking-gain",
            help="Original audio volume multiplier during dubbed speech.",
        ),
    ] = 0.25,
    speech_gain: Annotated[
        float,
        typer.Option(
            "--speech-gain",
            help="Dubbed speech volume multiplier.",
        ),
    ] = 1.0,
    ducking_margin_seconds: Annotated[
        float,
        typer.Option(
            "--ducking-margin",
            help="Extra ducking time before and after each speech segment.",
        ),
    ] = 0.05,
    ducking_fade_seconds: Annotated[
        float,
        typer.Option(
            "--ducking-fade",
            help="Fade time for ducking transitions.",
        ),
    ] = 0.05,
) -> None:
    """Mix dubbed speech over lowered original audio."""
    mixed_output = output_path or default_mixed_audio_path(speech_track_path)

    try:
        transcript = load_transcript(transcript_path)
        saved_path = mix_original_audio_with_dubbed_speech(
            transcript,
            original_audio_path=original_audio_path,
            speech_track_path=speech_track_path,
            output_path=mixed_output,
            original_gain=original_gain,
            ducking_gain=ducking_gain,
            speech_gain=speech_gain,
            ducking_margin_seconds=ducking_margin_seconds,
            ducking_fade_seconds=ducking_fade_seconds,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Mixed audio saved:[/green] {saved_path}")
