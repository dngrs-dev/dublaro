from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dublaro import __version__
from dublaro.adapters.asr import AsrAdapter, FakeAsrAdapter, TranscriptionOptions
from dublaro.audio.ffmpeg import (
    FFmpegError,
    extract_audio_from_video,
)
from dublaro.pipeline.transcribe import (
    default_transcript_path,
    save_transcript,
    transcribe_audio,
)

app = typer.Typer(
    name="dublaro",
    help="Open-source AI dubbing tools.",
    no_args_is_help=True,
)

console = Console()


def version_callback(show_version: bool) -> None:
    if show_version:
        console.print(f"dublaro {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            help="Show Dublaro version.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    pass


def create_asr_adapter(
    backend: str,
    *,
    model_size: str,
    device: str,
    compute_type: str,
) -> AsrAdapter:
    if backend == "fake":
        return FakeAsrAdapter()

    if backend == "faster-whisper":
        from dublaro.adapters.asr.faster_whisper import FastWhisperAsrAdapter

        return FastWhisperAsrAdapter(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )

    raise typer.BadParameter("ASR backend must be 'fake' or 'faster-whisper'.")


@app.command("extract-audio")
def extract_audio(
    input_video: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input video or audio file.",
        ),
    ],
    output_audio: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output WAV path. Defaults to input filename with .wav extension.",
        ),
    ] = None,
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            help="Output audio sample rate.",
        ),
    ] = 16_000,
    channels: Annotated[
        int,
        typer.Option(
            "--channels",
            help="Number of output audio channels.",
        ),
    ] = 1,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace output file if it already exists.",
        ),
    ] = False,
) -> None:
    """Extract clean WAV audio from a video file."""
    try:
        output_path = extract_audio_from_video(
            input_video,
            output_audio,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=overwrite,
        )
    except FFmpegError as error:
        console.print(f"[red]ffmpeg error:[/red] {error}")
        raise typer.Exit(code=1) from error
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Audio extracted:[/green] {output_path}")


@app.command("transcribe")
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
