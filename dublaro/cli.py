from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dublaro import __version__
from dublaro.adapters.asr import AsrAdapter, FakeAsrAdapter, TranscriptionOptions
from dublaro.adapters.text_adapter import FakeTextAdapter, TextAdapter
from dublaro.adapters.translation import (
    ArgosTranslationAdapter,
    FakeTranslationAdapter,
    TranslationAdapter,
)
from dublaro.audio.ffmpeg import (
    FFmpegError,
    extract_audio_from_video,
)
from dublaro.pipeline.adapt_text import (
    adapt_transcript_text,
    default_adapted_transcript_path,
)
from dublaro.pipeline.transcribe import (
    default_transcript_path,
    load_transcript,
    save_transcript,
    transcribe_audio,
)
from dublaro.pipeline.translate import (
    default_translated_transcript_path,
    translate_transcript,
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


def create_translation_adapter(
    backend: str,
    *,
    auto_install: bool = False,
) -> TranslationAdapter:
    if backend == "fake":
        return FakeTranslationAdapter()

    if backend == "argos":
        return ArgosTranslationAdapter(auto_install=auto_install)

    raise typer.BadParameter("Translation backend must be 'fake' or 'argos'.")


def create_text_adapter(backend: str) -> TextAdapter:
    if backend == "fake":
        return FakeTextAdapter()

    raise typer.BadParameter("Text adapter must be 'fake'.")


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


@app.command("translate")
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
        )
        saved_path = save_transcript(translated, translated_output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Translated transcript saved:[/green] {saved_path}")
    if translation_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake translation adapter.")


@app.command("adapt-text")
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
            help="Text adaptation backend: fake.",
        ),
    ] = "fake",
    max_chars_per_second: Annotated[
        float,
        typer.Option(
            "--max-chars-per-second",
            help="Target maximum spoken text density.",
        ),
    ] = 16.0,
) -> None:
    """Adapt translated transcript text for dubbing."""
    adapter = create_text_adapter(text_adapter_backend)
    adapted_output = output_path or default_adapted_transcript_path(transcript_path)

    try:
        transcript = load_transcript(transcript_path)
        adapted = adapt_transcript_text(
            transcript,
            adapter=adapter,
            target_language=target_language,
            source_language=source_language,
            max_chars_per_second=max_chars_per_second,
        )
        saved_path = save_transcript(adapted, adapted_output)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Adapted transcript saved:[/green] {saved_path}")
    if text_adapter_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake text adapter.")
