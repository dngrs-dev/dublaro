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
from dublaro.adapters.tts import FakeTtsAdapter, TtsAdapter
from dublaro.audio.ffmpeg import (
    FFmpegError,
    extract_audio_from_video,
)
from dublaro.pipeline.adapt_text import (
    adapt_transcript_text,
    default_adapted_transcript_path,
)
from dublaro.pipeline.align import (
    build_speech_timeline,
    default_speech_timeline_path,
)
from dublaro.pipeline.export import (
    default_dubbed_video_path,
    export_dubbed_video,
)
from dublaro.pipeline.synthesize import (
    default_speech_output_dir,
    default_synthesized_transcript_path,
    synthesize_transcript_speech,
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


def create_tts_adapter(backend: str) -> TtsAdapter:
    if backend == "fake":
        return FakeTtsAdapter()

    raise typer.BadParameter("TTS backend must be 'fake'.")


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


@app.command("synthesize")
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
) -> None:
    """Generate speech audio files from transcript segments."""
    adapter = create_tts_adapter(tts_backend)
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


@app.command("align-speech")
def align_speech(
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
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output speech track WAV path.",
        ),
    ] = None,
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            help="Expected generated speech sample rate.",
        ),
    ] = 24_000,
    duration: Annotated[
        float | None,
        typer.Option(
            "--duration",
            help="Override output track duration in seconds.",
        ),
    ] = None,
) -> None:
    """Build one timed speech track from generated segment clips."""
    speech_track_output = output_path or default_speech_timeline_path(transcript_path)

    try:
        transcript = load_transcript(transcript_path)
        saved_path = build_speech_timeline(
            transcript,
            output_path=speech_track_output,
            sample_rate=sample_rate,
            duration=duration,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Speech track saved:[/green] {saved_path}")


@app.command("export-video")
def export_video(
    video_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Original input video file.",
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
    target_language: Annotated[
        str,
        typer.Option(
            "--to",
            help="Target language code used for default output naming.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output dubbed video path.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace output file if it already exists.",
        ),
    ] = False,
) -> None:
    """Replace a video's audio with the dubbed speech track."""
    dubbed_output = output_path or default_dubbed_video_path(
        video_path,
        target_language,
    )

    try:
        saved_path = export_dubbed_video(
            video_path,
            speech_track_path,
            dubbed_output,
            overwrite=overwrite,
        )
    except (FFmpegError, FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Dubbed video saved:[/green] {saved_path}")
