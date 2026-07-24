from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from dublaro import __version__
from dublaro.adapters.asr import AsrAdapter, FakeAsrAdapter, TranscriptionOptions
from dublaro.adapters.text_adapter import (
    FakeTextAdapter,
    RuleBasedTextAdapter,
    TextAdapter,
)
from dublaro.adapters.translation import (
    ArgosTranslationAdapter,
    FakeTranslationAdapter,
    TranslationAdapter,
)
from dublaro.adapters.tts import FakeTtsAdapter, PiperTtsAdapter, TtsAdapter
from dublaro.audio.ffmpeg import (
    FFmpegError,
    extract_audio_from_video,
)
from dublaro.cli_config import DubCliOverrides, resolve_dub_settings
from dublaro.config import DublaroConfigError, load_config
from dublaro.pipeline.adapt_text import (
    adapt_transcript_text,
    default_adapted_transcript_path,
)
from dublaro.pipeline.align import (
    build_speech_timeline,
    default_speech_timeline_path,
)
from dublaro.pipeline.dub import (
    DubbingProgressStatus,
    DubbingProgressStep,
    dub_video,
)
from dublaro.pipeline.export import (
    default_dubbed_video_path,
    export_dubbed_video,
)
from dublaro.pipeline.fit_speech import (
    default_fitted_speech_output_dir,
    default_fitted_transcript_path,
    fit_generated_speech_to_segments,
)
from dublaro.pipeline.mix import (
    default_mixed_audio_path,
    mix_original_audio_with_dubbed_speech,
)
from dublaro.pipeline.preflight import DubPreflightReport, validate_dub_preflight
from dublaro.pipeline.subtitles import SrtTextMode, default_srt_path, save_srt
from dublaro.pipeline.synthesize import (
    default_speech_output_dir,
    default_synthesized_transcript_path,
    synthesize_transcript_speech,
)
from dublaro.pipeline.timing import analyze_speech_timing
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
from dublaro.pipeline.units import group_segments_for_translation

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

    if backend == "rules":
        return RuleBasedTextAdapter()

    raise typer.BadParameter("Text adapter must be 'fake' or 'rules'.")


def create_tts_adapter(
    backend: str,
    *,
    piper_model_path: Path | None = None,
    piper_config_path: Path | None = None,
    piper_executable: str = "piper",
    piper_speaker: int | None = None,
) -> TtsAdapter:
    if backend == "fake":
        return FakeTtsAdapter()

    if backend == "piper":
        if piper_model_path is None:
            raise typer.BadParameter("--piper-model is required when --tts piper.")

        return PiperTtsAdapter(
            piper_model_path,
            config_path=piper_config_path,
            executable=piper_executable,
            speaker=piper_speaker,
        )

    raise typer.BadParameter("TTS backend must be 'fake' or 'piper'.")


def parse_srt_text_mode(text_mode: str) -> SrtTextMode:
    allowed_modes = {"auto", "source", "translated", "adapted"}

    if text_mode not in allowed_modes:
        raise typer.BadParameter(
            "SRT text must be one of: auto, source, translated, adapted."
        )

    return cast(SrtTextMode, text_mode)


def print_preflight_report(report: DubPreflightReport) -> None:
    if not report.issues:
        console.print("[green]Preflight ok.[/green]")
        return

    if report.has_errors:
        console.print("[red]Preflight failed.[/red]")
    else:
        console.print("[yellow]Preflight warnings.[/yellow]")

    table = Table(title="Preflight")
    table.add_column("Severity")
    table.add_column("Code")
    table.add_column("Message", overflow="fold", ratio=3)
    table.add_column("Hint", overflow="fold", ratio=2)

    for issue in report.issues:
        style = "red" if issue.severity == "error" else "yellow"
        table.add_row(
            f"[{style}]{issue.severity}[/{style}]",
            issue.code,
            issue.message,
            issue.hint or "",
        )

    console.print(table)


def print_dub_progress(
    step: DubbingProgressStep,
    status: DubbingProgressStatus,
    message: str,
) -> None:
    if status == "started":
        console.print(f"[cyan]Starting:[/cyan] {message}")
        return

    if status == "failed":
        console.print(f"[red]Failed:[/red] {message}")
        return

    if status == "skipped":
        console.print(f"[yellow]Skipping:[/yellow] {message}")
        return


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
        )
        saved_path = save_transcript(translated, translated_output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Translated transcript saved:[/green] {saved_path}")
    if translation_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake translation adapter.")


@app.command("preview-units")
def preview_units(
    transcript_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input source transcript JSON file.",
        ),
    ],
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
) -> None:
    """Preview how transcript segments will be grouped before translation."""
    try:
        transcript = load_transcript(transcript_path)
        groups = group_segments_for_translation(
            transcript,
            max_pause_seconds=max_group_pause_seconds,
            max_duration_seconds=max_group_duration_seconds,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(
        "[green]Translation units:[/green] "
        f"{len(groups)} from {len(transcript.segments)} segments"
    )

    table = Table(title="Translation Unit Preview")
    table.add_column("Unit")
    table.add_column("Segments")
    table.add_column("Window")
    table.add_column("Duration")
    table.add_column("Speaker")
    table.add_column("Source text", overflow="fold", ratio=3)

    for group in groups:
        table.add_row(
            group.id,
            ", ".join(segment.id for segment in group.segments),
            f"{group.start:.2f}-{group.end:.2f}s",
            f"{group.duration:.2f}s",
            group.speaker or "",
            Text(group.source_text),
        )

    console.print(table)


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
            help="Text adaptation backend: fake or rules.",
        ),
    ] = "rules",
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


@app.command("export-srt")
def export_srt(
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
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output SRT subtitle path.",
        ),
    ] = None,
    text_mode: Annotated[
        str,
        typer.Option(
            "--text",
            help="Subtitle text: auto, source, translated, or adapted.",
        ),
    ] = "auto",
) -> None:
    """Export transcript JSON as SRT subtitles."""
    srt_output = output_path or default_srt_path(transcript_path)

    try:
        transcript = load_transcript(transcript_path)
        saved_path = save_srt(
            transcript,
            srt_output,
            text_mode=parse_srt_text_mode(text_mode),
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]SRT saved:[/green] {saved_path}")


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


@app.command("check-timing")
def check_timing(
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
    max_overrun_seconds: Annotated[
        float,
        typer.Option(
            "--max-overrun",
            help="Allowed seconds beyond the original segment duration.",
        ),
    ] = 0.15,
    max_ratio: Annotated[
        float,
        typer.Option(
            "--max-ratio",
            help="Allowed generated/original duration ratio.",
        ),
    ] = 1.10,
    fail_on_issues: Annotated[
        bool,
        typer.Option(
            "--fail-on-issues/--no-fail-on-issues",
            help="Exit with code 1 when timing issues are found.",
        ),
    ] = False,
) -> None:
    """Check whether generated speech clips fit transcript segment timing."""
    try:
        transcript = load_transcript(transcript_path)
        issues = analyze_speech_timing(
            transcript,
            max_overrun_seconds=max_overrun_seconds,
            max_ratio=max_ratio,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    if not issues:
        console.print("[green]Timing ok:[/green] generated speech fits segments.")
        return

    table = Table(title="Speech Timing Issues")
    table.add_column("Segment")
    table.add_column("Window")
    table.add_column("Target")
    table.add_column("Audio")
    table.add_column("Overrun")
    table.add_column("Ratio")

    for issue in issues:
        table.add_row(
            issue.segment_id,
            f"{issue.start:.2f}-{issue.end:.2f}s",
            f"{issue.target_duration:.2f}s",
            f"{issue.audio_duration:.2f}s",
            f"{issue.overrun_seconds:.2f}s",
            f"{issue.ratio:.2f}x",
        )

    console.print(table)

    if fail_on_issues:
        raise typer.Exit(code=1)


@app.command("fit-speech")
def fit_speech(
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
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for fitted speech clips.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output fitted transcript JSON path.",
        ),
    ] = None,
    max_speedup: Annotated[
        float,
        typer.Option(
            "--max-speedup",
            help="Maximum allowed audio speedup factor.",
        ),
    ] = 1.35,
    min_overrun_seconds: Annotated[
        float,
        typer.Option(
            "--min-overrun",
            help="Only fit clips longer than this tolerance.",
        ),
    ] = 0.05,
    ffmpeg_executable: Annotated[
        str,
        typer.Option(
            "--ffmpeg",
            help="ffmpeg executable name or path.",
        ),
    ] = "ffmpeg",
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace existing fitted audio files.",
        ),
    ] = False,
) -> None:
    """Speed up overlong speech clips so they fit segment timing."""
    fitted_dir = output_dir or default_fitted_speech_output_dir(transcript_path)
    fitted_output = output_path or default_fitted_transcript_path(transcript_path)

    try:
        transcript = load_transcript(transcript_path)
        fitted = fit_generated_speech_to_segments(
            transcript,
            output_dir=fitted_dir,
            max_speedup=max_speedup,
            min_overrun_seconds=min_overrun_seconds,
            overwrite=overwrite,
            executable=ffmpeg_executable,
        )
        saved_path = save_transcript(fitted, fitted_output)
    except (FFmpegError, FileExistsError, FileNotFoundError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Fitted speech clips:[/green] {fitted_dir}")
    console.print(f"[green]Fitted transcript saved:[/green] {saved_path}")
    console.print(
        "[green]Segments adjusted:[/green] "
        f"{fitted.metadata['speech_fitting_fitted_segments']}"
    )


@app.command("mix-audio")
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


@app.command("dub")
def dub(
    video_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input video file.",
        ),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to dublaro.toml config file.",
        ),
    ] = None,
    target_language: Annotated[
        str | None,
        typer.Option(
            "--to",
            help="Target language code.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output dubbed video path.",
        ),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Source language code.",
        ),
    ] = None,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Directory for intermediate artifacts.",
        ),
    ] = None,
    resume_enabled: Annotated[
        bool | None,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse valid intermediate workspace artifacts.",
        ),
    ] = None,
    asr_backend: Annotated[
        str | None,
        typer.Option(
            "--asr",
            help="ASR backend: fake or faster-whisper.",
        ),
    ] = None,
    translation_backend: Annotated[
        str | None,
        typer.Option(
            "--translator",
            help="Translation backend: fake or argos.",
        ),
    ] = None,
    text_adapter_backend: Annotated[
        str | None,
        typer.Option(
            "--text-adapter",
            help="Text adaptation backend: fake or rules.",
        ),
    ] = None,
    tts_backend: Annotated[
        str | None,
        typer.Option(
            "--tts",
            help="TTS backend: fake.",
        ),
    ] = None,
    model_size: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="faster-whisper model size.",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Inference device: cpu or cuda.",
        ),
    ] = None,
    compute_type: Annotated[
        str | None,
        typer.Option(
            "--compute-type",
            help="faster-whisper compute type.",
        ),
    ] = None,
    install_package: Annotated[
        bool | None,
        typer.Option(
            "--install-package/--no-install-package",
            help="Download and install translation package if missing.",
        ),
    ] = None,
    translation_group_segments: Annotated[
        bool | None,
        typer.Option(
            "--group-segments/--no-group-segments",
            help="Translate nearby sentence fragments as one natural unit.",
        ),
    ] = None,
    max_translation_group_pause_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-group-pause",
            help="Maximum pause between segments grouped for translation.",
        ),
    ] = None,
    max_translation_group_duration_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-group-duration",
            help="Maximum duration for one grouped translation unit.",
        ),
    ] = None,
    asr_sample_rate: Annotated[
        int | None,
        typer.Option(
            "--asr-sample-rate",
            help="Audio sample rate used for ASR.",
        ),
    ] = None,
    speech_sample_rate: Annotated[
        int | None,
        typer.Option(
            "--speech-sample-rate",
            help="Generated speech sample rate.",
        ),
    ] = None,
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
        str | None,
        typer.Option(
            "--piper-executable",
            help="Piper executable name or path.",
        ),
    ] = None,
    piper_speaker: Annotated[
        int | None,
        typer.Option(
            "--piper-speaker",
            help="Piper speaker id for multi-speaker voices.",
        ),
    ] = None,
    fit_speech_enabled: Annotated[
        bool | None,
        typer.Option(
            "--fit-speech/--no-fit-speech",
            help="Speed up overlong generated speech clips before alignment.",
        ),
    ] = None,
    max_speech_speedup: Annotated[
        float | None,
        typer.Option(
            "--max-speech-speedup",
            help="Maximum allowed audio speedup factor when fitting speech.",
        ),
    ] = None,
    min_speech_overrun_seconds: Annotated[
        float | None,
        typer.Option(
            "--min-speech-overrun",
            help="Only fit clips longer than this tolerance.",
        ),
    ] = None,
    mix_original_audio_enabled: Annotated[
        bool | None,
        typer.Option(
            "--mix-original-audio/--no-mix-original-audio",
            help="Mix dubbed speech over lowered original audio.",
        ),
    ] = None,
    original_audio_gain: Annotated[
        float | None,
        typer.Option(
            "--original-audio-gain",
            help="Original audio volume multiplier outside dubbed speech.",
        ),
    ] = None,
    ducking_gain: Annotated[
        float | None,
        typer.Option(
            "--ducking-gain",
            help="Original audio volume multiplier during dubbed speech.",
        ),
    ] = None,
    speech_gain: Annotated[
        float | None,
        typer.Option(
            "--speech-gain",
            help="Dubbed speech volume multiplier.",
        ),
    ] = None,
    ducking_margin_seconds: Annotated[
        float | None,
        typer.Option(
            "--ducking-margin",
            help="Extra ducking time before and after each speech segment.",
        ),
    ] = None,
    ducking_fade_seconds: Annotated[
        float | None,
        typer.Option(
            "--ducking-fade",
            help="Fade time for ducking transitions.",
        ),
    ] = None,
    export_srt_enabled: Annotated[
        bool | None,
        typer.Option(
            "--export-srt/--no-export-srt",
            help="Save an external SRT subtitle file for the final spoken text.",
        ),
    ] = None,
    srt_output_path: Annotated[
        Path | None,
        typer.Option(
            "--srt-output",
            help="Output SRT path. Defaults to output video path with .srt extension.",
        ),
    ] = None,
    srt_text_mode: Annotated[
        str | None,
        typer.Option(
            "--srt-text",
            help="SRT text: auto, source, translated, or adapted.",
        ),
    ] = None,
    write_manifest_enabled: Annotated[
        bool | None,
        typer.Option(
            "--manifest/--no-manifest",
            help="Save a JSON manifest describing this dubbing run.",
        ),
    ] = None,
    manifest_output_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest-output",
            help="Output manifest path. Defaults to the workspace manifest path.",
        ),
    ] = None,
    preflight_enabled: Annotated[
        bool | None,
        typer.Option(
            "--preflight/--no-preflight",
            help="Check tools and paths before starting the dubbing run.",
        ),
    ] = None,
    ffmpeg_executable: Annotated[
        str | None,
        typer.Option(
            "--ffmpeg",
            help="ffmpeg executable name or path.",
        ),
    ] = None,
    overwrite: Annotated[
        bool | None,
        typer.Option(
            "--overwrite/--no-overwrite",
            help="Replace existing intermediate and output files.",
        ),
    ] = None,
) -> None:
    """Run the full dubbing pipeline."""
    try:
        loaded_config = load_config(config_path)
        settings = resolve_dub_settings(
            video_path=video_path,
            loaded_config=loaded_config,
            overrides=DubCliOverrides(
                source_language=source_language,
                target_language=target_language,
                output_path=output_path,
                workspace_dir=workspace_dir,
                resume=resume_enabled,
                overwrite=overwrite,
                preflight=preflight_enabled,
                ffmpeg_executable=ffmpeg_executable,
                asr_sample_rate=asr_sample_rate,
                speech_sample_rate=speech_sample_rate,
                asr_backend=asr_backend,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                translation_backend=translation_backend,
                install_package=install_package,
                translation_group_segments=translation_group_segments,
                max_translation_group_pause_seconds=max_translation_group_pause_seconds,
                max_translation_group_duration_seconds=max_translation_group_duration_seconds,
                text_adapter_backend=text_adapter_backend,
                tts_backend=tts_backend,
                piper_model_path=piper_model_path,
                piper_config_path=piper_config_path,
                piper_executable=piper_executable,
                piper_speaker=piper_speaker,
                fit_speech=fit_speech_enabled,
                max_speech_speedup=max_speech_speedup,
                min_speech_overrun_seconds=min_speech_overrun_seconds,
                mix_original_audio=mix_original_audio_enabled,
                original_audio_gain=original_audio_gain,
                ducking_gain=ducking_gain,
                speech_gain=speech_gain,
                ducking_margin_seconds=ducking_margin_seconds,
                ducking_fade_seconds=ducking_fade_seconds,
                export_srt=export_srt_enabled,
                srt_output_path=srt_output_path,
                srt_text_mode=srt_text_mode,
                write_manifest=write_manifest_enabled,
                manifest_output_path=manifest_output_path,
            ),
        )
        parsed_srt_text_mode = parse_srt_text_mode(settings.srt_text_mode)

        if settings.manifest_output_path is not None and not settings.write_manifest:
            raise ValueError(
                "--manifest-output cannot be used when manifest writing is disabled."
            )

        if settings.resume and settings.overwrite:
            raise ValueError("--resume cannot be used with --overwrite.")
    except (DublaroConfigError, ValueError, typer.BadParameter) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    if settings.preflight:
        report = validate_dub_preflight(
            video_path=video_path,
            output_path=settings.output_path,
            workspace_dir=settings.workspace_dir,
            overwrite=settings.overwrite,
            ffmpeg_executable=settings.ffmpeg_executable,
            asr_backend=settings.asr_backend,
            translation_backend=settings.translation_backend,
            source_language=settings.source_language,
            target_language=settings.target_language,
            install_translation_package=settings.install_package,
            tts_backend=settings.tts_backend,
            piper_model_path=settings.piper_model_path,
            piper_config_path=settings.piper_config_path,
            piper_executable=settings.piper_executable,
            export_srt=settings.export_srt,
            srt_output_path=settings.srt_output_path,
            write_manifest=settings.write_manifest,
            manifest_output_path=settings.manifest_output_path,
            resume=settings.resume,
        )
        print_preflight_report(report)

        if report.has_errors:
            raise typer.Exit(code=1)

    try:
        artifacts = dub_video(
            video_path,
            settings.output_path,
            source_language=settings.source_language,
            target_language=settings.target_language,
            workspace_dir=settings.workspace_dir,
            asr_adapter=create_asr_adapter(
                settings.asr_backend,
                model_size=settings.model_size,
                device=settings.device,
                compute_type=settings.compute_type,
            ),
            translation_adapter=create_translation_adapter(
                settings.translation_backend,
                auto_install=settings.install_package,
            ),
            text_adapter=create_text_adapter(settings.text_adapter_backend),
            tts_adapter=create_tts_adapter(
                settings.tts_backend,
                piper_model_path=settings.piper_model_path,
                piper_config_path=settings.piper_config_path,
                piper_executable=settings.piper_executable,
                piper_speaker=settings.piper_speaker,
            ),
            translation_group_segments=settings.translation_group_segments,
            max_translation_group_pause_seconds=settings.max_translation_group_pause_seconds,
            max_translation_group_duration_seconds=settings.max_translation_group_duration_seconds,
            asr_sample_rate=settings.asr_sample_rate,
            speech_sample_rate=settings.speech_sample_rate,
            fit_speech=settings.fit_speech,
            max_speech_speedup=settings.max_speech_speedup,
            min_speech_overrun_seconds=settings.min_speech_overrun_seconds,
            mix_original_audio=settings.mix_original_audio,
            original_audio_gain=settings.original_audio_gain,
            ducking_gain=settings.ducking_gain,
            speech_gain=settings.speech_gain,
            ducking_margin_seconds=settings.ducking_margin_seconds,
            ducking_fade_seconds=settings.ducking_fade_seconds,
            export_srt=settings.export_srt,
            srt_output_path=settings.srt_output_path,
            srt_text_mode=parsed_srt_text_mode,
            progress_callback=print_dub_progress,
            write_manifest=settings.write_manifest,
            manifest_output_path=settings.manifest_output_path,
            ffmpeg_executable=settings.ffmpeg_executable,
            resume=settings.resume,
            overwrite=settings.overwrite,
        )
    except (
        FFmpegError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]Dubbed video saved:[/green] {artifacts.dubbed_video_path}")
    console.print(f"[green]Workspace:[/green] {artifacts.workspace_dir}")
    if artifacts.fitted_transcript_path is not None:
        console.print(
            f"[green]Fitted transcript:[/green] {artifacts.fitted_transcript_path}"
        )
    if artifacts.mixed_audio_path is not None:
        console.print(f"[green]Mixed audio:[/green] {artifacts.mixed_audio_path}")
    if artifacts.srt_path is not None:
        console.print(f"[green]SRT subtitles:[/green] {artifacts.srt_path}")
    if artifacts.manifest_path is not None:
        console.print(f"[green]Manifest:[/green] {artifacts.manifest_path}")

    if settings.asr_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake ASR adapter.")
    if settings.translation_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake translation adapter.")
    if settings.text_adapter_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake text adapter.")
    if settings.tts_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake TTS adapter.")
