from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import console
from dublaro.cli.services.adapter_factories import create_source_separation_adapter
from dublaro.pipeline.separate import (
    default_source_separation_paths,
    separate_background_audio,
)


def separate_audio(
    input_audio: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input audio file.",
        ),
    ],
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Source separation backend: fake or demucs.",
        ),
    ] = "demucs",
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for generated background and voice WAV files.",
        ),
    ] = None,
    background_output: Annotated[
        Path | None,
        typer.Option(
            "--background-output",
            help="Output background/no-vocals WAV path.",
        ),
    ] = None,
    voice_output: Annotated[
        Path | None,
        typer.Option(
            "--voice-output",
            help="Output voice/vocals WAV path.",
        ),
    ] = None,
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            help="Output WAV sample rate.",
        ),
    ] = 24_000,
    demucs_executable: Annotated[
        str,
        typer.Option(
            "--demucs-executable",
            help="Demucs executable used when --backend demucs.",
        ),
    ] = "demucs",
    demucs_model: Annotated[
        str,
        typer.Option(
            "--demucs-model",
            help="Demucs model used when --backend demucs.",
        ),
    ] = "htdemucs",
    demucs_device: Annotated[
        str | None,
        typer.Option(
            "--demucs-device",
            help="Demucs device, for example cpu or cuda.",
        ),
    ] = None,
    ffmpeg_executable: Annotated[
        str,
        typer.Option(
            "--ffmpeg",
            help="FFmpeg executable used to normalize separated WAV files.",
        ),
    ] = "ffmpeg",
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace output files if they already exist.",
        ),
    ] = False,
) -> None:
    """Separate an audio file into background and voice stems."""
    if sample_rate <= 0:
        raise typer.BadParameter("--sample-rate must be greater than 0.")

    default_paths = default_source_separation_paths(input_audio, output_dir=output_dir)
    resolved_background_output = (
        background_output or default_paths.background_audio_path
    )
    resolved_voice_output = voice_output or default_paths.voice_audio_path

    adapter = create_source_separation_adapter(
        backend,
        demucs_executable=demucs_executable,
        demucs_model=demucs_model,
        demucs_device=demucs_device,
        ffmpeg_executable=ffmpeg_executable,
    )

    if backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake source separation adapter.")

    try:
        result = separate_background_audio(
            input_audio,
            adapter=adapter,
            background_output_path=resolved_background_output,
            voice_output_path=resolved_voice_output,
            sample_rate=sample_rate,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    console.print(
        f"[green]Separated background audio:[/green] {result.background_audio_path}"
    )

    if result.voice_audio_path is not None:
        console.print(
            f"[green]Separated voice audio:[/green] {result.voice_audio_path}"
        )
