from pathlib import Path
from typing import Annotated

import typer

from dublaro.cli.rendering import (
    console,
    print_batch_discovered,
    print_batch_job_started,
    print_batch_summary,
    print_dub_artifacts,
    print_dub_progress,
    print_preflight_report,
)
from dublaro.cli.services.batch import run_batch_dubbing
from dublaro.config import (
    DublaroConfigError,
)


def batch(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="Input video file or directory.",
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
    text_workflow: Annotated[
        str | None,
        typer.Option(
            "--text-workflow",
            help="Text workflow: translate-then-adapt or llm-dubbing.",
        ),
    ] = None,
    background_mode: Annotated[
        str | None,
        typer.Option(
            "--background-mode",
            help="Background audio mode: speech-only, original, ducked, or separated.",
        ),
    ] = None,
    source_separation_backend: Annotated[
        str | None,
        typer.Option(
            "--source-separation",
            help="Source separation backend used when --background-mode separated.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for output dubbed videos.",
        ),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Source language code.",
        ),
    ] = None,
    workspace_root: Annotated[
        Path | None,
        typer.Option(
            "--workspace-root",
            help="Root directory for per-video workspaces. Defaults to .dublaro.",
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-r",
            help="Search input directories recursively.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show planned batch jobs without running dubbing.",
        ),
    ] = False,
    continue_on_error: Annotated[
        bool,
        typer.Option(
            "--continue-on-error",
            help="Keep processing remaining videos after a failed job.",
        ),
    ] = False,
    resume_enabled: Annotated[
        bool | None,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse valid intermediate workspace artifacts.",
        ),
    ] = None,
    preflight_enabled: Annotated[
        bool | None,
        typer.Option(
            "--preflight/--no-preflight",
            help="Check tools and paths before each dubbing run.",
        ),
    ] = None,
    repair_timing_enabled: Annotated[
        bool | None,
        typer.Option(
            "--repair-timing/--no-repair-timing",
            help="Rewrite and resynthesize overlong segments before speech/video fitting.",
        ),
    ] = None,
    max_timing_repair_attempts: Annotated[
        int | None,
        typer.Option(
            "--timing-repair-attempts",
            help="Maximum rewrite attempts for each overlong segment.",
        ),
    ] = None,
    timing_repair_target_speedup: Annotated[
        float | None,
        typer.Option(
            "--timing-repair-target-speedup",
            help="Repair text when generated speech needs more than this speedup.",
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
    """Run the full dubbing pipeline for multiple videos."""
    try:
        batch_result = run_batch_dubbing(
            input_path=input_path,
            config_path=config_path,
            target_language=target_language,
            text_workflow=text_workflow,
            background_mode=background_mode,
            source_separation_backend=source_separation_backend,
            output_dir=output_dir,
            source_language=source_language,
            workspace_root=workspace_root,
            recursive=recursive,
            dry_run=dry_run,
            continue_on_error=continue_on_error,
            resume_enabled=resume_enabled,
            preflight_enabled=preflight_enabled,
            repair_timing_enabled=repair_timing_enabled,
            max_timing_repair_attempts=max_timing_repair_attempts,
            timing_repair_target_speedup=timing_repair_target_speedup,
            ffmpeg_executable=ffmpeg_executable,
            overwrite=overwrite,
            on_batch_discovered=print_batch_discovered,
            on_job_started=print_batch_job_started,
            on_preflight_report=print_preflight_report,
            on_dub_progress=print_dub_progress,
            on_dub_artifacts=print_dub_artifacts,
        )
    except (DublaroConfigError, ValueError) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_batch_summary(batch_result.results)

    if batch_result.has_failures:
        raise typer.Exit(code=1)
