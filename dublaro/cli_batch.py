from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from dublaro.audio.ffmpeg import FFmpegError
from dublaro.cli_config import (
    DubCliOverrides,
    ResolvedDubSettings,
    resolve_dub_settings,
)
from dublaro.cli_dub_runner import (
    run_dub_preflight,
    run_resolved_dub,
    validate_resolved_dub_settings,
)
from dublaro.config import (
    DublaroConfigError,
    LoadedConfig,
    load_config,
    resolve_config_path,
)
from dublaro.pipeline.batch import (
    default_batch_output_dir,
    default_batch_workspace_dir,
    discover_batch_videos,
    format_video_extensions,
)
from dublaro.pipeline.dub import DubbingArtifacts, DubbingProgressCallback
from dublaro.pipeline.preflight import DubPreflightReport

BatchDubStatus = Literal["done", "planned", "failed"]


@dataclass(frozen=True)
class BatchDubJob:
    index: int
    total: int
    video_path: Path
    settings: ResolvedDubSettings


@dataclass(frozen=True)
class BatchDubResult:
    video_path: Path
    output_path: Path | None
    status: BatchDubStatus
    message: str


@dataclass(frozen=True)
class BatchDubRunResult:
    results: list[BatchDubResult]
    has_failures: bool


BatchDiscoveredCallback = Callable[[int], None]
BatchJobStartedCallback = Callable[[BatchDubJob], None]
PreflightReportCallback = Callable[[DubPreflightReport], None]
DubArtifactsCallback = Callable[[DubbingArtifacts], None]


def validate_batch_config(
    loaded_config: LoadedConfig,
    *,
    output_dir: Path | None,
    workspace_root: Path | None,
) -> None:
    config = loaded_config.config.dub

    if output_dir is None and config.output_path is not None:
        raise ValueError(
            "Batch cannot use dub.output_path because it is one exact file. "
            "Use --output-dir or dub.output_dir instead."
        )

    if workspace_root is None and config.workspace_dir is not None:
        raise ValueError(
            "Batch cannot use dub.workspace_dir because it is one exact directory. "
            "Use --workspace-root instead."
        )

    if config.srt.output_path is not None and config.srt.export is not False:
        raise ValueError(
            "Batch cannot use dub.srt.output_path because it is one exact file. "
            "Remove it and let Dublaro create per-video SRT paths."
        )

    if config.manifest.output_path is not None and config.manifest.write is not False:
        raise ValueError(
            "Batch cannot use dub.manifest.output_path because it is one exact file. "
            "Remove it and let Dublaro create per-video manifest paths."
        )


def run_batch_dubbing(
    *,
    input_path: Path,
    config_path: Path | None,
    target_language: str | None,
    output_dir: Path | None,
    source_language: str | None,
    workspace_root: Path | None,
    recursive: bool,
    dry_run: bool,
    continue_on_error: bool,
    resume_enabled: bool | None,
    preflight_enabled: bool | None,
    ffmpeg_executable: str | None,
    overwrite: bool | None,
    on_batch_discovered: BatchDiscoveredCallback | None = None,
    on_job_started: BatchJobStartedCallback | None = None,
    on_preflight_report: PreflightReportCallback | None = None,
    on_dub_progress: DubbingProgressCallback | None = None,
    on_dub_artifacts: DubArtifactsCallback | None = None,
) -> BatchDubRunResult:
    loaded_config = load_config(config_path)
    validate_batch_config(
        loaded_config,
        output_dir=output_dir,
        workspace_root=workspace_root,
    )

    videos = discover_batch_videos(input_path, recursive=recursive)
    if not videos:
        raise ValueError(
            "No supported video files found. Supported extensions: "
            f"{format_video_extensions()}"
        )

    output_root = output_dir or resolve_config_path(
        loaded_config.config.dub.output_dir,
        loaded_config.base_dir,
    )
    workspace_root_path = workspace_root or Path(".dublaro")

    if on_batch_discovered is not None:
        on_batch_discovered(len(videos))

    results: list[BatchDubResult] = []
    has_failures = False

    for index, video_path in enumerate(videos, start=1):
        resolved_output_path: Path | None = None

        try:
            job_output_dir = (
                default_batch_output_dir(input_path, video_path, output_root)
                if output_root is not None
                else None
            )

            settings = resolve_dub_settings(
                video_path=video_path,
                loaded_config=loaded_config,
                overrides=DubCliOverrides(
                    source_language=source_language,
                    target_language=target_language,
                    output_dir=job_output_dir,
                    workspace_dir=default_batch_workspace_dir(
                        input_path,
                        video_path,
                        workspace_root_path,
                    ),
                    resume=resume_enabled,
                    overwrite=overwrite,
                    preflight=preflight_enabled,
                    ffmpeg_executable=ffmpeg_executable,
                ),
            )
            resolved_output_path = settings.output_path

            parsed_srt_text_mode, parsed_subtitle_embed = (
                validate_resolved_dub_settings(settings)
            )

            if on_job_started is not None:
                on_job_started(
                    BatchDubJob(
                        index=index,
                        total=len(videos),
                        video_path=video_path,
                        settings=settings,
                    )
                )

            if settings.preflight:
                report = run_dub_preflight(video_path, settings)

                if on_preflight_report is not None:
                    on_preflight_report(report)

                if report.has_errors:
                    raise RuntimeError("Preflight failed.")

            if dry_run:
                results.append(
                    BatchDubResult(
                        video_path=video_path,
                        output_path=settings.output_path,
                        status="planned",
                        message="dry run",
                    )
                )
                continue

            artifacts = run_resolved_dub(
                video_path,
                settings,
                parsed_srt_text_mode=parsed_srt_text_mode,
                parsed_subtitle_embed=parsed_subtitle_embed,
                progress_callback=on_dub_progress,
            )

            if on_dub_artifacts is not None:
                on_dub_artifacts(artifacts)

            results.append(
                BatchDubResult(
                    video_path=video_path,
                    output_path=artifacts.dubbed_video_path,
                    status="done",
                    message="ok",
                )
            )
        except (
            DublaroConfigError,
            FFmpegError,
            FileExistsError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
            typer.BadParameter,
        ) as error:
            has_failures = True
            results.append(
                BatchDubResult(
                    video_path=video_path,
                    output_path=resolved_output_path,
                    status="failed",
                    message=str(error),
                )
            )

            if not continue_on_error and not dry_run:
                break

    return BatchDubRunResult(results=results, has_failures=has_failures)
