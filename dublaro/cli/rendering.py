from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from dublaro.cli.batch import BatchDubJob, BatchDubResult
from dublaro.cli.doctor import DoctorReport
from dublaro.cli.preview import (
    SpeakerPreview,
    TimingPreviewReport,
    TimingRepairPreviewReport,
    TranslationUnitsPreview,
    VoiceSamplesPreview,
    format_optional_factor,
    format_optional_seconds,
)
from dublaro.cli.workspace import WorkspaceInspectionReport
from dublaro.cli_config import ResolvedDubSettings
from dublaro.pipeline.dub import (
    DubbingArtifacts,
    DubbingProgressStatus,
    DubbingProgressStep,
)
from dublaro.pipeline.preflight import DubPreflightReport

console = Console()


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


def print_doctor_report(report: DoctorReport) -> None:
    if report.has_errors:
        console.print("[red]Doctor found problems.[/red]")
    elif report.has_warnings:
        console.print("[yellow]Doctor warnings.[/yellow]")
    else:
        console.print("[green]Doctor ok.[/green]")

    table = Table(title="Doctor")
    table.add_column("Status", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Check", no_wrap=True)
    table.add_column("Message", overflow="fold", ratio=3)
    table.add_column("Hint", overflow="fold", ratio=2)

    for check in report.checks:
        style = {
            "ok": "green",
            "warning": "yellow",
            "error": "red",
            "skipped": "dim",
        }[check.status]

        table.add_row(
            f"[{style}]{check.status}[/{style}]",
            check.category,
            check.name,
            check.message,
            check.hint or "",
        )

    console.print(table)


def print_workspace_inspection_report(report: WorkspaceInspectionReport) -> None:
    console.print(f"[green]Workspace:[/green] {report.workspace_dir}")
    console.print(
        "[green]Artifacts:[/green] "
        f"{report.present_count} present, {report.missing_count} missing, "
        f"{len(report.manifest_paths)} manifest file(s)"
    )

    if not report.artifacts:
        console.print("[yellow]No known workspace artifacts found.[/yellow]")
        return

    table = Table(title="Workspace Artifacts")
    table.add_column("Status", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Artifact", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Details", no_wrap=True)
    table.add_column("Path", overflow="fold", ratio=4)

    for artifact in report.artifacts:
        style = "green" if artifact.status == "present" else "yellow"
        table.add_row(
            f"[{style}]{artifact.status}[/{style}]",
            artifact.category,
            artifact.label,
            artifact.source,
            artifact.details or "",
            _format_workspace_artifact_path(report.workspace_dir, artifact.path),
        )

    console.print(table)


def _format_workspace_artifact_path(workspace_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(workspace_dir))
    except ValueError:
        return str(path)


def _optional_artifact_path(artifacts: object, name: str) -> Path | None:
    value = getattr(artifacts, name, None)
    return value if isinstance(value, Path) else None


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


def print_dub_artifacts(artifacts: DubbingArtifacts) -> None:
    console.print(f"[green]Dubbed video saved:[/green] {artifacts.dubbed_video_path}")
    console.print(f"[green]Workspace:[/green] {artifacts.workspace_dir}")

    timing_repaired_transcript_path = _optional_artifact_path(
        artifacts,
        "timing_repaired_transcript_path",
    )
    if timing_repaired_transcript_path is not None:
        console.print(
            "[green]Timing-repaired transcript:[/green] "
            f"{timing_repaired_transcript_path}"
        )

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


def print_adapter_notes(settings: ResolvedDubSettings) -> None:
    if settings.asr_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake ASR adapter.")
    if settings.diarize and settings.diarization_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake diarization adapter.")
    if settings.translation_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake translation adapter.")
    if settings.text_adapter_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake text adapter.")
    if settings.tts_backend == "fake":
        console.print("[yellow]Note:[/yellow] using fake TTS adapter.")


def print_batch_discovered(video_count: int) -> None:
    console.print(f"[green]Batch:[/green] {video_count} video(s) found.")


def print_batch_job_started(job: BatchDubJob) -> None:
    console.print()
    console.print(f"[bold]Batch {job.index}/{job.total}:[/bold] {job.video_path}")
    console.print(f"[green]Output:[/green] {job.settings.output_path}")
    console.print(f"[green]Workspace:[/green] {job.settings.workspace_dir}")


def print_batch_summary(results: list[BatchDubResult]) -> None:
    completed_count = sum(result.status == "done" for result in results)
    planned_count = sum(result.status == "planned" for result in results)
    failed_count = sum(result.status == "failed" for result in results)

    console.print(
        "[bold]Batch summary:[/bold] "
        f"{completed_count} done, {planned_count} planned, {failed_count} failed"
    )

    table = Table(title="Batch")
    table.add_column("Status")
    table.add_column("Input", overflow="fold", ratio=2)
    table.add_column("Output", overflow="fold", ratio=2)
    table.add_column("Message", overflow="fold", ratio=2)

    for result in results:
        style = {
            "done": "green",
            "planned": "yellow",
            "failed": "red",
        }.get(result.status, "white")

        table.add_row(
            f"[{style}]{result.status}[/{style}]",
            str(result.video_path),
            str(result.output_path or ""),
            result.message,
        )

    console.print(table)


def print_translation_units_preview(preview: TranslationUnitsPreview) -> None:
    console.print(
        "[green]Translation units:[/green] "
        f"{len(preview.groups)} from {preview.segment_count} segments"
    )

    table = Table(title="Translation Unit Preview")
    table.add_column("Unit")
    table.add_column("Segments")
    table.add_column("Window")
    table.add_column("Duration")
    table.add_column("Speaker")
    table.add_column("Source text", overflow="fold", ratio=3)

    for group in preview.groups:
        table.add_row(
            group.id,
            ", ".join(segment.id for segment in group.segments),
            f"{group.start:.2f}-{group.end:.2f}s",
            f"{group.duration:.2f}s",
            group.speaker or "",
            Text(group.source_text),
        )

    console.print(table)


def print_speaker_preview(preview: SpeakerPreview) -> None:
    console.print(
        "[green]Speakers:[/green] "
        f"{len(preview.rows)} from {preview.segment_count} segments"
    )

    table = Table(title="Speaker Preview")
    table.add_column("Speaker", no_wrap=True)
    table.add_column("Segments", justify="right", no_wrap=True)
    table.add_column("Speaking Time", justify="right", no_wrap=True)
    table.add_column("Window", no_wrap=True)
    table.add_column("Voice Route", overflow="fold", ratio=4)

    for row in preview.rows:
        table.add_row(
            row.speaker_id,
            str(row.segment_count),
            f"{row.total_duration_seconds:.2f}s",
            row.window,
            Text(row.voice_route),
        )

    console.print(table)

    if preview.configured_speaker_count and preview.unconfigured_speakers:
        console.print(
            "[yellow]Warning:[/yellow] No configured voice profile for detected "
            f"speakers: {', '.join(preview.unconfigured_speakers)}. "
            "They will use fallback TTS."
        )

    if preview.unused_voice_profiles:
        console.print(
            "[yellow]Warning:[/yellow] Configured voice profiles not present in "
            f"transcript: {', '.join(preview.unused_voice_profiles)}."
        )


def print_voice_samples_preview(preview: VoiceSamplesPreview) -> None:
    console.print(
        f"[green]Voice samples:[/green] {len(preview.samples)} saved to {preview.output_dir}"
    )

    table = Table(title="Voice Sample Preview")
    table.add_column("Speaker", no_wrap=True)
    table.add_column("Name")
    table.add_column("Backend")
    table.add_column("Output")

    for sample in preview.samples:
        table.add_row(
            sample.speaker_id,
            sample.display_name or "",
            sample.tts_backend,
            str(sample.output_path),
        )

    console.print(table)


def print_timing_preview_report(report: TimingPreviewReport) -> None:
    console.print(
        "[green]Timing preview:[/green] "
        f"{len(report.previews)} segments, "
        f"{report.attention_count} need attention, "
        f"{report.video_fit_count} need video fitting"
    )

    if not report.shown_previews:
        console.print("[green]No timing issues to show.[/green]")
        return

    table = Table(title="Timing Preview")
    table.add_column("Segment", no_wrap=True)
    table.add_column("Target", justify="right", no_wrap=True)
    table.add_column("Audio", justify="right", no_wrap=True)
    table.add_column("Overrun", justify="right", no_wrap=True)
    table.add_column("Req", justify="right", no_wrap=True)
    table.add_column("Video", no_wrap=True)
    table.add_column("Status", no_wrap=True)

    for preview in report.shown_previews:
        table.add_row(
            preview.segment_id,
            format_optional_seconds(preview.target_duration),
            format_optional_seconds(preview.audio_duration),
            format_optional_seconds(preview.overrun_seconds),
            format_optional_factor(preview.required_speedup),
            "yes" if preview.needs_video_fit else "no",
            preview.status,
        )

    console.print(table)


def print_timing_repair_preview_report(
    report: TimingRepairPreviewReport,
) -> None:
    console.print(
        "[green]Timing repair preview:[/green] "
        f"{report.attempted_count} attempted from {report.total_segments} segments, "
        f"{report.repaired_count} repaired, "
        f"{report.improved_count} improved, "
        f"{report.not_improved_count} not improved"
    )

    repair_mode = report.metadata.get("timing_repair")
    repair_adapter = report.metadata.get("timing_repair_adapter")
    if repair_mode or repair_adapter:
        console.print(
            "[green]Repair:[/green] "
            f"{repair_mode or 'unknown'}"
            f"{f' with {repair_adapter}' if repair_adapter else ''}"
        )

    if not report.rows:
        console.print("[yellow]No timing repair metadata found.[/yellow]")
        return

    for row in report.rows:
        style = {
            "repaired": "green",
            "improved": "yellow",
            "not_improved": "red",
            "not_attempted": "dim",
        }.get(row.status, "white")

        console.print(
            f"[{style}]{row.segment_id}[/{style}] "
            f"{row.status} | "
            f"attempts={_format_repair_optional_int(row.attempts)} | "
            f"audio={_format_repair_change(row.audio_duration_before_seconds, row.audio_duration_after_seconds, format_optional_seconds)} | "
            f"speedup={_format_repair_change(row.required_speedup_before, row.required_speedup_after, format_optional_factor)} | "
            f"reason={row.reason}"
        )

        details = [
            f"window={row.start:.2f}-{row.end:.2f}s",
            f"speaker={row.speaker or ''}",
            f"target={format_optional_seconds(row.target_duration_seconds) or '-'}",
            f"max={format_optional_seconds(row.max_audio_duration_seconds) or '-'}",
        ]
        console.print(f"  [dim]{' | '.join(details)}[/dim]")

        if row.text:
            console.print(Text(f"  {row.text}"))


def _format_repair_optional_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _format_repair_change(
    before: float | None,
    after: float | None,
    formatter: object,
) -> str:
    if not callable(formatter):
        raise TypeError("formatter must be callable")

    before_text = formatter(before) or "-"
    after_text = formatter(after) or "-"
    return f"{before_text} -> {after_text}"
