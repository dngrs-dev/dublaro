from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from dublaro.cli_config import ResolvedDubSettings
from dublaro.pipeline.dub import (
    DubbingArtifacts,
    DubbingProgressStatus,
    DubbingProgressStep,
)
from dublaro.pipeline.preflight import DubPreflightReport

console = Console()


@dataclass(frozen=True)
class BatchDubResult:
    video_path: Path
    output_path: Path | None
    status: str
    message: str


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


def print_dub_artifacts(artifacts: DubbingArtifacts) -> None:
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
