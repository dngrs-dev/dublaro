from importlib import import_module
from pathlib import Path

from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_preview_repairs_command_shows_timing_repair_decisions(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "audio.pl.timing-repaired.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            metadata={
                "timing_repair": "text-adapter",
                "timing_repair_adapter": "ollama",
            },
            segments=[
                Segment(
                    id="seg-repaired",
                    start=0.0,
                    end=1.0,
                    speaker="SPEAKER_00",
                    adapted_text="Short text.",
                    metadata={
                        "timing_repair_status": "repaired",
                        "timing_repair_attempts": "1",
                        "timing_repair_target_duration_seconds": "1",
                        "timing_repair_max_audio_duration_seconds": "1.15",
                        "timing_repair_original_audio_duration_seconds": "1.8",
                        "timing_repair_best_audio_duration_seconds": "1",
                        "timing_repair_required_speedup_before": "1.8",
                        "timing_repair_required_speedup_after": "1",
                        "timing_repair_model_reason": "Shorter.",
                    },
                ),
                Segment(
                    id="seg-improved",
                    start=1.0,
                    end=2.0,
                    speaker="SPEAKER_01",
                    adapted_text="Still long.",
                    metadata={
                        "timing_repair_status": "improved",
                        "timing_repair_attempts": "2",
                        "timing_repair_target_duration_seconds": "1",
                        "timing_repair_max_audio_duration_seconds": "1.15",
                        "timing_repair_original_audio_duration_seconds": "2",
                        "timing_repair_best_audio_duration_seconds": "1.4",
                        "timing_repair_required_speedup_before": "2",
                        "timing_repair_required_speedup_after": "1.4",
                    },
                ),
                Segment(
                    id="seg-clean",
                    start=2.0,
                    end=3.0,
                    speaker="SPEAKER_02",
                    adapted_text="Already fine.",
                ),
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(cli.app, ["preview-repairs", str(transcript_path)])

    assert result.exit_code == 0
    assert "Timing repair preview" in result.output
    assert "2 attempted from 3 segments" in result.output
    assert "1 repaired" in result.output
    assert "1 improved" in result.output
    assert "ollama" in result.output
    assert "seg-repaired" in result.output
    assert "Shorter." in result.output
    assert "seg-improved" in result.output
    assert "fits-target" in result.output
    assert "shorter-but-over-target" in result.output
    assert "seg-clean" not in result.output


def test_preview_repairs_command_can_show_not_attempted_segments(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "audio.pl.timing-repaired.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-clean",
                    start=0.0,
                    end=1.0,
                    adapted_text="Already fine.",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        ["preview-repairs", str(transcript_path), "--all"],
    )

    assert result.exit_code == 0
    assert "seg-clean" in result.output
    assert "not_attempted" in result.output
    assert "not-attempted" in result.output
