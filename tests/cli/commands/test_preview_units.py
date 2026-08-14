from importlib import import_module
from pathlib import Path

from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_preview_units_command_shows_grouped_segments(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.en.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    speaker="speaker-1",
                    source_text="I think that",
                ),
                Segment(
                    id="seg-0002",
                    start=1.2,
                    end=2.0,
                    speaker="speaker-1",
                    source_text="this matters.",
                ),
                Segment(
                    id="seg-0003",
                    start=3.0,
                    end=4.0,
                    speaker="speaker-1",
                    source_text="Next point.",
                ),
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "preview-units",
            str(transcript_path),
            "--max-group-pause",
            "0.8",
        ],
    )

    assert result.exit_code == 0
    assert "Translation units" in result.output
    assert "2 from 3 segments" in result.output
    assert "Next point." in result.output
