from importlib import import_module
from pathlib import Path

from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_export_srt_command_writes_subtitles(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.json"
    output_path = tmp_path / "audio.pl.srt"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.5,
                    source_text="Hello world",
                    translated_text="Czesc swiecie",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "export-srt",
            str(transcript_path),
            "--output",
            str(output_path),
            "--text",
            "translated",
        ],
    )

    assert result.exit_code == 0
    assert output_path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\nCzesc swiecie\n"
    )
