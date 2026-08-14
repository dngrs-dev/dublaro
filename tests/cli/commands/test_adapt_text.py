from importlib import import_module
from pathlib import Path

from dublaro.pipeline.transcribe import load_transcript, save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_adapt_text_command_writes_adapted_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.json"
    output_path = tmp_path / "audio.pl.adapted.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello world",
                    translated_text="  Cześć    świecie  ",
                    target_language="pl",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "adapt-text",
            str(transcript_path),
            "--output",
            str(output_path),
            "--max-chars-per-second",
            "14",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    transcript = load_transcript(output_path)

    assert transcript.segments[0].adapted_text == "Cześć świecie"
    assert transcript.metadata["text_adapter"] == "rules"
    assert transcript.metadata["text_adapter_max_chars_per_second"] == "14.0"
    assert transcript.metadata["text_adapter_preserve_meaning"] == "true"
