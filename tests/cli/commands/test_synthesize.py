from importlib import import_module
from pathlib import Path

from dublaro.pipeline.transcribe import load_transcript, save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_synthesize_command_writes_audio_and_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.adapted.json"
    output_dir = tmp_path / "speech"
    output_path = tmp_path / "audio.pl.synthesized.json"

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
                    translated_text="Cześć świecie",
                    adapted_text="Cześć świecie",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "synthesize",
            str(transcript_path),
            "--output-dir",
            str(output_dir),
            "--output",
            str(output_path),
            "--sample-rate",
            "16000",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert (output_dir / "seg-0001.wav").exists()

    transcript = load_transcript(output_path)

    assert transcript.metadata["tts_adapter"] == "fake-tts"
    assert transcript.metadata["tts_sample_rate"] == "16000"
    assert transcript.segments[0].generated_audio_path == str(
        output_dir / "seg-0001.wav"
    )
