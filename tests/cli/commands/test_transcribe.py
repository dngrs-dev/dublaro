from importlib import import_module
from pathlib import Path

from dublaro.pipeline.transcribe import load_transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_transcribe_command_writes_transcript(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "transcribe.json"
    audio_path.write_bytes(b"fake audio")

    result = runner.invoke(
        cli.app,
        [
            "transcribe",
            str(audio_path),
            "--output",
            str(output_path),
            "--language",
            "en",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    transcript = load_transcript(output_path)

    assert transcript.id == "audio"
    assert transcript.source_language == "en"
    assert transcript.segments[0].source_text == "This is a placeholder transcript."
    assert transcript.metadata["adapter"] == "fake-asr"
