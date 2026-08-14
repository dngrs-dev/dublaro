from importlib import import_module
from pathlib import Path

from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_preview_speakers_command_shows_voice_routes(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.json"
    config_path = tmp_path / "dublaro.toml"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(id="seg-1", start=0.0, end=1.0, speaker="SPEAKER_00"),
                Segment(id="seg-2", start=1.5, end=2.5, speaker="SPEAKER_01"),
                Segment(id="seg-3", start=3.0, end=4.0, speaker="SPEAKER_00"),
            ],
        ),
        transcript_path,
    )

    config_path.write_text(
        """
[dub.tts]
backend = "fake"

[voices."SPEAKER_00"]
display_name = "Host"
tts_backend = "piper"
piper_model_path = "models/host.onnx"
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["preview-speakers", str(transcript_path), "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert "Speakers" in result.output
    assert "SPEAKER_00" in result.output
    assert "SPEAKER_01" in result.output
    assert "Host" in result.output
    assert "configured" in result.output
    assert "fallback" in result.output
    assert "host.onnx" in result.output


def test_preview_speakers_command_warns_for_speaker_voice_mismatches(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "audio.json"
    config_path = tmp_path / "dublaro.toml"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(id="seg-1", start=0.0, end=1.0, speaker="SPEAKER_00"),
                Segment(id="seg-2", start=1.0, end=2.0, speaker="SPEAKER_02"),
            ],
        ),
        transcript_path,
    )

    config_path.write_text(
        """
[voices."SPEAKER_00"]
tts_backend = "fake"

[voices."SPEAKER_99"]
tts_backend = "fake"
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["preview-speakers", str(transcript_path), "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert "No configured voice profile" in result.output
    assert "SPEAKER_02" in result.output
    assert "fallback TTS" in result.output
    assert "Configured voice profiles not present" in result.output
    assert "SPEAKER_99" in result.output
