from importlib import import_module
from pathlib import Path

from dublaro.audio.wav import read_mono_pcm16_wav
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_preview_voices_command_generates_configured_voice_samples(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "dublaro.toml"
    output_dir = tmp_path / "samples"

    config_path.write_text(
        """
[dub]
target_language = "pl"
speech_sample_rate = 16000

[dub.tts]
backend = "fake"

[voices."SPEAKER_00"]
display_name = "Host"
tts_backend = "fake"

[voices."SPEAKER_01"]
display_name = "Guest"
tts_backend = "fake"
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "preview-voices",
            "--config",
            str(config_path),
            "--text",
            "Hello",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Voice samples" in result.output
    assert "SPEAKER_00" in result.output
    assert "SPEAKER_01" in result.output
    assert (output_dir / "SPEAKER_00.wav").exists()
    assert (output_dir / "SPEAKER_01.wav").exists()

    sample_rate, audio = read_mono_pcm16_wav(output_dir / "SPEAKER_00.wav")

    assert sample_rate == 16000
    assert len(audio) > 0
