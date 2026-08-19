from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")
cli_command_normalize_audio = import_module("dublaro.cli.commands.normalize_audio")


runner = CliRunner()


def test_normalize_audio_command_calls_normalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_audio = tmp_path / "speech.wav"
    output_audio = tmp_path / "speech.normalized.wav"
    input_audio.write_bytes(b"fake wav")

    calls: list[dict[str, object]] = []

    def fake_normalize_audio_loudness(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        target_lufs: float = -16.0,
        true_peak: float = -1.5,
        loudness_range: float = 11.0,
        sample_rate: int | None = None,
        channels: int | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        calls.append(
            {
                "input_path": Path(input_path),
                "output_path": Path(output_path) if output_path is not None else None,
                "target_lufs": target_lufs,
                "true_peak": true_peak,
                "loudness_range": loudness_range,
                "sample_rate": sample_rate,
                "channels": channels,
                "overwrite": overwrite,
                "executable": executable,
            }
        )
        return Path(output_path) if output_path is not None else input_audio

    monkeypatch.setattr(
        cli_command_normalize_audio,
        "normalize_audio_loudness",
        fake_normalize_audio_loudness,
    )

    result = runner.invoke(
        cli.app,
        [
            "normalize-audio",
            str(input_audio),
            "--output",
            str(output_audio),
            "--target-lufs",
            "-18",
            "--true-peak",
            "-2",
            "--loudness-range",
            "9",
            "--sample-rate",
            "24000",
            "--channels",
            "1",
            "--ffmpeg",
            "ffmpeg-test",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert "Normalized audio saved" in result.output
    assert calls == [
        {
            "input_path": input_audio,
            "output_path": output_audio,
            "target_lufs": -18.0,
            "true_peak": -2.0,
            "loudness_range": 9.0,
            "sample_rate": 24_000,
            "channels": 1,
            "overwrite": True,
            "executable": "ffmpeg-test",
        }
    ]


def test_normalize_audio_command_reports_normalizer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_audio = tmp_path / "speech.wav"
    input_audio.write_bytes(b"fake wav")

    def fake_normalize_audio_loudness(*args: object, **kwargs: object) -> Path:
        raise FileExistsError("already exists")

    monkeypatch.setattr(
        cli_command_normalize_audio,
        "normalize_audio_loudness",
        fake_normalize_audio_loudness,
    )

    result = runner.invoke(cli.app, ["normalize-audio", str(input_audio)])

    assert result.exit_code == 1
    assert "already exists" in result.output
