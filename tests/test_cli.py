from pathlib import Path

import pytest
from dublaro import cli
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_shows_version() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert "dublaro" in result.output


def test_extract_audio_command_calls_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "video.mp4"
    output_audio = tmp_path / "voice.wav"
    input_video.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    def fake_extract_audio_from_video(
        input_path: Path,
        output_path: Path | None = None,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        overwrite: bool = False,
    ) -> Path:
        calls.append(
            {
                "input_path": input_path,
                "output_path": output_path,
                "sample_rate": sample_rate,
                "channels": channels,
                "overwrite": overwrite,
            }
        )
        return output_path or input_path.with_suffix(".wav")

    monkeypatch.setattr(
        cli,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )

    result = runner.invoke(
        cli.app,
        [
            "extract-audio",
            str(input_video),
            "--output",
            str(output_audio),
            "--sample-rate",
            "24000",
            "--channels",
            "2",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert "Audio extracted" in result.output
    assert calls == [
        {
            "input_path": input_video,
            "output_path": output_audio,
            "sample_rate": 24_000,
            "channels": 2,
            "overwrite": True,
        }
    ]


def test_extract_audio_command_reports_extractor_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "video.mp4"
    input_video.write_bytes(b"fake video")

    def fake_extract_audio_from_video(*args, **kwargs) -> Path:
        raise FileExistsError("already exists")

    monkeypatch.setattr(
        cli,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )

    result = runner.invoke(cli.app, ["extract-audio", str(input_video)])

    assert result.exit_code == 1
    assert "already exists" in result.output
