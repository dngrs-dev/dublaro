from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")
cli_command_export_video = import_module("dublaro.cli.commands.export_video")


runner = CliRunner()


def test_export_video_command_writes_dubbed_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    speech_track_path = tmp_path / "speech-track.wav"
    output_path = tmp_path / "video.pl.dubbed.mp4"

    video_path.write_bytes(b"fake video")
    speech_track_path.write_bytes(b"fake audio")

    calls: list[dict[str, object]] = []

    def fake_export_dubbed_video(
        video_path: Path,
        speech_track_path: Path,
        output_path: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        calls.append(
            {
                "video_path": video_path,
                "speech_track_path": speech_track_path,
                "output_path": output_path,
                "overwrite": overwrite,
            }
        )
        output_path.write_bytes(b"fake dubbed video")
        return output_path

    monkeypatch.setattr(
        cli_command_export_video,
        "export_dubbed_video",
        fake_export_dubbed_video,
    )

    result = runner.invoke(
        cli.app,
        [
            "export-video",
            str(video_path),
            str(speech_track_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert calls == [
        {
            "video_path": video_path,
            "speech_track_path": speech_track_path,
            "output_path": output_path,
            "overwrite": True,
        }
    ]
