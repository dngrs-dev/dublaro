import subprocess
from importlib import import_module
from pathlib import Path

import pytest
from dublaro.cli.reports.doctor import build_doctor_report
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_doctor_command_checks_default_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_which(executable: str) -> str | None:
        if executable == "ffmpeg":
            return str(tmp_path / "ffmpeg.exe")
        return None

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="ffmpeg version test\n",
            stderr="",
        )

    monkeypatch.setattr("dublaro.cli.reports.doctor.shutil.which", fake_which)
    monkeypatch.setattr("dublaro.cli.reports.doctor.subprocess.run", fake_run)

    report = build_doctor_report()

    assert not report.has_errors
    assert any(
        check.category == "config"
        and check.status == "skipped"
        and "No config file loaded" in check.message
        for check in report.checks
    )
    assert any(
        check.name == "ffmpeg" and check.status == "ok" for check in report.checks
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Doctor" in result.output
    assert "ffmpeg" in result.output


def test_doctor_command_reports_missing_configured_piper_voice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"

[dub.tts]
backend = "fake"

[voices."SPEAKER_00"]
tts_backend = "piper"
piper_model_path = "models/missing.onnx"
""",
        encoding="utf-8",
    )

    def fake_which(executable: str) -> str | None:
        if executable == "ffmpeg":
            return str(tmp_path / "ffmpeg.exe")
        return None

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="ffmpeg version test\n")

    monkeypatch.setattr("dublaro.cli.reports.doctor.shutil.which", fake_which)
    monkeypatch.setattr("dublaro.cli.reports.doctor.subprocess.run", fake_run)

    report = build_doctor_report(config_path=config_path)

    assert report.has_errors
    assert any(
        check.category == "piper"
        and check.name == 'Speaker "SPEAKER_00" Piper model'
        and check.status == "error"
        and "does not exist" in check.message
        for check in report.checks
    )
    assert any(
        check.category == "piper"
        and check.name == "Piper executable"
        and check.status == "error"
        for check in report.checks
    )

    result = runner.invoke(cli.app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Doctor found problems" in result.output
    assert "SPEAKER_00" in result.output
    assert "Piper executable" in result.output


def test_doctor_command_reports_missing_demucs_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"
background_mode = "separated"

[dub.source_separation]
backend = "demucs"
demucs_executable = "demucs-custom"
""",
        encoding="utf-8",
    )

    def fake_which(executable: str) -> str | None:
        if executable == "ffmpeg":
            return str(tmp_path / "ffmpeg.exe")

        return None

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="ffmpeg version test\n")

    monkeypatch.setattr("dublaro.cli.reports.doctor.shutil.which", fake_which)
    monkeypatch.setattr("dublaro.cli.reports.doctor.subprocess.run", fake_run)

    report = build_doctor_report(config_path=config_path)

    assert report.has_errors
    assert any(
        check.category == "source-separation"
        and check.name == "Demucs executable"
        and check.status == "error"
        and "demucs-custom" in check.message
        for check in report.checks
    )

    result = runner.invoke(cli.app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Doctor found problems" in result.output
    assert "Demucs executable" in result.output
