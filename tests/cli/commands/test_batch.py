from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")
cli_batch = import_module("dublaro.cli.services.batch")
cli_dub_runner = import_module("dublaro.cli.services.dub_runner")


runner = CliRunner()


def test_batch_command_runs_each_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    workspace_root = tmp_path / "work"

    input_dir.mkdir()
    first_video = input_dir / "lesson.mp4"
    second_video = input_dir / "zoo.mov"

    first_video.write_bytes(b"fake video")
    second_video.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    @dataclass
    class FakeArtifacts:
        dubbed_video_path: Path
        workspace_dir: Path
        fitted_transcript_path: Path | None = None
        mixed_audio_path: Path | None = None
        srt_path: Path | None = None
        manifest_path: Path | None = None

    def fake_dub_video(
        video_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> FakeArtifacts:
        calls.append({"video_path": video_path, "output_path": output_path, **kwargs})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake dubbed video")

        workspace_dir = kwargs["workspace_dir"]
        assert isinstance(workspace_dir, Path)

        return FakeArtifacts(output_path, workspace_dir)

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(input_dir),
            "--to",
            "pl",
            "--output-dir",
            str(output_dir),
            "--workspace-root",
            str(workspace_root),
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 2
    assert calls[0]["video_path"] == first_video
    assert calls[0]["output_path"] == output_dir / "lesson.pl.dubbed.mp4"
    assert calls[0]["workspace_dir"] == workspace_root / "lesson"
    assert calls[1]["video_path"] == second_video
    assert calls[1]["output_path"] == output_dir / "zoo.pl.dubbed.mov"
    assert calls[1]["workspace_dir"] == workspace_root / "zoo"
    assert "Batch summary" in result.output


def test_batch_command_passes_timing_repair_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config_path = tmp_path / "dublaro.toml"

    input_dir.mkdir()
    video_path = input_dir / "lesson.mp4"
    video_path.write_bytes(b"fake video")

    config_path.write_text(
        """
[dub.text_adapter]
backend = "ollama"
ollama_model = "llama3.1"

[dub.tts]
backend = "fake"
""",
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    @dataclass
    class FakeArtifacts:
        dubbed_video_path: Path
        workspace_dir: Path
        fitted_transcript_path: Path | None = None
        mixed_audio_path: Path | None = None
        srt_path: Path | None = None
        manifest_path: Path | None = None

    def fake_dub_video(
        video_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> FakeArtifacts:
        calls.append({"video_path": video_path, "output_path": output_path, **kwargs})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake dubbed video")

        workspace_dir = kwargs["workspace_dir"]
        assert isinstance(workspace_dir, Path)

        return FakeArtifacts(output_path, workspace_dir)

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(input_dir),
            "--config",
            str(config_path),
            "--to",
            "pl",
            "--output-dir",
            str(output_dir),
            "--repair-timing",
            "--timing-repair-attempts",
            "4",
            "--timing-repair-target-speedup",
            "1.2",
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["repair_timing"] is True
    assert calls[0]["max_timing_repair_attempts"] == 4
    assert calls[0]["timing_repair_target_speedup"] == 1.2


def test_batch_command_dry_run_does_not_call_dub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"fake video")

    calls: list[str] = []

    def fake_dub_video(*args: object, **kwargs: object) -> object:
        calls.append("dub")
        raise AssertionError("dub_video should not run during dry run")

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(input_dir),
            "--to",
            "pl",
            "--dry-run",
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert calls == []
    assert "planned" in result.output


def test_batch_command_continues_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()
    bad_video = input_dir / "bad.mp4"
    good_video = input_dir / "good.mp4"

    bad_video.write_bytes(b"fake video")
    good_video.write_bytes(b"fake video")

    calls: list[Path] = []

    @dataclass
    class FakeArtifacts:
        dubbed_video_path: Path
        workspace_dir: Path
        fitted_transcript_path: Path | None = None
        mixed_audio_path: Path | None = None
        srt_path: Path | None = None
        manifest_path: Path | None = None

    def fake_dub_video(
        video_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> FakeArtifacts:
        calls.append(video_path)

        if video_path.name == "bad.mp4":
            raise RuntimeError("boom")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake dubbed video")

        workspace_dir = kwargs["workspace_dir"]
        assert isinstance(workspace_dir, Path)

        return FakeArtifacts(output_path, workspace_dir)

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(input_dir),
            "--to",
            "pl",
            "--output-dir",
            str(output_dir),
            "--continue-on-error",
            "--no-preflight",
        ],
    )

    assert result.exit_code == 1
    assert calls == [bad_video, good_video]
    assert "failed" in result.output
    assert "done" in result.output


def test_batch_command_rejects_exact_config_output_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"fake video")

    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"
output_path = "exact.mp4"
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(input_dir),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1
    assert "Batch cannot use dub.output_path" in result.output


def test_batch_command_dry_run_continues_after_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dublaro.pipeline.preflight import DubPreflightReport, PreflightIssue

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()
    bad_video = input_dir / "bad.mp4"
    good_video = input_dir / "good.mp4"

    bad_video.write_bytes(b"fake video")
    good_video.write_bytes(b"fake video")

    calls: list[str] = []

    def fake_run_dub_preflight(
        video_path: Path,
        settings: object,
    ) -> DubPreflightReport:
        if video_path.name == "bad.mp4":
            return DubPreflightReport(
                (
                    PreflightIssue(
                        severity="error",
                        code="output_exists",
                        message="Output already exists",
                    ),
                )
            )

        return DubPreflightReport(())

    def fake_dub_video(*args: object, **kwargs: object) -> object:
        calls.append("dub")
        raise AssertionError("dub_video should not run during dry run")

    monkeypatch.setattr(cli_batch, "run_dub_preflight", fake_run_dub_preflight)
    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(input_dir),
            "--to",
            "pl",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert calls == []
    assert "bad.mp4" in result.output
    assert "good.mp4" in result.output
    assert "failed" in result.output
    assert "planned" in result.output
