from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import pytest
from dublaro.adapters.diarization import FakeDiarizationAdapter
from dublaro.adapters.source_separation import FakeSourceSeparationAdapter
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")
cli_dub_runner = import_module("dublaro.cli.services.dub_runner")
cli_command_dub = import_module("dublaro.cli.commands.dub")


runner = CliRunner()


def test_dub_command_runs_full_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "video.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

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
        *,
        source_language: str | None,
        target_language: str,
        text_workflow: str = "translate-then-adapt",
        workspace_dir: Path,
        asr_adapter: object,
        diarization_adapter: object | None = None,
        translation_adapter: object,
        text_adapter: object,
        tts_adapter: object,
        speaker_voices: object | None = None,
        diarize: bool = False,
        diarization_min_speakers: int | None = None,
        diarization_max_speakers: int | None = None,
        dubbing_script_adapter: object | None = None,
        source_separation_adapter: object | None = None,
        background_mode: str = "speech-only",
        translation_group_segments: bool = True,
        max_translation_group_pause_seconds: float = 0.8,
        max_translation_group_duration_seconds: float = 12.0,
        max_translation_sentence_group_duration_seconds: float = 24.0,
        asr_sample_rate: int = 16_000,
        speech_sample_rate: int = 24_000,
        repair_timing: bool = False,
        max_timing_repair_attempts: int = 2,
        timing_repair_target_speedup: float = 1.15,
        fit_speech: bool = False,
        max_speech_speedup: float = 1.35,
        min_speech_overrun_seconds: float = 0.05,
        fit_video: bool = False,
        max_video_slowdown: float = 1.5,
        mix_original_audio: bool = False,
        original_audio_gain: float = 1.0,
        ducking_gain: float = 0.25,
        speech_gain: float = 1.0,
        ducking_margin_seconds: float = 0.05,
        ducking_fade_seconds: float = 0.05,
        export_srt: bool = False,
        srt_output_path: Path | None = None,
        srt_text_mode: str = "adapted",
        subtitle_embed: str = "none",
        write_manifest: bool = True,
        manifest_output_path: Path | None = None,
        progress_callback: object | None = None,
        ffmpeg_executable: str = "ffmpeg",
        resume: bool = False,
        overwrite: bool = False,
    ) -> FakeArtifacts:
        calls.append(
            {
                "video_path": video_path,
                "output_path": output_path,
                "source_language": source_language,
                "target_language": target_language,
                "text_workflow": text_workflow,
                "workspace_dir": workspace_dir,
                "speaker_voices": speaker_voices,
                "diarize": diarize,
                "diarization_min_speakers": diarization_min_speakers,
                "diarization_max_speakers": diarization_max_speakers,
                "dubbing_script_adapter": dubbing_script_adapter,
                "source_separation_adapter": source_separation_adapter,
                "background_mode": background_mode,
                "translation_group_segments": translation_group_segments,
                "max_translation_group_pause_seconds": max_translation_group_pause_seconds,
                "max_translation_group_duration_seconds": max_translation_group_duration_seconds,
                "max_translation_sentence_group_duration_seconds": max_translation_sentence_group_duration_seconds,
                "asr_sample_rate": asr_sample_rate,
                "speech_sample_rate": speech_sample_rate,
                "repair_timing": repair_timing,
                "max_timing_repair_attempts": max_timing_repair_attempts,
                "timing_repair_target_speedup": timing_repair_target_speedup,
                "fit_speech": fit_speech,
                "max_speech_speedup": max_speech_speedup,
                "min_speech_overrun_seconds": min_speech_overrun_seconds,
                "fit_video": fit_video,
                "max_video_slowdown": max_video_slowdown,
                "mix_original_audio": mix_original_audio,
                "original_audio_gain": original_audio_gain,
                "ducking_gain": ducking_gain,
                "speech_gain": speech_gain,
                "ducking_margin_seconds": ducking_margin_seconds,
                "ducking_fade_seconds": ducking_fade_seconds,
                "export_srt": export_srt,
                "srt_output_path": srt_output_path,
                "srt_text_mode": srt_text_mode,
                "subtitle_embed": subtitle_embed,
                "write_manifest": write_manifest,
                "manifest_output_path": manifest_output_path,
                "progress_callback": progress_callback,
                "ffmpeg_executable": ffmpeg_executable,
                "resume": resume,
                "overwrite": overwrite,
            }
        )
        output_path.write_bytes(b"fake dubbed video")
        return FakeArtifacts(dubbed_video_path=output_path, workspace_dir=workspace_dir)

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--from",
            "en",
            "--to",
            "pl",
            "--output",
            str(output_path),
            "--workspace",
            str(workspace_dir),
            "--max-sentence-group-duration",
            "20",
            "--asr-sample-rate",
            "16000",
            "--speech-sample-rate",
            "24000",
            "--fit-speech",
            "--max-speech-speedup",
            "1.25",
            "--min-speech-overrun",
            "0.1",
            "--fit-video",
            "--max-video-slowdown",
            "1.4",
            "--mix-original-audio",
            "--original-audio-gain",
            "0.8",
            "--ducking-gain",
            "0.2",
            "--speech-gain",
            "1.1",
            "--ducking-margin",
            "0.2",
            "--ducking-fade",
            "0.03",
            "--export-srt",
            "--srt-output",
            str(tmp_path / "video.pl.srt"),
            "--srt-text",
            "adapted",
            "--subtitle-embed",
            "soft",
            "--manifest-output",
            str(tmp_path / "video.pl.manifest.json"),
            "--no-preflight",
            "--ffmpeg",
            "ffmpeg-test",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert calls == [
        {
            "video_path": video_path,
            "output_path": output_path,
            "source_language": "en",
            "target_language": "pl",
            "text_workflow": "translate-then-adapt",
            "workspace_dir": workspace_dir,
            "speaker_voices": None,
            "diarize": False,
            "diarization_min_speakers": None,
            "diarization_max_speakers": None,
            "dubbing_script_adapter": None,
            "source_separation_adapter": None,
            "background_mode": "ducked",
            "translation_group_segments": True,
            "max_translation_group_pause_seconds": 0.8,
            "max_translation_group_duration_seconds": 12.0,
            "max_translation_sentence_group_duration_seconds": 20.0,
            "asr_sample_rate": 16_000,
            "speech_sample_rate": 24_000,
            "repair_timing": False,
            "max_timing_repair_attempts": 2,
            "timing_repair_target_speedup": 1.15,
            "fit_speech": True,
            "max_speech_speedup": 1.25,
            "min_speech_overrun_seconds": 0.1,
            "fit_video": True,
            "max_video_slowdown": 1.4,
            "mix_original_audio": True,
            "original_audio_gain": 0.8,
            "ducking_gain": 0.2,
            "speech_gain": 1.1,
            "ducking_margin_seconds": 0.2,
            "ducking_fade_seconds": 0.03,
            "export_srt": True,
            "srt_output_path": tmp_path / "video.pl.srt",
            "srt_text_mode": "adapted",
            "subtitle_embed": "soft",
            "write_manifest": True,
            "manifest_output_path": tmp_path / "video.pl.manifest.json",
            "progress_callback": cli_command_dub.print_dub_progress,
            "ffmpeg_executable": "ffmpeg-test",
            "resume": False,
            "overwrite": True,
        }
    ]


def test_dub_command_stops_on_preflight_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dublaro.pipeline.preflight import DubPreflightReport, PreflightIssue

    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "video.pl.dubbed.mp4"
    video_path.write_bytes(b"fake video")

    calls: list[str] = []

    def fake_validate_dub_preflight(**kwargs: object) -> DubPreflightReport:
        return DubPreflightReport(
            (
                PreflightIssue(
                    severity="error",
                    code="ffmpeg_missing",
                    message="ffmpeg was not found",
                ),
            )
        )

    def fake_dub_video(*args: object, **kwargs: object) -> object:
        calls.append("dub")
        raise AssertionError("dub_video should not run")

    monkeypatch.setattr(
        cli_dub_runner,
        "validate_dub_preflight",
        fake_validate_dub_preflight,
    )
    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "Preflight failed" in result.output
    assert "ffmpeg_missing" in result.output
    assert calls == []


def test_dub_command_passes_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "video.pl.dubbed.mp4"

    video_path.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    @dataclass
    class FakeArtifacts:
        dubbed_video_path: Path
        workspace_dir: Path
        fitted_transcript_path: Path | None = None
        mixed_audio_path: Path | None = None
        srt_path: Path | None = None
        manifest_path: Path | None = None

    def fake_dub_video(*args: object, **kwargs: object) -> FakeArtifacts:
        calls.append(kwargs)
        output_path.write_bytes(b"fake dubbed video")
        return FakeArtifacts(
            dubbed_video_path=output_path,
            workspace_dir=tmp_path / "workspace",
        )

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
            "--workspace",
            str(tmp_path / "workspace"),
            "--resume",
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["resume"] is True


def test_dub_command_rejects_resume_with_overwrite(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "video.pl.dubbed.mp4"

    video_path.write_bytes(b"fake video")

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
            "--resume",
            "--overwrite",
        ],
    )

    assert result.exit_code != 0
    assert "--resume cannot be used with --overwrite" in result.output


def test_dub_command_reads_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
source_language = "en"
target_language = "pl"
output_path = "out/video.pl.dubbed.mp4"
workspace_dir = ".dublaro/video"
resume = true
preflight = false
ffmpeg_executable = "ffmpeg-config"
asr_sample_rate = 8000
speech_sample_rate = 16000

[dub.asr]
backend = "fake"
model_size = "tiny"
device = "cpu"
compute_type = "int8"

[dub.translation]
backend = "fake"
group_segments = false
max_group_pause_seconds = 0.4
max_group_duration_seconds = 9.0

[dub.text_adapter]
backend = "fake"

[dub.tts]
backend = "fake"

[dub.fit_speech]
enabled = true
max_speedup = 1.2
min_overrun_seconds = 0.2

[dub.fit_video]
enabled = true
max_slowdown = 1.4

[dub.mix]
enabled = true
original_audio_gain = 0.7
ducking_gain = 0.3
speech_gain = 1.1
ducking_margin_seconds = 0.1
ducking_fade_seconds = 0.2

[dub.srt]
export = true
output_path = "subs/video.pl.srt"
text_mode = "translated"
embed = "hard"

[dub.manifest]
write = true
output_path = "runs/manifest.json"
""",
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    class FakeArtifacts:
        def __init__(self, dubbed_video_path: Path, workspace_dir: Path) -> None:
            self.dubbed_video_path = dubbed_video_path
            self.workspace_dir = workspace_dir
            self.fitted_transcript_path = None
            self.mixed_audio_path = None
            self.srt_path = None
            self.manifest_path = None

    def fake_dub_video(
        video_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> FakeArtifacts:
        calls.append({"video_path": video_path, "output_path": output_path, **kwargs})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake dubbed video")
        return FakeArtifacts(output_path, kwargs["workspace_dir"])  # type: ignore[arg-type]

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        ["dub", str(video_path), "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert calls[0]["target_language"] == "pl"
    assert calls[0]["source_language"] == "en"
    assert calls[0]["output_path"] == tmp_path / "out" / "video.pl.dubbed.mp4"
    assert calls[0]["workspace_dir"] == tmp_path / ".dublaro" / "video"
    assert calls[0]["resume"] is True
    assert calls[0]["ffmpeg_executable"] == "ffmpeg-config"
    assert calls[0]["asr_sample_rate"] == 8000
    assert calls[0]["speech_sample_rate"] == 16000
    assert calls[0]["translation_group_segments"] is False
    assert calls[0]["fit_speech"] is True
    assert calls[0]["fit_video"] is True
    assert calls[0]["max_video_slowdown"] == 1.4
    assert calls[0]["mix_original_audio"] is True
    assert calls[0]["srt_output_path"] == tmp_path / "subs" / "video.pl.srt"
    assert calls[0]["srt_text_mode"] == "translated"
    assert calls[0]["subtitle_embed"] == "hard"
    assert calls[0]["manifest_output_path"] == tmp_path / "runs" / "manifest.json"


def test_dub_command_cli_values_override_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    cli_output_path = tmp_path / "cli-output.mp4"
    video_path.write_bytes(b"fake video")

    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"
output_path = "config-output.mp4"
preflight = false
""",
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    class FakeArtifacts:
        def __init__(self, dubbed_video_path: Path, workspace_dir: Path) -> None:
            self.dubbed_video_path = dubbed_video_path
            self.workspace_dir = workspace_dir
            self.fitted_transcript_path = None
            self.mixed_audio_path = None
            self.srt_path = None
            self.manifest_path = None

    def fake_dub_video(
        video_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> FakeArtifacts:
        calls.append({"video_path": video_path, "output_path": output_path, **kwargs})
        output_path.write_bytes(b"fake dubbed video")
        return FakeArtifacts(output_path, kwargs["workspace_dir"])  # type: ignore[arg-type]

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--config",
            str(config_path),
            "--to",
            "de",
            "--output",
            str(cli_output_path),
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["target_language"] == "de"
    assert calls[0]["output_path"] == cli_output_path


def test_dub_command_defaults_output_path_next_to_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "zoo.mp4"
    video_path.write_bytes(b"fake video")

    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"
preflight = false
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
        output_path.write_bytes(b"fake dubbed video")

        workspace_dir = kwargs["workspace_dir"]
        assert isinstance(workspace_dir, Path)

        return FakeArtifacts(
            dubbed_video_path=output_path,
            workspace_dir=workspace_dir,
        )

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        ["dub", str(video_path), "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert calls[0]["output_path"] == tmp_path / "zoo.pl.dubbed.mp4"


def test_dub_command_detects_piper_sample_rate_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "zoo.mp4"
    video_path.write_bytes(b"fake video")

    model_dir = tmp_path / "models" / "piper"
    model_dir.mkdir(parents=True)

    model_path = model_dir / "voice.onnx"
    piper_config_path = model_dir / "voice.onnx.json"

    model_path.write_bytes(b"fake model")
    piper_config_path.write_text(
        '{"audio": {"sample_rate": 22050}}',
        encoding="utf-8",
    )

    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"
preflight = false

[dub.tts]
backend = "piper"
piper_model_path = "models/piper/voice.onnx"
piper_config_path = "models/piper/voice.onnx.json"
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
        output_path.write_bytes(b"fake dubbed video")

        workspace_dir = kwargs["workspace_dir"]
        assert isinstance(workspace_dir, Path)

        return FakeArtifacts(
            dubbed_video_path=output_path,
            workspace_dir=workspace_dir,
        )

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        ["dub", str(video_path), "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert calls[0]["speech_sample_rate"] == 22050


def test_dub_command_uses_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "input" / "zoo.mp4"
    output_dir = tmp_path / "output"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake video")

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
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output-dir",
            str(output_dir),
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["output_path"] == output_dir / "zoo.pl.dubbed.mp4"


def test_dub_command_rejects_output_with_output_dir(tmp_path: Path) -> None:
    video_path = tmp_path / "zoo.mp4"
    video_path.write_bytes(b"fake video")

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output",
            str(tmp_path / "exact.mp4"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code != 0
    assert "--output cannot be used with --output-dir" in result.output


def test_dub_command_passes_diarization_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "video.pl.dubbed.mp4"
    video_path.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    class FakeArtifacts:
        def __init__(self, dubbed_video_path: Path, workspace_dir: Path) -> None:
            self.dubbed_video_path = dubbed_video_path
            self.workspace_dir = workspace_dir
            self.fitted_transcript_path = None
            self.mixed_audio_path = None
            self.srt_path = None
            self.manifest_path = None

    def fake_dub_video(
        video_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> FakeArtifacts:
        calls.append({"video_path": video_path, "output_path": output_path, **kwargs})
        output_path.write_bytes(b"fake dubbed video")
        return FakeArtifacts(output_path, kwargs["workspace_dir"])  # type: ignore[arg-type]

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
            "--diarize",
            "--diarizer",
            "fake",
            "--min-speakers",
            "1",
            "--max-speakers",
            "2",
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["diarize"] is True
    assert calls[0]["diarization_min_speakers"] == 1
    assert calls[0]["diarization_max_speakers"] == 2

    diarization_adapter = calls[0]["diarization_adapter"]

    assert isinstance(diarization_adapter, FakeDiarizationAdapter)
    assert diarization_adapter.name == "fake-diarization"


def test_dub_command_passes_text_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "video.pl.dubbed.mp4"
    video_path.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    class FakeArtifacts:
        def __init__(self, dubbed_video_path: Path, workspace_dir: Path) -> None:
            self.dubbed_video_path = dubbed_video_path
            self.workspace_dir = workspace_dir
            self.fitted_transcript_path = None
            self.mixed_audio_path = None
            self.srt_path = None
            self.manifest_path = None

    def fake_dub_video(
        video_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> FakeArtifacts:
        calls.append(kwargs)
        output_path.write_bytes(b"fake dubbed video")
        return FakeArtifacts(output_path, tmp_path / "workspace")

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
            "--translator",
            "ollama",
            "--text-workflow",
            "llm-dubbing",
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["text_workflow"] == "llm-dubbing"
    assert calls[0]["dubbing_script_adapter"] is not None


def test_dub_command_passes_source_separation_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    output_path = tmp_path / "video.pl.dubbed.mp4"
    video_path.write_bytes(b"fake video")

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
        output_path.write_bytes(b"fake dubbed video")
        return FakeArtifacts(output_path, tmp_path / "workspace")

    monkeypatch.setattr(cli_dub_runner, "dub_video", fake_dub_video)

    result = runner.invoke(
        cli.app,
        [
            "dub",
            str(video_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
            "--background-mode",
            "separated",
            "--source-separation",
            "fake",
            "--no-preflight",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["video_path"] == video_path
    assert calls[0]["output_path"] == output_path
    assert calls[0]["background_mode"] == "separated"
    assert calls[0]["mix_original_audio"] is True

    source_separation_adapter = calls[0]["source_separation_adapter"]

    assert isinstance(source_separation_adapter, FakeSourceSeparationAdapter)
    assert source_separation_adapter.name == "fake-source-separation"
