import subprocess
from array import array
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import pytest
from dublaro.adapters.diarization import FakeDiarizationAdapter
from dublaro.adapters.translation import FakeTranslationAdapter
from dublaro.audio.wav import read_mono_pcm16_wav, write_mono_pcm16_wav
from dublaro.cli.doctor import build_doctor_report
from dublaro.pipeline.transcribe import load_transcript, save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")
cli_batch = import_module("dublaro.cli.batch")
cli_dub_runner = import_module("dublaro.cli.dub_runner")
cli_doctor = import_module("dublaro.cli.doctor")


runner = CliRunner()


def test_cli_shows_version() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert "dublaro" in result.output


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

    monkeypatch.setattr("dublaro.cli.doctor.shutil.which", fake_which)
    monkeypatch.setattr("dublaro.cli.doctor.subprocess.run", fake_run)

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

    monkeypatch.setattr("dublaro.cli.doctor.shutil.which", fake_which)
    monkeypatch.setattr("dublaro.cli.doctor.subprocess.run", fake_run)

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


def test_inspect_workspace_command_shows_workspace_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "lesson.audio.wav").write_bytes(b"audio")

    result = runner.invoke(cli.app, ["inspect-workspace", str(workspace)])

    assert result.exit_code == 0
    assert "Workspace" in result.output
    assert "Artifacts" in result.output
    assert "Workspace Artifacts" in result.output


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


def test_translate_command_writes_translated_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.en.json"
    output_path = tmp_path / "audio.pl.json"
    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello world",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "translate",
            str(transcript_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    transcript = load_transcript(output_path)

    assert transcript.target_language == "pl"
    assert transcript.segments[0].translated_text == "[pl] Hello world"
    assert transcript.metadata["translation_adapter"] == "fake-translation"


def test_translate_command_passes_translator_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript_path = tmp_path / "audio.en.json"
    output_path = tmp_path / "audio.pl.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello world",
                )
            ],
        ),
        transcript_path,
    )

    calls: list[dict[str, object]] = []

    def fake_create_translation_adapter(
        backend: str,
        *,
        auto_install: bool = False,
    ) -> FakeTranslationAdapter:
        calls.append({"backend": backend, "auto_install": auto_install})
        return FakeTranslationAdapter()

    monkeypatch.setattr(
        cli,
        "create_translation_adapter",
        fake_create_translation_adapter,
    )

    result = runner.invoke(
        cli.app,
        [
            "translate",
            str(transcript_path),
            "--to",
            "es",
            "--output",
            str(output_path),
            "--translator",
            "argos",
            "--install-package",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"backend": "argos", "auto_install": True}]


def test_preview_units_command_shows_grouped_segments(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.en.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    speaker="speaker-1",
                    source_text="I think that",
                ),
                Segment(
                    id="seg-0002",
                    start=1.2,
                    end=2.0,
                    speaker="speaker-1",
                    source_text="this matters.",
                ),
                Segment(
                    id="seg-0003",
                    start=3.0,
                    end=4.0,
                    speaker="speaker-1",
                    source_text="Next point.",
                ),
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "preview-units",
            str(transcript_path),
            "--max-group-pause",
            "0.8",
        ],
    )

    assert result.exit_code == 0
    assert "Translation units" in result.output
    assert "2 from 3 segments" in result.output
    assert "Next point." in result.output


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


def test_preview_timing_command_shows_speedup_and_video_fit(tmp_path: Path) -> None:
    speedup_clip = tmp_path / "speedup.wav"
    video_clip = tmp_path / "video.wav"
    transcript_path = tmp_path / "audio.pl.synthesized.json"

    write_mono_pcm16_wav(speedup_clip, samples=array("h", [0] * 12), sample_rate=10)
    write_mono_pcm16_wav(video_clip, samples=array("h", [0] * 20), sample_rate=10)

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-speedup",
                    start=0.0,
                    end=1.0,
                    generated_audio_path=str(speedup_clip),
                ),
                Segment(
                    id="seg-video",
                    start=1.0,
                    end=2.0,
                    generated_audio_path=str(video_clip),
                ),
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "preview-timing",
            str(transcript_path),
            "--max-speedup",
            "1.35",
            "--min-overrun",
            "0.05",
        ],
    )

    assert result.exit_code == 0
    assert "Timing preview" in result.output
    assert "2 segments" in result.output
    assert "seg-speedup" in result.output
    assert "seg-video" in result.output
    assert "speed-up" in result.output
    assert "needs-video" in result.output
    assert "2.00x" in result.output


def test_adapt_text_command_writes_adapted_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.json"
    output_path = tmp_path / "audio.pl.adapted.json"

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
                    translated_text="  Cześć    świecie  ",
                    target_language="pl",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "adapt-text",
            str(transcript_path),
            "--output",
            str(output_path),
            "--max-chars-per-second",
            "14",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    transcript = load_transcript(output_path)

    assert transcript.segments[0].adapted_text == "Cześć świecie"
    assert transcript.metadata["text_adapter"] == "rules"
    assert transcript.metadata["text_adapter_max_chars_per_second"] == "14.0"
    assert transcript.metadata["text_adapter_preserve_meaning"] == "true"


def test_export_srt_command_writes_subtitles(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.json"
    output_path = tmp_path / "audio.pl.srt"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.5,
                    source_text="Hello world",
                    translated_text="Czesc swiecie",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "export-srt",
            str(transcript_path),
            "--output",
            str(output_path),
            "--text",
            "translated",
        ],
    )

    assert result.exit_code == 0
    assert output_path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\nCzesc swiecie\n"
    )


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


def test_align_speech_command_writes_speech_track(tmp_path: Path) -> None:
    clip_path = tmp_path / "seg-0001.wav"
    transcript_path = tmp_path / "audio.pl.synthesized.json"
    output_path = tmp_path / "speech-track.wav"

    write_mono_pcm16_wav(
        clip_path,
        samples=array("h", [1000, 1000, 1000]),
        sample_rate=10,
    )

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            duration=1.0,
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.5,
                    end=0.8,
                    generated_audio_path=str(clip_path),
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "align-speech",
            str(transcript_path),
            "--output",
            str(output_path),
            "--sample-rate",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    sample_rate, samples = read_mono_pcm16_wav(output_path)

    assert sample_rate == 10
    assert list(samples[5:8]) == [1000, 1000, 1000]


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

    monkeypatch.setattr(cli, "export_dubbed_video", fake_export_dubbed_video)

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


def test_mix_audio_command_writes_mixed_audio(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.synthesized.json"
    original_path = tmp_path / "original.wav"
    speech_path = tmp_path / "speech-track.wav"
    output_path = tmp_path / "mixed.wav"

    write_mono_pcm16_wav(
        original_path,
        samples=array("h", [1000] * 10),
        sample_rate=10,
    )
    write_mono_pcm16_wav(
        speech_path,
        samples=array("h", [0, 0, 200, 200, 200, 0, 0, 0, 0, 0]),
        sample_rate=10,
    )

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            duration=1.0,
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.2,
                    end=0.5,
                    generated_audio_path=str(tmp_path / "seg-0001.wav"),
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "mix-audio",
            str(transcript_path),
            str(original_path),
            str(speech_path),
            "--output",
            str(output_path),
            "--ducking-gain",
            "0.5",
            "--ducking-margin",
            "0",
            "--ducking-fade",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    sample_rate, samples = read_mono_pcm16_wav(output_path)

    assert sample_rate == 10
    assert list(samples) == [1000, 1000, 700, 700, 700, 1000, 1000, 1000, 1000, 1000]


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
                "workspace_dir": workspace_dir,
                "speaker_voices": speaker_voices,
                "diarize": diarize,
                "diarization_min_speakers": diarization_min_speakers,
                "diarization_max_speakers": diarization_max_speakers,
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
            "workspace_dir": workspace_dir,
            "speaker_voices": None,
            "diarize": False,
            "diarization_min_speakers": None,
            "diarization_max_speakers": None,
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
            "progress_callback": cli.print_dub_progress,
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
