import json
from array import array
from importlib import import_module
from pathlib import Path

from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")

runner = CliRunner()


def test_report_command_summarizes_workspace_quality(tmp_path: Path) -> None:
    workspace = tmp_path / ".dublaro" / "lesson"
    workspace.mkdir(parents=True)

    audio_path = workspace / "seg-0001.wav"
    transcript_path = workspace / "lesson.pl.synthesized.json"
    manifest_path = workspace / "lesson.pl.manifest.json"

    write_mono_pcm16_wav(
        audio_path,
        array("h", [0] * 20),
        sample_rate=10,
    )

    save_transcript(
        Transcript(
            id="lesson",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    speaker="SPEAKER_00",
                    adapted_text="Dlugie zdanie.",
                    generated_audio_path=str(audio_path),
                )
            ],
        ),
        transcript_path,
    )

    manifest_path.write_text(
        json.dumps(
            {
                "input_video_path": "data/input/lesson.mp4",
                "output_video_path": "data/output/lesson.pl.dubbed.mp4",
                "language": {"source": "en", "target": "pl"},
                "adapters": {
                    "asr": {"name": "faster-whisper"},
                    "translation": {"name": "argos"},
                    "text_adapter": {"name": "ollama"},
                    "source_separation": {"name": "demucs"},
                    "tts": {"name": "piper"},
                    "speaker_voices": {"SPEAKER_00": {"adapter": {"name": "piper"}}},
                },
                "options": {
                    "text_workflow": "separate",
                    "background_mode": "separated",
                    "repair_timing": True,
                    "fit_speech": True,
                    "fit_video": True,
                    "max_speech_speedup": 1.35,
                    "min_speech_overrun_seconds": 0.05,
                    "subtitle_embed": "soft",
                    "normalize_final_audio": True,
                },
                "artifacts": {
                    "workspace_dir": str(workspace),
                    "synthesized_transcript_path": str(transcript_path),
                    "dubbed_video_path": str(workspace / "missing.mp4"),
                    "manifest_path": str(manifest_path),
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["report", str(workspace)])

    assert result.exit_code == 0
    assert "Quality report" in result.output
    assert "Artifacts" in result.output
    assert "demucs" in result.output
    assert "separated" in result.output
    assert "1 need attention" in result.output
    assert "1 need video fitting" in result.output


def test_report_command_can_fail_on_quality_issues(tmp_path: Path) -> None:
    workspace = tmp_path / ".dublaro" / "lesson"
    workspace.mkdir(parents=True)
    (workspace / "lesson.pl.manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "workspace_dir": str(workspace),
                    "dubbed_video_path": str(workspace / "missing.mp4"),
                }
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["report", str(workspace), "--fail-on-issues"],
    )

    assert result.exit_code == 1
    assert "missing" in result.output
