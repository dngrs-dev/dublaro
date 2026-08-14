from array import array
from importlib import import_module
from pathlib import Path

from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


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
