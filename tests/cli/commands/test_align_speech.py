from array import array
from importlib import import_module
from pathlib import Path

from dublaro.audio.wav import read_mono_pcm16_wav, write_mono_pcm16_wav
from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


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
