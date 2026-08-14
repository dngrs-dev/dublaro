from array import array
from importlib import import_module
from pathlib import Path

from dublaro.audio.wav import read_mono_pcm16_wav, write_mono_pcm16_wav
from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


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
