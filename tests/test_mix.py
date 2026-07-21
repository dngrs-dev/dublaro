from array import array
from pathlib import Path

import pytest
from dublaro.audio.wav import read_mono_pcm16_wav, write_mono_pcm16_wav
from dublaro.pipeline.mix import (
    default_mixed_audio_path,
    mix_original_audio_with_dubbed_speech,
)
from dublaro.schemas import Segment, Transcript


def test_mix_original_audio_with_dubbed_speech_ducks_original_audio(
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "original.wav"
    speech_path = tmp_path / "speech.wav"
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

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.2,
                end=0.5,
                source_text="Hello",
                translated_text="Czesc",
                generated_audio_path=str(tmp_path / "seg-0001.wav"),
            )
        ],
    )

    saved_path = mix_original_audio_with_dubbed_speech(
        transcript,
        original_audio_path=original_path,
        speech_track_path=speech_path,
        output_path=output_path,
        ducking_gain=0.5,
        ducking_margin_seconds=0,
        ducking_fade_seconds=0,
    )

    sample_rate, samples = read_mono_pcm16_wav(saved_path)

    assert sample_rate == 10
    assert list(samples) == [1000, 1000, 700, 700, 700, 1000, 1000, 1000, 1000, 1000]


def test_mix_original_audio_with_dubbed_speech_rejects_sample_rate_mismatch(
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "original.wav"
    speech_path = tmp_path / "speech.wav"

    write_mono_pcm16_wav(original_path, samples=array("h", [0]), sample_rate=10)
    write_mono_pcm16_wav(speech_path, samples=array("h", [0]), sample_rate=20)

    transcript = Transcript(id="lesson-1", source_language="en")

    with pytest.raises(ValueError, match="same sample rate"):
        mix_original_audio_with_dubbed_speech(
            transcript,
            original_audio_path=original_path,
            speech_track_path=speech_path,
            output_path=tmp_path / "mixed.wav",
        )


def test_mix_original_audio_with_dubbed_speech_rejects_invalid_ducking_gain(
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "original.wav"
    speech_path = tmp_path / "speech.wav"

    write_mono_pcm16_wav(original_path, samples=array("h", [0]), sample_rate=10)
    write_mono_pcm16_wav(speech_path, samples=array("h", [0]), sample_rate=10)

    transcript = Transcript(id="lesson-1", source_language="en")

    with pytest.raises(ValueError, match="ducking_gain"):
        mix_original_audio_with_dubbed_speech(
            transcript,
            original_audio_path=original_path,
            speech_track_path=speech_path,
            output_path=tmp_path / "mixed.wav",
            original_gain=0.5,
            ducking_gain=0.8,
        )


def test_default_mixed_audio_path() -> None:
    assert default_mixed_audio_path("lesson.pl.speech-track.wav") == Path(
        "lesson.pl.speech-track.mixed.wav"
    )
