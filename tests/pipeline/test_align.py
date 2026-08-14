from array import array
from pathlib import Path

import pytest
from dublaro.audio.wav import read_mono_pcm16_wav, write_mono_pcm16_wav
from dublaro.pipeline.align import (
    build_speech_timeline,
    default_speech_timeline_path,
)
from dublaro.schemas import Segment, Transcript


def test_build_speech_timeline_places_clip_at_segment_start(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "seg-0001.wav"
    output_path = tmp_path / "speech-track.wav"

    write_mono_pcm16_wav(
        clip_path,
        array("h", [1000, 1000, 1000]),
        sample_rate=10,
    )

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        duration=1.0,
        segments=[
            Segment(
                id="seg-0001",
                start=0.5,
                end=0.8,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    result = build_speech_timeline(
        transcript,
        output_path=output_path,
        sample_rate=10,
    )

    sample_rate, samples = read_mono_pcm16_wav(result)

    assert sample_rate == 10
    assert len(samples) == 10
    assert list(samples[:5]) == [0, 0, 0, 0, 0]
    assert list(samples[5:8]) == [1000, 1000, 1000]


def test_build_speech_timeline_mixes_overlapping_clips(tmp_path: Path) -> None:
    first_clip = tmp_path / "first.wav"
    second_clip = tmp_path / "second.wav"
    output_path = tmp_path / "speech-track.wav"

    write_mono_pcm16_wav(first_clip, array("h", [20_000, 20_000]), sample_rate=10)
    write_mono_pcm16_wav(second_clip, array("h", [20_000, 20_000]), sample_rate=10)

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        duration=0.2,
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=0.2,
                generated_audio_path=str(first_clip),
            ),
            Segment(
                id="seg-0002",
                start=0.0,
                end=0.2,
                generated_audio_path=str(second_clip),
            ),
        ],
    )

    result = build_speech_timeline(
        transcript,
        output_path=output_path,
        sample_rate=10,
    )

    _, samples = read_mono_pcm16_wav(result)

    assert list(samples) == [32767, 32767]


def test_build_speech_timeline_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    clip_path = tmp_path / "seg-0001.wav"

    write_mono_pcm16_wav(clip_path, array("h", [1000]), sample_rate=8)

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    with pytest.raises(ValueError, match="sample rate"):
        build_speech_timeline(
            transcript,
            output_path=tmp_path / "speech-track.wav",
            sample_rate=10,
        )


def test_default_speech_timeline_path() -> None:
    assert default_speech_timeline_path("lesson.pl.synthesized.json") == Path(
        "lesson.pl.synthesized.speech-track.wav"
    )
