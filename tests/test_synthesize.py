import wave
from pathlib import Path

from dublaro.adapters.tts import FakeTtsAdapter
from dublaro.pipeline.synthesize import (
    default_speech_output_dir,
    default_synthesized_transcript_path,
    synthesize_transcript_speech,
)
from dublaro.schemas import Segment, Transcript


def test_synthesize_transcript_speech_generates_audio_files(tmp_path: Path) -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                speaker="speaker-1",
                source_text="Hello world",
                translated_text="Cześć świecie",
                adapted_text="Cześć świecie",
            )
        ],
    )

    synthesized = synthesize_transcript_speech(
        transcript,
        adapter=FakeTtsAdapter(),
        output_dir=tmp_path / "speech",
        sample_rate=16_000,
    )

    generated_path = Path(synthesized.segments[0].generated_audio_path or "")

    assert generated_path.exists()
    assert synthesized.metadata["tts_adapter"] == "fake-tts"
    assert synthesized.metadata["tts_language"] == "pl"
    assert synthesized.metadata["tts_sample_rate"] == "16000"

    with wave.open(str(generated_path), "rb") as audio_file:
        assert audio_file.getnchannels() == 1
        assert audio_file.getframerate() == 16_000
        assert audio_file.getnframes() == 16_000


def test_synthesize_transcript_speech_does_not_mutate_original(tmp_path: Path) -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                adapted_text="Cześć",
            )
        ],
    )

    synthesized = synthesize_transcript_speech(
        transcript,
        adapter=FakeTtsAdapter(),
        output_dir=tmp_path / "speech",
    )

    assert transcript.segments[0].generated_audio_path is None
    assert synthesized.segments[0].generated_audio_path is not None


def test_synthesize_transcript_speech_skips_empty_segments(tmp_path: Path) -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
            )
        ],
    )

    synthesized = synthesize_transcript_speech(
        transcript,
        adapter=FakeTtsAdapter(),
        output_dir=tmp_path / "speech",
    )

    assert synthesized.segments[0].generated_audio_path is None
    assert list((tmp_path / "speech").iterdir()) == []


def test_default_synthesized_transcript_path() -> None:
    assert default_synthesized_transcript_path("lesson.pl.adapted.json") == Path(
        "lesson.pl.adapted.synthesized.json"
    )


def test_default_speech_output_dir() -> None:
    assert default_speech_output_dir("lesson.pl.adapted.json") == Path(
        "lesson.pl.adapted.speech"
    )
