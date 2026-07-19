from pathlib import Path

from dublaro.adapters.text_adapter import FakeTextAdapter
from dublaro.pipeline.adapt_text import (
    adapt_transcript_text,
    default_adapted_transcript_path,
)
from dublaro.schemas import Segment, Transcript


def test_adapt_transcript_text_uses_translated_text() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.5,
                source_text="Hello world",
                translated_text="  Cześć    świecie  ",
                target_language="pl",
            )
        ],
    )

    adapted = adapt_transcript_text(
        transcript,
        adapter=FakeTextAdapter(),
    )

    assert adapted.segments[0].adapted_text == "Cześć świecie"
    assert adapted.segments[0].translated_text == "  Cześć    świecie  "
    assert adapted.metadata["text_adapter"] == "fake-text-adapter"


def test_adapt_transcript_text_falls_back_to_source_text() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                source_text="  Hello    world  ",
            )
        ],
    )

    adapted = adapt_transcript_text(
        transcript,
        adapter=FakeTextAdapter(),
        target_language="en",
    )

    assert adapted.segments[0].adapted_text == "Hello world"


def test_adapt_transcript_text_does_not_mutate_original() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                translated_text="Cześć",
            )
        ],
    )

    adapted = adapt_transcript_text(
        transcript,
        adapter=FakeTextAdapter(),
    )

    assert transcript.segments[0].adapted_text == ""
    assert adapted.segments[0].adapted_text == "Cześć"


def test_default_adapted_transcript_path() -> None:
    assert default_adapted_transcript_path("lesson.pl.json") == Path(
        "lesson.pl.adapted.json"
    )
