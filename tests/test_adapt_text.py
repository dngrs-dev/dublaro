from pathlib import Path

from dublaro.adapters.text_adapter import FakeTextAdapter, RuleBasedTextAdapter
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


def test_adapt_transcript_text_records_timing_metadata() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=2.0,
                translated_text="Talk about trees. Do you like trees?",
            )
        ],
    )

    adapted = adapt_transcript_text(
        transcript,
        adapter=RuleBasedTextAdapter(),
        max_chars_per_second=10.0,
    )

    segment = adapted.segments[0]

    assert segment.adapted_text == "Talk about trees. Do you like trees?"
    assert segment.metadata["adaptation_char_budget"] == "20"
    assert segment.metadata["adaptation_over_budget"] == "true"
    assert segment.metadata["adaptation_status"] == "over_budget_preserved"
    assert segment.metadata["adaptation_required_chars_per_second"] == "18.00"
    assert adapted.metadata["text_adapter_preserve_meaning"] == "true"


def test_default_adapted_transcript_path() -> None:
    assert default_adapted_transcript_path("lesson.pl.json") == Path(
        "lesson.pl.adapted.json"
    )
