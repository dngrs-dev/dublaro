from pathlib import Path

from dublaro.adapters.translation import FakeTranslationAdapter
from dublaro.pipeline.translate import (
    default_translated_transcript_path,
    translate_transcript,
)
from dublaro.schemas import Segment, Transcript


def test_translate_transcript_translates_segments() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.5,
                speaker="speaker-1",
                source_text="Hello world",
            )
        ],
    )

    translated = translate_transcript(
        transcript,
        adapter=FakeTranslationAdapter(),
        target_language="pl",
    )

    assert translated.target_language == "pl"
    assert translated.segments[0].target_language == "pl"
    assert translated.segments[0].translated_text == "[pl] Hello world"
    assert translated.segments[0].start == 0.0
    assert translated.segments[0].end == 1.5
    assert translated.segments[0].speaker == "speaker-1"
    assert translated.metadata["translation_adapter"] == "fake-translation"
    assert (
        translated.metadata["translation_max_sentence_group_duration_seconds"] == "24.0"
    )


def test_translate_transcript_does_not_mutate_original() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                source_text="Hello",
            )
        ],
    )

    translated = translate_transcript(
        transcript,
        adapter=FakeTranslationAdapter(),
        target_language="uk",
    )

    assert transcript.target_language is None
    assert transcript.segments[0].translated_text == ""
    assert translated.segments[0].translated_text == "[uk] Hello"


def test_translate_transcript_keeps_empty_text_empty() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                source_text="",
            )
        ],
    )

    translated = translate_transcript(
        transcript,
        adapter=FakeTranslationAdapter(),
        target_language="pl",
    )

    assert translated.segments[0].translated_text == ""


def test_default_translated_transcript_path() -> None:
    assert default_translated_transcript_path("audio.en.json", "pl") == Path(
        "audio.en.pl.json"
    )


def test_translate_transcript_groups_segments_before_translation() -> None:
    transcript = Transcript(
        id="lesson",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                speaker="a",
                source_text="I think that",
            ),
            Segment(
                id="seg-0002",
                start=1.2,
                end=2.0,
                speaker="a",
                source_text="this matters.",
            ),
        ],
    )

    translated = translate_transcript(
        transcript,
        adapter=FakeTranslationAdapter(),
        target_language="pl",
    )

    assert len(translated.segments) == 1
    assert translated.segments[0].id == "seg-0001_to_seg-0002"
    assert translated.segments[0].source_text == "I think that this matters."
    assert translated.segments[0].translated_text == "[pl] I think that this matters."


def test_translate_transcript_can_keep_original_segments() -> None:
    transcript = Transcript(
        id="lesson",
        source_language="en",
        segments=[
            Segment(id="seg-0001", start=0.0, end=1.0, source_text="Hello"),
            Segment(id="seg-0002", start=1.2, end=2.0, source_text="world"),
        ],
    )

    translated = translate_transcript(
        transcript,
        adapter=FakeTranslationAdapter(),
        target_language="pl",
        group_segments=False,
    )

    assert len(translated.segments) == 2
    assert translated.segments[0].translated_text == "[pl] Hello"
    assert translated.segments[1].translated_text == "[pl] world"
