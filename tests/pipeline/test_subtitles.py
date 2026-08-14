from pathlib import Path

import pytest
from dublaro.pipeline.subtitles import (
    default_srt_path,
    format_srt_timestamp,
    save_srt,
    transcript_to_srt,
)
from dublaro.schemas import Segment, Transcript


def test_transcript_to_srt_uses_auto_text_priority() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0002",
                start=2.0,
                end=3.5,
                source_text="Second source",
                translated_text="Second translated",
            ),
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.25,
                source_text="First source",
                translated_text="First translated",
                adapted_text="First adapted",
            ),
        ],
    )

    assert transcript_to_srt(transcript) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "First adapted\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,500\n"
        "Second translated\n"
    )


def test_transcript_to_srt_can_use_source_text() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                source_text="Hello world",
                translated_text="Czesc swiecie",
            )
        ],
    )

    assert "Hello world" in transcript_to_srt(transcript, text_mode="source")
    assert "Czesc swiecie" not in transcript_to_srt(transcript, text_mode="source")


def test_transcript_to_srt_skips_empty_selected_text() -> None:
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

    assert transcript_to_srt(transcript, text_mode="translated") == ""


def test_save_srt_writes_file(tmp_path: Path) -> None:
    output_path = tmp_path / "lesson.srt"
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

    saved_path = save_srt(transcript, output_path)

    assert saved_path == output_path
    assert output_path.read_text(encoding="utf-8").startswith("1\n")


def test_format_srt_timestamp() -> None:
    assert format_srt_timestamp(3661.234) == "01:01:01,234"


def test_format_srt_timestamp_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        format_srt_timestamp(-0.1)


def test_default_srt_path() -> None:
    assert default_srt_path("lesson.pl.adapted.json") == Path("lesson.pl.adapted.srt")
