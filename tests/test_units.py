from dublaro.pipeline.units import group_segments_for_translation, merge_segment_group
from dublaro.schemas import Segment, Transcript


def test_group_segments_for_translation_joins_sentence_fragments() -> None:
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
            Segment(
                id="seg-0003",
                start=3.0,
                end=4.0,
                speaker="a",
                source_text="Next point.",
            ),
        ],
    )

    groups = group_segments_for_translation(transcript, max_pause_seconds=0.8)

    assert [[segment.id for segment in group.segments] for group in groups] == [
        ["seg-0001", "seg-0002"],
        ["seg-0003"],
    ]


def test_group_segments_for_translation_splits_on_speaker_change() -> None:
    transcript = Transcript(
        id="lesson",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001", start=0.0, end=1.0, speaker="a", source_text="Hello"
            ),
            Segment(id="seg-0002", start=1.1, end=2.0, speaker="b", source_text="Hi"),
        ],
    )

    groups = group_segments_for_translation(transcript)

    assert [[segment.id for segment in group.segments] for group in groups] == [
        ["seg-0001"],
        ["seg-0002"],
    ]


def test_merge_segment_group_creates_one_dubbing_unit() -> None:
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

    group = group_segments_for_translation(transcript)[0]
    merged = merge_segment_group(group)

    assert merged.id == "seg-0001_to_seg-0002"
    assert merged.start == 0.0
    assert merged.end == 2.0
    assert merged.speaker == "a"
    assert merged.source_text == "I think that this matters."
    assert merged.metadata["source_segment_ids"] == "seg-0001,seg-0002"


def test_group_segments_allows_unfinished_sentence_past_soft_duration() -> None:
    transcript = Transcript(
        id="lesson",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=7.0,
                speaker="a",
                source_text="They rent it out to people",
            ),
            Segment(
                id="seg-0002",
                start=7.0,
                end=14.0,
                speaker="a",
                source_text="because you get to see how they live",
            ),
            Segment(
                id="seg-0003",
                start=14.0,
                end=16.0,
                speaker="a",
                source_text="and the whole experience.",
            ),
        ],
    )

    groups = group_segments_for_translation(
        transcript,
        max_duration_seconds=12.0,
        max_sentence_duration_seconds=24.0,
    )

    assert [[segment.id for segment in group.segments] for group in groups] == [
        ["seg-0001", "seg-0002", "seg-0003"],
    ]


def test_group_segments_splits_unfinished_sentence_at_hard_duration() -> None:
    transcript = Transcript(
        id="lesson",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=10.0,
                speaker="a",
                source_text="This sentence starts",
            ),
            Segment(
                id="seg-0002",
                start=10.0,
                end=25.0,
                speaker="a",
                source_text="and keeps going for too long",
            ),
        ],
    )

    groups = group_segments_for_translation(
        transcript,
        max_duration_seconds=12.0,
        max_sentence_duration_seconds=20.0,
    )

    assert [[segment.id for segment in group.segments] for group in groups] == [
        ["seg-0001"],
        ["seg-0002"],
    ]
