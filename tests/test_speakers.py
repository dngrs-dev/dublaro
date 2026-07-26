from dublaro.pipeline.speakers import (
    find_unconfigured_speakers,
    find_unused_voice_profiles,
    summarize_transcript_speakers,
)
from dublaro.schemas import Segment, Transcript


def test_summarize_transcript_speakers_counts_segments_and_time() -> None:
    transcript = Transcript(
        id="audio",
        source_language="en",
        segments=[
            Segment(id="seg-1", start=0.0, end=1.0, speaker="SPEAKER_01"),
            Segment(id="seg-2", start=1.5, end=2.0, speaker="SPEAKER_00"),
            Segment(id="seg-3", start=2.0, end=3.5, speaker="SPEAKER_01"),
        ],
    )

    summaries = summarize_transcript_speakers(transcript)

    assert [summary.speaker_id for summary in summaries] == [
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert summaries[0].segment_count == 2
    assert summaries[0].total_duration_seconds == 2.5
    assert summaries[0].first_start == 0.0
    assert summaries[0].last_end == 3.5


def test_find_speaker_voice_mismatches() -> None:
    transcript = Transcript(
        id="audio",
        source_language="en",
        segments=[
            Segment(id="seg-1", start=0.0, end=1.0, speaker="SPEAKER_00"),
            Segment(id="seg-2", start=1.0, end=2.0, speaker="SPEAKER_02"),
            Segment(id="seg-3", start=2.0, end=3.0),
        ],
    )

    configured_speakers = ["SPEAKER_00", "SPEAKER_99"]

    assert find_unconfigured_speakers(transcript, configured_speakers) == ["SPEAKER_02"]
    assert find_unused_voice_profiles(transcript, configured_speakers) == ["SPEAKER_99"]
