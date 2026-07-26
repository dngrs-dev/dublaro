from dataclasses import dataclass

from dublaro.schemas import Transcript

UNASSIGNED_SPEAKER_ID = "unassigned"


@dataclass(frozen=True)
class SpeakerSummary:
    speaker_id: str
    segment_count: int
    total_duration_seconds: float
    first_start: float
    last_end: float


def summarize_transcript_speakers(transcript: Transcript) -> list[SpeakerSummary]:
    segments_by_speaker = _group_segments_by_speaker(transcript)

    summaries = [
        SpeakerSummary(
            speaker_id=speaker_id,
            segment_count=len(segments),
            total_duration_seconds=sum(
                max(0.0, segment.duration) for segment in segments
            ),
            first_start=min(segment.start for segment in segments),
            last_end=max(segment.end for segment in segments),
        )
        for speaker_id, segments in segments_by_speaker.items()
    ]

    return sorted(
        summaries, key=lambda summary: (summary.first_start, summary.speaker_id)
    )


def _group_segments_by_speaker(transcript: Transcript):
    segments_by_speaker = {}

    for segment in transcript.sorted_segments():
        speaker_id = segment.speaker or UNASSIGNED_SPEAKER_ID
        segments_by_speaker.setdefault(speaker_id, []).append(segment)

    return segments_by_speaker
