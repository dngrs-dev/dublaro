from pathlib import Path

from dublaro.adapters.diarization import (
    DiarizationAdapter,
    DiarizationOptions,
    DiarizationTurn,
)
from dublaro.schemas import Segment, Transcript


def diarize_transcript(
    audio_path: str | Path,
    transcript: Transcript,
    *,
    adapter: DiarizationAdapter,
    options: DiarizationOptions | None = None,
) -> Transcript:
    turns = adapter.diarize(Path(audio_path), options or DiarizationOptions())
    return assign_speakers_to_transcript(transcript, turns, adapter_name=adapter.name)


def assign_speakers_to_transcript(
    transcript: Transcript,
    turns: list[DiarizationTurn],
    *,
    adapter_name: str,
    default_speaker: str = "speaker-1",
) -> Transcript:
    diarized = transcript.model_copy(deep=True)

    for segment in diarized.segments:
        segment.speaker = (
            _best_speaker(segment, turns) or segment.speaker or default_speaker
        )

    diarized.metadata = {
        **diarized.metadata,
        "diarization_adapter": adapter_name,
        "diarization_speaker_count": str(len(diarized.speakers())),
    }
    return diarized


def _best_speaker(segment: Segment, turns: list[DiarizationTurn]) -> str | None:
    scores: dict[str, float] = {}

    for turn in turns:
        overlap = _overlap_seconds(segment.start, segment.end, turn.start, turn.end)
        if overlap <= 0:
            continue
        scores[turn.speaker] = scores.get(turn.speaker, 0.0) + overlap

    if not scores:
        return None

    return max(scores.items(), key=lambda item: item[1])[0]


def _overlap_seconds(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))
