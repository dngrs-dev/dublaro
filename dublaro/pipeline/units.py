from dataclasses import dataclass

from dublaro.schemas import Segment, Transcript, WordTiming

_SENTENCE_ENDINGS = (".", "!", "?")
_TRAILING_CLOSERS = "\"')]}"


@dataclass(frozen=True)
class SegmentGroup:
    segments: tuple[Segment, ...]

    @property
    def id(self) -> str:
        if len(self.segments) == 1:
            return self.segments[0].id

        return f"{self.segments[0].id}_to_{self.segments[-1].id}"

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def speaker(self) -> str | None:
        speakers = {segment.speaker for segment in self.segments}
        if len(speakers) == 1:
            return self.segments[0].speaker

        return None

    @property
    def source_text(self) -> str:
        return " ".join(
            segment.source_text.strip()
            for segment in self.segments
            if segment.source_text.strip()
        )

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def group_segments_for_translation(
    transcript: Transcript,
    *,
    max_pause_seconds: float = 0.8,
    max_duration_seconds: float = 12.0,
    max_sentence_duration_seconds: float = 24.0,
) -> list[SegmentGroup]:
    if max_pause_seconds < 0:
        raise ValueError("max_pause_seconds must be >= 0")

    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be > 0")

    if max_sentence_duration_seconds <= 0:
        raise ValueError("max_sentence_duration_seconds must be > 0")

    if max_sentence_duration_seconds < max_duration_seconds:
        raise ValueError(
            "max_sentence_duration_seconds must be >= max_duration_seconds"
        )

    groups: list[SegmentGroup] = []
    current: list[Segment] = []

    for segment in transcript.sorted_segments():
        if not current:
            current.append(segment)
            continue

        previous = current[-1]
        pause_seconds = segment.start - previous.end
        candidate_duration = segment.end - current[0].start
        previous_ends_sentence = _ends_sentence(previous.source_text)

        should_split = (
            segment.speaker != previous.speaker
            or pause_seconds > max_pause_seconds
            or previous_ends_sentence
            or candidate_duration > max_sentence_duration_seconds
        )

        if should_split:
            groups.append(SegmentGroup(tuple(current)))
            current = [segment]
        else:
            current.append(segment)

    if current:
        groups.append(SegmentGroup(tuple(current)))

    return groups


def merge_segment_group(group: SegmentGroup) -> Segment:
    if not group.segments:
        raise ValueError("Cannot merge an empty segment group.")

    segments = group.segments
    source_segment_ids = [segment.id for segment in segments]

    words: list[WordTiming] = [
        word.model_copy(deep=True) for segment in segments for word in segment.words
    ]

    confidence_values = [
        segment.confidence for segment in segments if segment.confidence is not None
    ]

    confidence = (
        sum(confidence_values) / len(confidence_values) if confidence_values else None
    )

    metadata = {
        **segments[0].metadata,
        "source_segment_ids": ",".join(source_segment_ids),
        "source_segment_count": str(len(source_segment_ids)),
    }

    return Segment(
        id=group.id,
        start=group.start,
        end=group.end,
        speaker=group.speaker,
        source_text=group.source_text,
        translated_text=_join_text(segment.translated_text for segment in segments),
        adapted_text=_join_text(segment.adapted_text for segment in segments),
        source_language=segments[0].source_language,
        target_language=segments[0].target_language,
        words=words,
        confidence=confidence,
        generated_audio_path=(
            segments[0].generated_audio_path if len(segments) == 1 else None
        ),
        metadata=metadata,
    )


def _ends_sentence(text: str) -> bool:
    stripped = text.strip()

    while stripped and stripped[-1] in _TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()

    return stripped.endswith(_SENTENCE_ENDINGS)


def _join_text(values) -> str:
    return " ".join(value.strip() for value in values if value.strip())
