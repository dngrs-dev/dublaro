from pathlib import Path
from typing import Literal

from dublaro.schemas import Segment, Transcript

SrtTextMode = Literal["auto", "source", "translated", "adapted"]


def transcript_to_srt(
    transcript: Transcript,
    *,
    text_mode: SrtTextMode = "auto",
) -> str:
    blocks: list[str] = []

    for index, segment in enumerate(_caption_segments(transcript, text_mode), start=1):
        text = _normalize_srt_text(_segment_text(segment, text_mode))

        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(segment.start)} --> "
                    f"{format_srt_timestamp(segment.end)}",
                    text,
                ]
            )
        )

    if not blocks:
        return ""

    return "\n\n".join(blocks) + "\n"


def save_srt(
    transcript: Transcript,
    output_path: str | Path,
    *,
    text_mode: SrtTextMode = "auto",
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        transcript_to_srt(transcript, text_mode=text_mode),
        encoding="utf-8",
    )
    return output_file


def default_srt_path(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_suffix(".srt")


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("SRT timestamp seconds must be >= 0")

    total_milliseconds = round(seconds * 1000)

    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    total_minutes, seconds_part = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)

    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"


def _caption_segments(
    transcript: Transcript,
    text_mode: SrtTextMode,
) -> list[Segment]:
    return [
        segment
        for segment in transcript.sorted_segments()
        if _segment_text(segment, text_mode).strip()
    ]


def _segment_text(segment: Segment, text_mode: SrtTextMode) -> str:
    if text_mode == "source":
        return segment.source_text

    if text_mode == "translated":
        return segment.translated_text

    if text_mode == "adapted":
        return segment.adapted_text

    if text_mode == "auto":
        return segment.adapted_text or segment.translated_text or segment.source_text

    raise ValueError("text_mode must be one of: auto, source, translated, adapted.")


def _normalize_srt_text(text: str) -> str:
    return " ".join(text.split())
