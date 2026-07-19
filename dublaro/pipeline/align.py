from array import array
from pathlib import Path

from dublaro.audio.wav import (
    create_silence,
    mix_mono_pcm16_at,
    read_mono_pcm16_wav,
    write_mono_pcm16_wav,
)
from dublaro.schemas import Transcript


def build_speech_timeline(
    transcript: Transcript,
    *,
    output_path: str | Path,
    sample_rate: int = 24_000,
    duration: float | None = None,
) -> Path:
    clips = _load_segment_clips(transcript, sample_rate=sample_rate)

    timeline_duration = _resolve_timeline_duration(
        transcript,
        clips=clips,
        duration=duration,
        sample_rate=sample_rate,
    )
    timeline_frame_count = round(timeline_duration * sample_rate)
    timeline = create_silence(timeline_frame_count)

    for start_seconds, samples in clips:
        start_frame = round(start_seconds * sample_rate)
        mix_mono_pcm16_at(timeline, samples, start_frame=start_frame)

    return write_mono_pcm16_wav(
        output_path,
        timeline,
        sample_rate=sample_rate,
    )


def default_speech_timeline_path(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(f"{transcript_file.stem}.speech-track.wav")


def _load_segment_clips(
    transcript: Transcript,
    *,
    sample_rate: int,
) -> list[tuple[float, array]]:
    clips: list[tuple[float, array]] = []

    for segment in transcript.sorted_segments():
        if segment.generated_audio_path is None:
            continue

        clip_path = Path(segment.generated_audio_path)

        if not clip_path.exists():
            raise FileNotFoundError(f"Generated audio file does not exist: {clip_path}")

        clip_sample_rate, samples = read_mono_pcm16_wav(clip_path)

        if clip_sample_rate != sample_rate:
            raise ValueError(
                f"Expected generated audio sample rate {sample_rate}, "
                f"got {clip_sample_rate}: {clip_path}"
            )

        clips.append((segment.start, samples))

    return clips


def _resolve_timeline_duration(
    transcript: Transcript,
    *,
    clips: list[tuple[float, array]],
    duration: float | None,
    sample_rate: int,
) -> float:
    candidates: list[float] = []

    if duration is not None:
        candidates.append(duration)

    if transcript.duration is not None:
        candidates.append(transcript.duration)

    candidates.extend(segment.end for segment in transcript.segments)

    candidates.extend(
        start_seconds + (len(samples) / sample_rate) for start_seconds, samples in clips
    )

    if not candidates:
        raise ValueError("Cannot build a speech timeline without duration information.")

    return max(candidates)
