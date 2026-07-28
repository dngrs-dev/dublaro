from dataclasses import dataclass
from typing import Literal

from dublaro.audio.wav import read_mono_pcm16_wav
from dublaro.schemas import Transcript

TimingPreviewStatus = Literal[
    "ok", "speed-up", "needs-video", "missing-audio", "invalid-timing"
]


@dataclass(frozen=True)
class TimingIssue:
    segment_id: str
    start: float
    end: float
    target_duration: float
    audio_duration: float
    overrun_seconds: float
    ratio: float


@dataclass(frozen=True)
class TimingPreview:
    segment_id: str
    start: float
    end: float
    target_duration: float
    audio_duration: float | None
    overrun_seconds: float | None
    required_speedup: float | None
    applied_speedup: float | None
    needs_video_fit: bool
    status: TimingPreviewStatus


def analyze_speech_timing(
    transcript: Transcript,
    *,
    max_overrun_seconds: float = 0.15,
    max_ratio: float = 1.10,
) -> list[TimingIssue]:
    issues: list[TimingIssue] = []

    for segment in transcript.sorted_segments():
        if segment.generated_audio_path is None:
            continue

        audio_duration = _read_audio_duration(segment.generated_audio_path)
        target_duration = segment.duration

        if target_duration <= 0:
            continue

        overrun_seconds = audio_duration - target_duration
        ratio = audio_duration / target_duration

        if overrun_seconds > max_overrun_seconds or ratio > max_ratio:
            issues.append(
                TimingIssue(
                    segment_id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    target_duration=target_duration,
                    audio_duration=audio_duration,
                    overrun_seconds=overrun_seconds,
                    ratio=ratio,
                )
            )

    return issues


def preview_speech_timing(
    transcript: Transcript,
    *,
    max_speedup: float = 1.35,
    min_overrun_seconds: float = 0.05,
) -> list[TimingPreview]:
    if max_speedup < 1.0:
        raise ValueError("max_speedup must be >= 1.0")

    if min_overrun_seconds < 0:
        raise ValueError("min_overrun_seconds must be >= 0")

    previews: list[TimingPreview] = []

    for segment in transcript.sorted_segments():
        target_duration = segment.duration

        if target_duration <= 0:
            previews.append(
                TimingPreview(
                    segment_id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    target_duration=target_duration,
                    audio_duration=None,
                    overrun_seconds=None,
                    required_speedup=None,
                    applied_speedup=None,
                    needs_video_fit=False,
                    status="invalid-timing",
                )
            )
            continue

        if segment.generated_audio_path is None:
            previews.append(
                TimingPreview(
                    segment_id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    target_duration=target_duration,
                    audio_duration=None,
                    overrun_seconds=None,
                    required_speedup=None,
                    applied_speedup=None,
                    needs_video_fit=False,
                    status="missing-audio",
                )
            )
            continue

        audio_duration = _read_audio_duration(segment.generated_audio_path)
        overrun_seconds = max(0.0, audio_duration - target_duration)
        required_speedup = max(1.0, audio_duration / target_duration)

        if overrun_seconds <= min_overrun_seconds:
            status: TimingPreviewStatus = "ok"
            applied_speedup = 1.0
            needs_video_fit = False
        elif required_speedup <= max_speedup:
            status = "speed-up"
            applied_speedup = required_speedup
            needs_video_fit = False
        else:
            status = "needs-video"
            applied_speedup = max_speedup
            needs_video_fit = True

        previews.append(
            TimingPreview(
                segment_id=segment.id,
                start=segment.start,
                end=segment.end,
                target_duration=target_duration,
                audio_duration=audio_duration,
                overrun_seconds=overrun_seconds,
                required_speedup=required_speedup,
                applied_speedup=applied_speedup,
                needs_video_fit=needs_video_fit,
                status=status,
            )
        )

    return previews


def _read_audio_duration(path: str) -> float:
    sample_rate, samples = read_mono_pcm16_wav(path)
    return len(samples) / sample_rate
