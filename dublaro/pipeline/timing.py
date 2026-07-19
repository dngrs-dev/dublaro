from dataclasses import dataclass

from dublaro.audio.wav import read_mono_pcm16_wav
from dublaro.schemas import Transcript


@dataclass(frozen=True)
class TimingIssue:
    segment_id: str
    start: float
    end: float
    target_duration: float
    audio_duration: float
    overrun_seconds: float
    ratio: float


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

        sample_rate, samples = read_mono_pcm16_wav(segment.generated_audio_path)
        audio_duration = len(samples) / sample_rate
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
