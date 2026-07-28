from dataclasses import dataclass

from dublaro.audio.wav import read_mono_pcm16_wav
from dublaro.pipeline.timing_diagnostics import update_video_timing_diagnostics
from dublaro.schemas import Transcript


@dataclass(frozen=True)
class VideoSlowdownPlan:
    slowdown_factor: float
    overlong_segment_count: int
    limiting_segment_id: str | None = None


def plan_video_slowdown(
    transcript: Transcript,
    *,
    max_slowdown: float = 1.5,
    min_overrun_seconds: float = 0.05,
) -> VideoSlowdownPlan:
    if max_slowdown < 1.0:
        raise ValueError("max_slowdown must be >= 1.0")

    if min_overrun_seconds < 0:
        raise ValueError("min_overrun_seconds must be >= 0")

    slowdown_factor = 1.0
    overlong_segment_count = 0
    limiting_segment_id = None

    for segment in transcript.sorted_segments():
        if segment.generated_audio_path is None or segment.duration <= 0:
            continue

        sample_rate, samples = read_mono_pcm16_wav(segment.generated_audio_path)
        audio_duration = len(samples) / sample_rate
        overrun_seconds = audio_duration - segment.duration

        if overrun_seconds <= min_overrun_seconds:
            continue

        overlong_segment_count += 1
        required_slowdown = audio_duration / segment.duration

        if required_slowdown > slowdown_factor:
            slowdown_factor = required_slowdown
            limiting_segment_id = segment.id

    if slowdown_factor > max_slowdown:
        raise ValueError(
            f"Generated speech needs {slowdown_factor:.2f}x video slowdown, "
            f"which is above max_video_slowdown={max_slowdown:.2f}."
        )

    return VideoSlowdownPlan(
        slowdown_factor=slowdown_factor,
        overlong_segment_count=overlong_segment_count,
        limiting_segment_id=limiting_segment_id,
    )


def scale_transcript_timing(
    transcript: Transcript,
    *,
    slowdown_factor: float,
    min_overrun_seconds: float = 0.05,
) -> Transcript:
    if slowdown_factor < 1.0:
        raise ValueError("slowdown_factor must be >= 1.0")

    scaled = transcript.model_copy(deep=True)

    for segment in scaled.segments:
        original_duration = segment.duration

        if segment.generated_audio_path is not None and original_duration > 0:
            sample_rate, samples = read_mono_pcm16_wav(segment.generated_audio_path)
            audio_duration = len(samples) / sample_rate

            if audio_duration - original_duration > min_overrun_seconds:
                update_video_timing_diagnostics(
                    segment,
                    target_duration=original_duration,
                    audio_duration=audio_duration,
                    applied_video_slowdown=slowdown_factor,
                )

        segment.start *= slowdown_factor
        segment.end *= slowdown_factor

    if scaled.duration is not None:
        scaled.duration *= slowdown_factor

    scaled.metadata = {
        **scaled.metadata,
        "video_fitting": "ffmpeg-setpts",
        "video_fitting_slowdown_factor": f"{slowdown_factor:.6g}",
    }

    return scaled
