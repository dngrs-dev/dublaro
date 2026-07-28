from dublaro.schemas import Segment


def format_timing_value(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def duration_ratio(audio_duration: float, target_duration: float) -> float:
    if target_duration <= 0:
        return 1.0
    return max(1.0, audio_duration / target_duration)


def update_speech_timing_diagnostics(
    segment: Segment,
    *,
    target_duration: float,
    original_audio_duration: float,
    fitted_audio_duration: float,
    applied_speedup: float,
    max_speedup: float,
    min_overrun_seconds: float,
    status: str,
) -> None:
    remaining_speedup = duration_ratio(fitted_audio_duration, target_duration)

    segment.metadata = {
        **segment.metadata,
        "timing_target_duration_seconds": format_timing_value(target_duration),
        "timing_original_audio_duration_seconds": format_timing_value(
            original_audio_duration
        ),
        "timing_fitted_audio_duration_seconds": format_timing_value(
            fitted_audio_duration
        ),
        "timing_overrun_seconds": format_timing_value(
            max(0.0, original_audio_duration - target_duration)
        ),
        "timing_required_speedup": format_timing_value(
            duration_ratio(original_audio_duration, target_duration)
        ),
        "timing_applied_speedup": format_timing_value(applied_speedup),
        "timing_remaining_speedup": format_timing_value(remaining_speedup),
        "timing_required_video_slowdown": format_timing_value(remaining_speedup),
        "timing_fit_status": status,
        "timing_max_speech_speedup": format_timing_value(max_speedup),
        "timing_min_overrun_seconds": format_timing_value(min_overrun_seconds),
    }


def update_video_timing_diagnostics(
    segment: Segment,
    *,
    target_duration: float,
    audio_duration: float,
    applied_video_slowdown: float,
) -> None:
    required_video_slowdown = duration_ratio(audio_duration, target_duration)

    segment.metadata = {
        **segment.metadata,
        "timing_required_video_slowdown": format_timing_value(required_video_slowdown),
        "timing_applied_video_slowdown": format_timing_value(applied_video_slowdown),
        "timing_video_fitted_duration_seconds": format_timing_value(
            target_duration * applied_video_slowdown
        ),
        "timing_video_fit_status": "video_slowed",
    }
