from pathlib import Path

from dublaro.audio.ffmpeg import change_audio_tempo
from dublaro.audio.wav import read_mono_pcm16_wav
from dublaro.pipeline.timing_diagnostics import update_speech_timing_diagnostics
from dublaro.schemas import Transcript


def fit_generated_speech_to_segments(
    transcript: Transcript,
    *,
    output_dir: str | Path,
    max_speedup: float = 1.35,
    min_overrun_seconds: float = 0.05,
    allow_unfit_overruns: bool = False,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Transcript:
    if max_speedup < 1.0:
        raise ValueError("max_speedup must be >= 1.0")

    if min_overrun_seconds < 0:
        raise ValueError("min_overrun_seconds must be >= 0")

    fitted = transcript.model_copy(deep=True)
    fitted_dir = Path(output_dir)
    fitted_dir.mkdir(parents=True, exist_ok=True)

    fitted_count = 0
    unresolved_count = 0

    for segment in fitted.sorted_segments():
        if segment.generated_audio_path is None:
            continue

        input_path = Path(segment.generated_audio_path)
        sample_rate, samples = read_mono_pcm16_wav(input_path)

        target_duration = segment.duration
        if target_duration <= 0:
            continue

        original_audio_duration = len(samples) / sample_rate
        overrun_seconds = original_audio_duration - target_duration
        required_tempo_factor = original_audio_duration / target_duration

        if overrun_seconds <= min_overrun_seconds:
            update_speech_timing_diagnostics(
                segment,
                target_duration=target_duration,
                original_audio_duration=original_audio_duration,
                fitted_audio_duration=original_audio_duration,
                applied_speedup=1.0,
                max_speedup=max_speedup,
                min_overrun_seconds=min_overrun_seconds,
                status="within_tolerance",
            )
            continue

        if required_tempo_factor > max_speedup:
            if not allow_unfit_overruns:
                raise ValueError(
                    f"Segment {segment.id} needs {required_tempo_factor:.2f}x speedup, "
                    f"which is above max_speedup={max_speedup:.2f}."
                )

            tempo_factor = max_speedup
            unresolved_count += 1
            status = "speech_capped_for_video"
        else:
            tempo_factor = required_tempo_factor
            status = "speech_fitted"

        output_path = fitted_dir / _fitted_audio_filename(segment.id)

        fitted_audio_path = change_audio_tempo(
            input_path,
            output_path,
            tempo_factor=tempo_factor,
            sample_rate=sample_rate,
            overwrite=overwrite,
            executable=executable,
        )

        fitted_sample_rate, fitted_samples = read_mono_pcm16_wav(fitted_audio_path)
        fitted_audio_duration = len(fitted_samples) / fitted_sample_rate

        segment.generated_audio_path = str(fitted_audio_path)
        update_speech_timing_diagnostics(
            segment,
            target_duration=target_duration,
            original_audio_duration=original_audio_duration,
            fitted_audio_duration=fitted_audio_duration,
            applied_speedup=tempo_factor,
            max_speedup=max_speedup,
            min_overrun_seconds=min_overrun_seconds,
            status=status,
        )
        fitted_count += 1

    fitted.metadata = {
        **fitted.metadata,
        "speech_fitting": "ffmpeg-atempo",
        "speech_fitting_fitted_segments": str(fitted_count),
        "speech_fitting_unresolved_segments": str(unresolved_count),
        "speech_fitting_max_speedup": str(max_speedup),
        "speech_fitting_min_overrun_seconds": str(min_overrun_seconds),
    }

    return fitted


def default_fitted_speech_output_dir(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(f"{transcript_file.stem}.fitted-speech")


def default_fitted_transcript_path(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(
        f"{transcript_file.stem}.fitted{transcript_file.suffix}"
    )


def _fitted_audio_filename(segment_id: str) -> str:
    safe_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in segment_id
    ).strip("_")

    return f"{safe_id or 'segment'}.fit.wav"
