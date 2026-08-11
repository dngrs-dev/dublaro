from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

from dublaro.adapters.text_adapter import (
    TextAdapter,
    TextTimingRepairOptions,
    TimingRepairTextAdapter,
)
from dublaro.adapters.tts import SpeechSynthesisOptions, TtsAdapter
from dublaro.audio.wav import read_mono_pcm16_wav
from dublaro.pipeline.timing_diagnostics import duration_ratio, format_timing_value
from dublaro.pipeline.voices import SpeakerVoice
from dublaro.schemas import Segment, Transcript


def repair_overlong_speech_segments(
    transcript: Transcript,
    *,
    text_adapter: TextAdapter,
    tts_adapter: TtsAdapter,
    output_dir: str | Path,
    language: str,
    sample_rate: int,
    speaker_voices: Mapping[str, SpeakerVoice] | None = None,
    max_attempts: int = 2,
    target_speedup: float = 1.15,
    min_overrun_seconds: float = 0.05,
    overwrite: bool = False,
) -> Transcript:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")

    if target_speedup < 1.0:
        raise ValueError("target_speedup must be >= 1.0.")

    if min_overrun_seconds < 0:
        raise ValueError("min_overrun_seconds must be >= 0.")

    repaired = transcript.model_copy(deep=True)
    repair_dir = Path(output_dir)
    repair_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(text_adapter, TimingRepairTextAdapter):
        repaired.metadata = {
            **repaired.metadata,
            "timing_repair": "skipped_adapter_unsupported",
            "timing_repair_adapter": text_adapter.name,
        }
        return repaired

    attempted_count = 0
    improved_count = 0
    repaired_count = 0

    for segment in repaired.sorted_segments():
        if segment.generated_audio_path is None or segment.duration <= 0:
            continue

        current_audio_path = Path(segment.generated_audio_path)
        current_audio_duration = _audio_duration(current_audio_path)
        max_audio_duration = segment.duration * target_speedup

        if current_audio_duration - max_audio_duration <= min_overrun_seconds:
            continue

        attempted_count += 1
        attempts_made = 0

        original_audio_duration = current_audio_duration
        best_text = _spoken_text(segment)
        best_audio_path = current_audio_path
        best_audio_duration = current_audio_duration

        for attempt in range(1, max_attempts + 1):
            repair_input = segment.model_copy(deep=True)
            repair_input.adapted_text = best_text
            repair_input.generated_audio_path = str(best_audio_path)

            repaired_text = _normalize_spacing(
                text_adapter.repair_segment_timing(
                    repair_input,
                    TextTimingRepairOptions(
                        source_language=transcript.source_language,
                        target_language=transcript.target_language or language,
                        target_duration_seconds=segment.duration,
                        current_audio_duration_seconds=best_audio_duration,
                        max_audio_duration_seconds=max_audio_duration,
                        attempt=attempt,
                        max_attempts=max_attempts,
                    ),
                )
            )

            if not repaired_text or _same_spoken_text(repaired_text, best_text):
                continue

            if len(repaired_text) >= len(best_text):
                continue

            attempts_made += 1
            candidate_segment = segment.model_copy(deep=True)
            candidate_segment.adapted_text = repaired_text
            candidate_segment.generated_audio_path = None

            candidate_audio_path = repair_dir / _repair_audio_filename(
                segment.id,
                attempt,
            )
            if candidate_audio_path.exists() and overwrite:
                candidate_audio_path.unlink()

            segment_tts_adapter, synthesis_options = _segment_synthesis_target(
                candidate_segment,
                fallback_adapter=tts_adapter,
                speaker_voices=speaker_voices,
                language=language,
                sample_rate=sample_rate,
            )

            generated_audio_path = segment_tts_adapter.synthesize_segment(
                candidate_segment,
                candidate_audio_path,
                synthesis_options,
            )
            generated_audio_duration = _audio_duration(generated_audio_path)

            if generated_audio_duration < best_audio_duration:
                if best_audio_path != current_audio_path:
                    _delete_file(best_audio_path)

                best_text = repaired_text
                best_audio_path = generated_audio_path
                best_audio_duration = generated_audio_duration
            else:
                _delete_file(generated_audio_path)

            if best_audio_duration - max_audio_duration <= min_overrun_seconds:
                break

        if best_audio_path != current_audio_path:
            segment.adapted_text = best_text
            segment.generated_audio_path = str(best_audio_path)
            improved_count += 1

        fits_after_repair = best_audio_duration - max_audio_duration <= (
            min_overrun_seconds
        )
        if fits_after_repair:
            repaired_count += 1

        segment.metadata = {
            **segment.metadata,
            "timing_repair_status": _repair_status(
                improved=best_audio_path != current_audio_path,
                fits=fits_after_repair,
            ),
            "timing_repair_attempts": str(attempts_made),
            "timing_repair_target_duration_seconds": format_timing_value(
                segment.duration
            ),
            "timing_repair_original_audio_duration_seconds": format_timing_value(
                original_audio_duration
            ),
            "timing_repair_best_audio_duration_seconds": format_timing_value(
                best_audio_duration
            ),
            "timing_repair_max_audio_duration_seconds": format_timing_value(
                max_audio_duration
            ),
            "timing_repair_required_speedup_before": format_timing_value(
                duration_ratio(original_audio_duration, segment.duration)
            ),
            "timing_repair_required_speedup_after": format_timing_value(
                duration_ratio(best_audio_duration, segment.duration)
            ),
        }

    repaired.metadata = {
        **repaired.metadata,
        "timing_repair": "text-adapter",
        "timing_repair_adapter": text_adapter.name,
        "timing_repair_attempted_segments": str(attempted_count),
        "timing_repair_improved_segments": str(improved_count),
        "timing_repair_repaired_segments": str(repaired_count),
        "timing_repair_max_attempts": str(max_attempts),
        "timing_repair_target_speedup": format_timing_value(target_speedup),
        "timing_repair_min_overrun_seconds": format_timing_value(min_overrun_seconds),
    }

    return repaired


def _segment_synthesis_target(
    segment: Segment,
    *,
    fallback_adapter: TtsAdapter,
    speaker_voices: Mapping[str, SpeakerVoice] | None,
    language: str,
    sample_rate: int,
) -> tuple[TtsAdapter, SpeechSynthesisOptions]:
    speaker_voice = None
    if speaker_voices is not None and segment.speaker is not None:
        speaker_voice = speaker_voices.get(segment.speaker)

    segment_adapter = fallback_adapter
    voice_profile = None
    segment_language = language

    if speaker_voice is not None:
        segment_adapter = speaker_voice.adapter
        voice_profile = speaker_voice.profile
        if voice_profile.language is not None:
            segment_language = voice_profile.language

    return (
        segment_adapter,
        SpeechSynthesisOptions(
            language=segment_language,
            sample_rate=sample_rate,
            speaker_id=segment.speaker,
            voice_profile=voice_profile,
        ),
    )


def _audio_duration(path: str | Path) -> float:
    sample_rate, samples = read_mono_pcm16_wav(path)
    return len(samples) / sample_rate


def _spoken_text(segment: Segment) -> str:
    return _normalize_spacing(
        segment.adapted_text or segment.translated_text or segment.source_text
    )


def _repair_audio_filename(segment_id: str, attempt: int) -> str:
    safe_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in segment_id
    ).strip("_")

    return f"{safe_id or 'segment'}.repair-{attempt}.wav"


def _repair_status(*, improved: bool, fits: bool) -> str:
    if improved and fits:
        return "repaired"

    if improved:
        return "improved"

    return "not_improved"


def _same_spoken_text(left: str, right: str) -> bool:
    return _comparison_text(left) == _comparison_text(right)


def _comparison_text(text: str) -> str:
    return _normalize_spacing(text).strip(".,!?;:").casefold()


def _delete_file(path: str | Path) -> None:
    with suppress(FileNotFoundError):
        Path(path).unlink()


def _normalize_spacing(text: str) -> str:
    return " ".join(text.split())
