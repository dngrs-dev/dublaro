from array import array
from pathlib import Path

from dublaro.audio.wav import (
    PCM_MAX,
    PCM_MIN,
    create_silence,
    read_mono_pcm16_wav,
    write_mono_pcm16_wav,
)
from dublaro.schemas import Transcript


def mix_original_audio_with_dubbed_speech(
    transcript: Transcript,
    *,
    original_audio_path: str | Path,
    speech_track_path: str | Path,
    output_path: str | Path,
    original_gain: float = 1.0,
    ducking_gain: float = 0.25,
    speech_gain: float = 1.0,
    ducking_margin_seconds: float = 0.05,
    ducking_fade_seconds: float = 0.05,
) -> Path:
    _validate_gain("original_gain", original_gain)
    _validate_gain("ducking_gain", ducking_gain)
    _validate_gain("speech_gain", speech_gain)

    if ducking_gain > original_gain:
        raise ValueError("ducking_gain must be <= original_gain")

    if ducking_margin_seconds < 0:
        raise ValueError("ducking_margin_seconds must be >= 0")

    if ducking_fade_seconds < 0:
        raise ValueError("ducking_fade_seconds must be >= 0")

    original_sample_rate, original_samples = read_mono_pcm16_wav(original_audio_path)
    speech_sample_rate, speech_samples = read_mono_pcm16_wav(speech_track_path)

    if original_sample_rate != speech_sample_rate:
        raise ValueError(
            "Original audio and speech track must have the same sample rate: "
            f"{original_sample_rate} != {speech_sample_rate}"
        )

    frame_count = max(len(original_samples), len(speech_samples))
    mixed = create_silence(frame_count)

    for frame_index in range(frame_count):
        original_sample = _sample_at(original_samples, frame_index)
        mixed[frame_index] = _clip_pcm16(round(original_sample * original_gain))

    _duck_original_audio(
        mixed,
        original_samples,
        transcript,
        sample_rate=original_sample_rate,
        original_gain=original_gain,
        ducking_gain=ducking_gain,
        margin_seconds=ducking_margin_seconds,
        fade_seconds=ducking_fade_seconds,
    )

    for frame_index in range(frame_count):
        speech_sample = _sample_at(speech_samples, frame_index)
        mixed[frame_index] = _clip_pcm16(
            round(int(mixed[frame_index]) + (speech_sample * speech_gain))
        )

    return write_mono_pcm16_wav(
        output_path,
        mixed,
        sample_rate=original_sample_rate,
    )


def default_mixed_audio_path(speech_track_path: str | Path) -> Path:
    speech_track = Path(speech_track_path)
    return speech_track.with_name(f"{speech_track.stem}.mixed.wav")


def _duck_original_audio(
    mixed: array,
    original_samples: array,
    transcript: Transcript,
    *,
    sample_rate: int,
    original_gain: float,
    ducking_gain: float,
    margin_seconds: float,
    fade_seconds: float,
) -> None:
    margin_frames = round(margin_seconds * sample_rate)
    fade_frames = round(fade_seconds * sample_rate)

    for segment in transcript.sorted_segments():
        if segment.generated_audio_path is None:
            continue

        start_frame = max(0, round(segment.start * sample_rate) - margin_frames)
        end_frame = min(
            len(mixed),
            round(segment.end * sample_rate) + margin_frames,
        )

        if end_frame <= start_frame:
            continue

        active_fade_frames = min(fade_frames, (end_frame - start_frame) // 2)

        for frame_index in range(start_frame, end_frame):
            gain = _ducking_gain_for_frame(
                frame_index,
                start_frame=start_frame,
                end_frame=end_frame,
                fade_frames=active_fade_frames,
                original_gain=original_gain,
                ducking_gain=ducking_gain,
            )
            original_sample = _sample_at(original_samples, frame_index)
            mixed[frame_index] = _clip_pcm16(round(original_sample * gain))


def _ducking_gain_for_frame(
    frame_index: int,
    *,
    start_frame: int,
    end_frame: int,
    fade_frames: int,
    original_gain: float,
    ducking_gain: float,
) -> float:
    if fade_frames <= 0:
        return ducking_gain

    attack_end = start_frame + fade_frames
    release_start = end_frame - fade_frames

    if frame_index < attack_end:
        progress = (frame_index - start_frame) / fade_frames
        return _lerp(original_gain, ducking_gain, progress)

    if frame_index >= release_start:
        progress = (frame_index - release_start) / fade_frames
        return _lerp(ducking_gain, original_gain, progress)

    return ducking_gain


def _sample_at(samples: array, index: int) -> int:
    if index >= len(samples):
        return 0

    return int(samples[index])


def _validate_gain(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _lerp(start: float, end: float, progress: float) -> float:
    return start + ((end - start) * progress)


def _clip_pcm16(value: int) -> int:
    return max(PCM_MIN, min(PCM_MAX, value))
