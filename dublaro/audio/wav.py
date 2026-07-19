import sys
import wave
from array import array
from pathlib import Path

PCM_MIN = -32768
PCM_MAX = 32767


def read_mono_pcm16_wav(path: str | Path) -> tuple[int, array]:
    audio_path = Path(path)

    with wave.open(str(audio_path), "rb") as audio_file:
        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        sample_rate = audio_file.getframerate()
        frames = audio_file.readframes(audio_file.getnframes())

    if channels != 1:
        raise ValueError(f"Expected mono WAV file: {audio_path}")

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV file: {audio_path}")

    samples = array("h")
    samples.frombytes(frames)

    if sys.byteorder != "little":
        samples.byteswap()

    return sample_rate, samples


def write_mono_pcm16_wav(
    path: str | Path,
    samples: array,
    *,
    sample_rate: int,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_samples = array("h", samples)

    if sys.byteorder != "little":
        output_samples.byteswap()

    with wave.open(str(output_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(output_samples.tobytes())

    return output_path


def mix_mono_pcm16_at(
    timeline: array,
    clip: array,
    *,
    start_frame: int,
) -> None:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    for index, sample in enumerate(clip):
        timeline_index = start_frame + index

        if timeline_index >= len(timeline):
            break

        timeline[timeline_index] = _clip_pcm16(
            int(timeline[timeline_index]) + int(sample)
        )


def create_silence(frame_count: int) -> array:
    if frame_count < 0:
        raise ValueError("frame_count must be >= 0")

    return array("h", [0]) * frame_count


def _clip_pcm16(value: int) -> int:
    return max(PCM_MIN, min(PCM_MAX, value))
