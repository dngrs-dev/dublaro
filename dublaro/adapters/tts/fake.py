import wave
from pathlib import Path

from dublaro.adapters.tts.base import SpeechSynthesisOptions
from dublaro.schemas import Segment


class FakeTtsAdapter:
    name = "fake-tts"

    def synthesize_segment(
        self,
        segment: Segment,
        output_path: Path,
        options: SpeechSynthesisOptions,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration_seconds = max(segment.duration, 0.1)
        frame_count = int(duration_seconds * options.sample_rate)

        with wave.open(str(output_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(options.sample_rate)
            audio_file.writeframes(b"\x00\x00" * frame_count)

        return output_path
