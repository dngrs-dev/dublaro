from pathlib import Path

from dublaro.adapters.asr.base import TranscriptionOptions
from dublaro.schemas import Segment, Transcript


class FakeAsrAdapter:
    name = "fake-asr"

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> Transcript:
        language = options.source_language or "unknown"

        return Transcript(
            id=audio_path.stem,
            source_language=language,
            duration=None,
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="This is a placeholder transcript.",
                    source_language=language,
                )
            ],
            metadata={"adapter": self.name},
        )
