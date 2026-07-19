from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from dublaro.schemas import Segment, VoiceProfile


class SpeechSynthesisOptions(BaseModel):
    language: str = Field(min_length=2)
    sample_rate: int = Field(default=24_000, ge=8_000)
    speaker_id: str | None = None
    voice_profile: VoiceProfile | None = None


class TtsAdapter(Protocol):
    name: str

    def synthesize_segment(
        self,
        segment: Segment,
        output_path: Path,
        options: SpeechSynthesisOptions,
    ) -> Path:
        """Generate speech audio for one transcript segment."""
        ...
