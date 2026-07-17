from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from dublaro.schemas.transcript import Transcript


class TranscriptionOptions(BaseModel):
    source_language: str | None = None
    beam_size: int = Field(default=5, ge=1)
    word_timestamps: bool = True


class AsrAdapter(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> Transcript:
        """Convert an audio file into a timestamped transcript."""
        ...
