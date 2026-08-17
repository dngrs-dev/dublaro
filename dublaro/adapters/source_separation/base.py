from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class SourceSeparationOptions(BaseModel):
    sample_rate: int = Field(gt=0)


class SourceSeparationResult(BaseModel):
    background_audio_path: Path
    voice_audio_path: Path | None = None


class SourceSeparationAdapter(Protocol):
    name: str

    def separate_sources(
        self,
        audio_path: str | Path,
        *,
        background_output_path: str | Path,
        voice_output_path: str | Path,
        options: SourceSeparationOptions,
        overwrite: bool = False,
    ) -> SourceSeparationResult:
        """Separate original audio into background and voice/dialogue stems."""
        ...
