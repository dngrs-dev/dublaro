from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, model_validator


class DiarizationTurn(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str
    confidence: float | None = Field(ge=0, le=1, default=None)

    @model_validator(mode="after")
    def validate_time_order(self) -> "DiarizationTurn":
        if self.end < self.start:
            raise ValueError("diarization turn end must be >= start")
        return self


class DiarizationOptions(BaseModel):
    min_speakers: int | None = Field(ge=1, default=None)
    max_speakers: int | None = Field(ge=1, default=None)

    @model_validator(mode="after")
    def validate_speaker_range(self) -> "DiarizationOptions":
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot be greater than max_speakers")
        return self


class DiarizationAdapter(Protocol):
    name: str

    def diarize(
        self,
        audio_path: Path,
        options: DiarizationOptions,
    ) -> list[DiarizationTurn]: ...
