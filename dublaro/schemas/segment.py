from pydantic import BaseModel, Field, model_validator


class WordTiming(BaseModel):
    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    confidence: float | None = Field(ge=0, le=1, default=None)

    @model_validator(mode="after")
    def validate_word_time_order(self):
        if self.end < self.start:
            raise ValueError("word end must be >= start")
        return self


class Segment(BaseModel):
    id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str | None = None

    source_text: str = ""
    translated_text: str = ""
    adapted_text: str = ""

    source_language: str | None = None
    target_language: str | None = None

    words: list[WordTiming] = Field(default_factory=list)
    confidence: float | None = Field(ge=0, le=1, default=None)

    metadata: dict[str, str] = Field(default_factory=dict)
    generated_audio_path: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @model_validator(mode="after")
    def validate_time_order(self):
        if self.end < self.start:
            raise ValueError("segment end must be >= start")
        return self
