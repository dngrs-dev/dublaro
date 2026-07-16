from pydantic import BaseModel, Field

from dublaro.schemas.segment import Segment


class Transcript(BaseModel):
    id: str
    source_language: str
    target_language: str | None = None
    duration: float | None = Field(ge=0, default=None)

    segments: list[Segment] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def sorted_segments(self) -> list[Segment]:
        return sorted(self.segments, key=lambda s: s.start)

    def source_text(self) -> str:
        return " ".join([s.source_text for s in self.sorted_segments()])

    def translated_text(self) -> str:
        return " ".join([s.translated_text for s in self.sorted_segments()])

    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments if s.speaker})
