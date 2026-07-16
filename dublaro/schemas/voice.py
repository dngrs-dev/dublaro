from enum import StrEnum

from pydantic import BaseModel, Field


class ConsentStatus(StrEnum):
    UNKNOWN = "unknown"
    GRANTED = "granted"
    REVOKED = "revoked"


class VoiceProfile(BaseModel):
    speaker_id: str
    display_name: str | None = None

    language: str | None = None
    tts_backend: str | None = None

    reference_audio_paths: list[str] = Field(default_factory=list)
    generated_smaple_path: str | None = None

    consent_status: ConsentStatus = ConsentStatus.UNKNOWN
    consent_note: str | None = None

    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def can_clone(self) -> bool:
        return self.consent_status == ConsentStatus.GRANTED
