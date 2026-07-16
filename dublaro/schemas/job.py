from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelSelection(BaseModel):
    asr: str | None = None
    diarization: str | None = None
    translation: str | None = None
    text_adapter: str | None = None
    tts: str | None = None


class DubbingJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    input_path: str
    workspace_dir: str
    output_path: str | None = None

    source_language: str | None = None
    target_language: str

    status: JobStatus = JobStatus.QUEUED
    models: ModelSelection = Field(default_factory=ModelSelection)

    transcript_path: str | None = None
    dubbed_audio_path: str | None = None
    final_video_path: str | None = None

    error: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, str] = Field(default_factory=dict)
