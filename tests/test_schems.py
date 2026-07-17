import pytest
from dublaro.schemas import (
    DubbingJob,
    JobStatus,
    ModelSelection,
    Segment,
    Transcript,
    VoiceProfile,
)
from dublaro.schemas.voice import ConsentStatus
from pydantic import ValidationError


def test_segment_exposes_duration() -> None:
    segment = Segment(
        id="seg-1",
        start=1.5,
        end=4.0,
        speaker="speaker-1",
        source_text="Hello world",
    )

    assert segment.duration == 2.5


def test_segment_rejects_invalid_time_order() -> None:
    with pytest.raises(ValidationError):
        Segment(id="seg-1", start=4.0, end=1.5)


def test_transcript_sorts_segments_and_collects_speakers() -> None:
    transcript = Transcript(
        id="transcript-1",
        source_language="en",
        segments=[
            Segment(id="seg-2", start=3.0, end=4.0, speaker="speaker-2"),
            Segment(id="seg-1", start=1.0, end=2.0, speaker="speaker-1"),
        ],
    )

    assert [segment.id for segment in transcript.sorted_segments()] == [
        "seg-1",
        "seg-2",
    ]
    assert transcript.speakers() == ["speaker-1", "speaker-2"]


def test_dubbing_job_defaults_to_queued_status() -> None:
    job = DubbingJob(
        input_path="input.mp4",
        workspace_dir=".dublaro/jobs/job-1",
        target_language="pl",
    )

    assert job.status == JobStatus.QUEUED
    assert job.models.asr is None
    assert job.models.diarization is None
    assert job.models.translation is None
    assert job.models.tts is None


def test_voice_profile_requires_consent_for_cloning() -> None:
    unknown_voice = VoiceProfile(speaker_id="speaker-1")
    granted_voice = VoiceProfile(
        speaker_id="speaker-1",
        consent_status=ConsentStatus.GRANTED,
    )

    assert unknown_voice.can_clone is False
    assert granted_voice.can_clone is True


def test_dubbing_job_accepts_explicit_model_selection() -> None:
    job = DubbingJob(
        input_path="input.mp4",
        workspace_dir=".dublaro/jobs/job-1",
        target_language="pl",
        models=ModelSelection(asr="faster-whisper"),
    )

    assert job.models.asr == "faster-whisper"
