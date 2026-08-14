from pathlib import Path

import pytest
from dublaro.adapters.asr.base import TranscriptionOptions
from dublaro.pipeline.transcribe import (
    default_transcript_path,
    load_transcript,
    save_transcript,
    transcribe_audio,
)
from dublaro.schemas import Segment, Transcript


class FakeAsrAdapter:
    name = "fake-asr"

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> Transcript:
        return Transcript(
            id=audio_path.stem,
            source_language=options.source_language or "en",
            duration=2.0,
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=2.0,
                    source_text="Hello world",
                    source_language=options.source_language or "en",
                )
            ],
            metadata={"adapter": self.name},
        )


def test_transcribe_audio_uses_adapter(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fake audio")

    transcript = transcribe_audio(
        audio_path,
        adapter=FakeAsrAdapter(),
        options=TranscriptionOptions(source_language="en"),
    )

    assert transcript.id == "voice"
    assert transcript.source_language == "en"
    assert transcript.segments[0].source_text == "Hello world"
    assert transcript.metadata["adapter"] == "fake-asr"


def test_transcribe_audio_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_audio(tmp_path / "missing.wav", adapter=FakeAsrAdapter())


def test_default_transcript_path() -> None:
    assert default_transcript_path("voice.wav") == Path("voice.transcript.json")


def test_save_and_load_transcript(tmp_path: Path) -> None:
    transcript = Transcript(
        id="voice",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                source_text="Hello",
            )
        ],
    )

    output_path = save_transcript(
        transcript,
        tmp_path / "transcripts" / "voice.json",
    )

    loaded = load_transcript(output_path)

    assert output_path.exists()
    assert loaded.id == "voice"
    assert loaded.segments[0].source_text == "Hello"
