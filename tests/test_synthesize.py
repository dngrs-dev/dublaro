import wave
from pathlib import Path

from dublaro.adapters.tts import FakeTtsAdapter, SpeechSynthesisOptions
from dublaro.pipeline.synthesize import (
    default_speech_output_dir,
    default_synthesized_transcript_path,
    synthesize_transcript_speech,
)
from dublaro.pipeline.voices import SpeakerVoice
from dublaro.schemas import Segment, Transcript, VoiceProfile


class RecordingTtsAdapter:
    def __init__(self, name: str, calls: list[tuple[str, str | None]]) -> None:
        self.name = name
        self.calls = calls

    def synthesize_segment(
        self,
        segment: Segment,
        output_path: Path,
        options: SpeechSynthesisOptions,
    ) -> Path:
        self.calls.append((self.name, options.speaker_id))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(options.sample_rate)
            audio_file.writeframes(b"\x00\x00")

        return output_path


def test_synthesize_transcript_speech_generates_audio_files(tmp_path: Path) -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                speaker="speaker-1",
                source_text="Hello world",
                translated_text="Cześć świecie",
                adapted_text="Cześć świecie",
            )
        ],
    )

    synthesized = synthesize_transcript_speech(
        transcript,
        adapter=FakeTtsAdapter(),
        output_dir=tmp_path / "speech",
        sample_rate=16_000,
    )

    generated_path = Path(synthesized.segments[0].generated_audio_path or "")

    assert generated_path.exists()
    assert synthesized.metadata["tts_adapter"] == "fake-tts"
    assert synthesized.metadata["tts_language"] == "pl"
    assert synthesized.metadata["tts_sample_rate"] == "16000"

    with wave.open(str(generated_path), "rb") as audio_file:
        assert audio_file.getnchannels() == 1
        assert audio_file.getframerate() == 16_000
        assert audio_file.getnframes() == 16_000


def test_synthesize_transcript_speech_does_not_mutate_original(tmp_path: Path) -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                adapted_text="Cześć",
            )
        ],
    )

    synthesized = synthesize_transcript_speech(
        transcript,
        adapter=FakeTtsAdapter(),
        output_dir=tmp_path / "speech",
    )

    assert transcript.segments[0].generated_audio_path is None
    assert synthesized.segments[0].generated_audio_path is not None


def test_synthesize_transcript_speech_skips_empty_segments(tmp_path: Path) -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
            )
        ],
    )

    synthesized = synthesize_transcript_speech(
        transcript,
        adapter=FakeTtsAdapter(),
        output_dir=tmp_path / "speech",
    )

    assert synthesized.segments[0].generated_audio_path is None
    assert list((tmp_path / "speech").iterdir()) == []


def test_default_synthesized_transcript_path() -> None:
    assert default_synthesized_transcript_path("lesson.pl.adapted.json") == Path(
        "lesson.pl.adapted.synthesized.json"
    )


def test_default_speech_output_dir() -> None:
    assert default_speech_output_dir("lesson.pl.adapted.json") == Path(
        "lesson.pl.adapted.speech"
    )


def test_synthesize_uses_speaker_voice_adapters(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []

    transcript = Transcript(
        id="lesson",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(id="seg-1", start=0, end=1, speaker="speaker-1", adapted_text="A"),
            Segment(id="seg-2", start=1, end=2, speaker="speaker-2", adapted_text="B"),
            Segment(id="seg-3", start=2, end=3, adapted_text="C"),
        ],
    )

    result = synthesize_transcript_speech(
        transcript,
        adapter=RecordingTtsAdapter("fallback", calls),
        output_dir=tmp_path / "speech",
        sample_rate=16_000,
        speaker_voices={
            "speaker-1": SpeakerVoice(
                profile=VoiceProfile(speaker_id="speaker-1", tts_backend="fake"),
                adapter=RecordingTtsAdapter("voice-1", calls),
            ),
            "speaker-2": SpeakerVoice(
                profile=VoiceProfile(speaker_id="speaker-2", tts_backend="fake"),
                adapter=RecordingTtsAdapter("voice-2", calls),
            ),
        },
    )

    assert calls == [
        ("voice-1", "speaker-1"),
        ("voice-2", "speaker-2"),
        ("fallback", None),
    ]
    assert result.metadata["tts_speaker_voice_count"] == "2"
