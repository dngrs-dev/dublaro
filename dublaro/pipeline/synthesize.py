from pathlib import Path

from dublaro.adapters.tts.base import SpeechSynthesisOptions, TtsAdapter
from dublaro.schemas import Transcript, VoiceProfile


def synthesize_transcript_speech(
    transcript: Transcript,
    adapter: TtsAdapter,
    *,
    output_dir: str | Path,
    language: str | None = None,
    sample_rate: int = 24_000,
    voice_profiles: dict[str, VoiceProfile] | None = None,
) -> Transcript:
    resolved_language = (
        language or transcript.target_language or transcript.source_language
    )
    speech_dir = Path(output_dir)
    speech_dir.mkdir(parents=True, exist_ok=True)

    synthesized = transcript.model_copy(deep=True)
    synthesized.metadata = {
        **synthesized.metadata,
        "tts_adapter": adapter.name,
        "tts_language": resolved_language,
        "tts_sample_rate": str(sample_rate),
    }

    for segment in synthesized.segments:
        spoken_text = (
            segment.adapted_text or segment.translated_text or segment.source_text
        )

        if not spoken_text.strip():
            continue

        voice_profile = None
        if voice_profiles is not None and segment.speaker is not None:
            voice_profile = voice_profiles.get(segment.speaker)

        output_path = speech_dir / _segment_audio_filename(segment.id)

        options = SpeechSynthesisOptions(
            language=resolved_language,
            sample_rate=sample_rate,
            speaker_id=segment.speaker,
            voice_profile=voice_profile,
        )

        generated_path = adapter.synthesize_segment(segment, output_path, options)
        segment.generated_audio_path = str(generated_path)

    return synthesized


def default_speech_output_dir(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(f"{transcript_file.stem}.speech")


def default_synthesized_transcript_path(transcript_path: str | Path) -> Path:
    transcript_file = Path(transcript_path)
    return transcript_file.with_name(
        f"{transcript_file.stem}.synthesized{transcript_file.suffix}"
    )


def _segment_audio_filename(segment_id: str) -> str:
    safe_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in segment_id
    ).strip("_")

    return f"{safe_id or 'segment'}.wav"
