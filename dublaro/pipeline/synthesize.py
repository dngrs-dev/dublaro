from collections.abc import Mapping
from pathlib import Path

from dublaro.adapters.tts.base import SpeechSynthesisOptions, TtsAdapter
from dublaro.pipeline.voices import SpeakerVoice
from dublaro.schemas import Transcript


def synthesize_transcript_speech(
    transcript: Transcript,
    adapter: TtsAdapter,
    *,
    output_dir: str | Path,
    language: str | None = None,
    sample_rate: int = 24_000,
    speaker_voices: Mapping[str, SpeakerVoice] | None = None,
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
        "tts_speaker_voice_count": str(len(speaker_voices or {})),
    }

    for segment in synthesized.segments:
        spoken_text = (
            segment.adapted_text or segment.translated_text or segment.source_text
        )
        if not spoken_text.strip():
            continue

        speaker_voice = None
        if speaker_voices is not None and segment.speaker is not None:
            speaker_voice = speaker_voices.get(segment.speaker)

        segment_adapter = adapter
        voice_profile = None
        segment_language = resolved_language

        if speaker_voice is not None:
            segment_adapter = speaker_voice.adapter
            voice_profile = speaker_voice.profile
            if voice_profile.language is not None:
                segment_language = voice_profile.language

        output_path = speech_dir / _segment_audio_filename(segment.id)

        options = SpeechSynthesisOptions(
            language=segment_language,
            sample_rate=sample_rate,
            speaker_id=segment.speaker,
            voice_profile=voice_profile,
        )

        generated_path = segment_adapter.synthesize_segment(
            segment, output_path, options
        )
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
