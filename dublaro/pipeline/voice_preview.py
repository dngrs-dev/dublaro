from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dublaro.adapters.tts import SpeechSynthesisOptions, TtsAdapter
from dublaro.pipeline.voices import SpeakerVoice
from dublaro.schemas import Segment, VoiceProfile


@dataclass(frozen=True)
class VoiceSample:
    speaker_id: str
    display_name: str | None
    tts_backend: str
    output_path: Path


def synthesize_voice_samples(
    *,
    text: str,
    output_dir: str | Path,
    language: str,
    sample_rate: int,
    speaker_voices: Mapping[str, SpeakerVoice] | None = None,
    fallback_adapter: TtsAdapter | None = None,
    fallback_tts_backend: str = "unknown",
    fallback_speaker_id: str = "fallback",
) -> list[VoiceSample]:
    preview_text = text.strip()
    if not preview_text:
        raise ValueError("Preview text cannot be empty.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    active_voices = dict(speaker_voices or {})
    if not active_voices:
        if fallback_adapter is None:
            raise ValueError(
                "fallback_adapter is required when no speaker voices are configured."
            )

        active_voices[fallback_speaker_id] = SpeakerVoice(
            VoiceProfile(
                speaker_id=fallback_speaker_id,
                display_name="Fallback",
                language=language,
                tts_backend=fallback_tts_backend,
            ),
            fallback_adapter,
        )

    samples: list[VoiceSample] = []

    for speaker_id, speaker_voice in sorted(active_voices.items()):
        profile = speaker_voice.profile
        adapter = speaker_voice.adapter
        sample_path = output_path / f"{_safe_sample_stem(speaker_id)}.wav"
        sample_language = profile.language or language

        segment = Segment(
            id=f"preview-{_safe_sample_stem(speaker_id)}",
            start=0.0,
            end=1.0,
            speaker=speaker_id,
            adapted_text=preview_text,
            target_language=sample_language,
        )

        adapter.synthesize_segment(
            segment,
            sample_path,
            options=SpeechSynthesisOptions(
                language=sample_language,
                sample_rate=sample_rate,
                speaker_id=speaker_id,
                voice_profile=profile,
            ),
        )

        samples.append(
            VoiceSample(
                speaker_id=speaker_id,
                display_name=profile.display_name,
                tts_backend=profile.tts_backend or adapter.name,
                output_path=sample_path,
            )
        )

    return samples


def _safe_sample_stem(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )
    return safe or "voice"
