from pathlib import Path

import pytest
from dublaro.adapters.tts import FakeTtsAdapter
from dublaro.audio.wav import read_mono_pcm16_wav
from dublaro.pipeline.voice_preview import synthesize_voice_samples
from dublaro.pipeline.voices import SpeakerVoice
from dublaro.schemas import VoiceProfile


def test_synthesize_voice_samples_generates_fallback_sample(tmp_path: Path) -> None:
    samples = synthesize_voice_samples(
        text="Hello",
        output_dir=tmp_path,
        language="pl",
        sample_rate=16000,
        fallback_adapter=FakeTtsAdapter(),
        fallback_tts_backend="fake",
    )

    assert len(samples) == 1
    assert samples[0].speaker_id == "fallback"
    assert samples[0].tts_backend == "fake"
    assert samples[0].output_path == tmp_path / "fallback.wav"
    assert samples[0].output_path.exists()

    sample_rate, audio = read_mono_pcm16_wav(samples[0].output_path)

    assert sample_rate == 16000
    assert len(audio) > 0


def test_synthesize_voice_samples_generates_one_sample_per_speaker(
    tmp_path: Path,
) -> None:
    speaker_voices = {
        "SPEAKER_01": SpeakerVoice(
            profile=VoiceProfile(
                speaker_id="SPEAKER_01",
                display_name="Guest",
                language="pl",
                tts_backend="fake",
            ),
            adapter=FakeTtsAdapter(),
        ),
        "SPEAKER_00": SpeakerVoice(
            profile=VoiceProfile(
                speaker_id="SPEAKER_00",
                display_name="Host",
                language="pl",
                tts_backend="fake",
            ),
            adapter=FakeTtsAdapter(),
        ),
    }

    samples = synthesize_voice_samples(
        text="Hello",
        output_dir=tmp_path,
        language="pl",
        sample_rate=24000,
        speaker_voices=speaker_voices,
    )

    assert [sample.speaker_id for sample in samples] == ["SPEAKER_00", "SPEAKER_01"]
    assert (tmp_path / "SPEAKER_00.wav").exists()
    assert (tmp_path / "SPEAKER_01.wav").exists()


def test_synthesize_voice_samples_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Preview text cannot be empty"):
        synthesize_voice_samples(
            text=" ",
            output_dir=tmp_path,
            language="pl",
            sample_rate=24000,
            fallback_adapter=FakeTtsAdapter(),
            fallback_tts_backend="fake",
        )
