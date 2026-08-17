from pathlib import Path

import pytest
from dublaro.adapters.source_separation import (
    FakeSourceSeparationAdapter,
    SourceSeparationOptions,
)


def test_fake_source_separation_copies_audio_to_background_and_voice(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "source.wav"
    background_path = tmp_path / "background.wav"
    voice_path = tmp_path / "voice.wav"

    audio_path.write_bytes(b"audio")

    result = FakeSourceSeparationAdapter().separate_sources(
        audio_path,
        background_output_path=background_path,
        voice_output_path=voice_path,
        options=SourceSeparationOptions(sample_rate=16_000),
        overwrite=False,
    )

    assert result.background_audio_path == background_path
    assert result.voice_audio_path == voice_path
    assert background_path.read_bytes() == b"audio"
    assert voice_path.read_bytes() == b"audio"


def test_fake_source_separation_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "source.wav"
    background_path = tmp_path / "background.wav"
    voice_path = tmp_path / "voice.wav"

    audio_path.write_bytes(b"audio")
    background_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Output already exists"):
        FakeSourceSeparationAdapter().separate_sources(
            audio_path,
            background_output_path=background_path,
            voice_output_path=voice_path,
            options=SourceSeparationOptions(sample_rate=16_000),
            overwrite=False,
        )
