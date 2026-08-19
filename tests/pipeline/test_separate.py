from pathlib import Path

from dublaro.adapters.source_separation import (
    SourceSeparationOptions,
    SourceSeparationResult,
)
from dublaro.pipeline.separate import (
    default_source_separation_paths,
    separate_background_audio,
)


class RecordingSourceSeparationAdapter:
    name = "recording-source-separation"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def separate_sources(
        self,
        audio_path: str | Path,
        *,
        background_output_path: str | Path,
        voice_output_path: str | Path,
        options: SourceSeparationOptions,
        overwrite: bool = False,
    ) -> SourceSeparationResult:
        background_path = Path(background_output_path)
        voice_path = Path(voice_output_path)

        self.calls.append(
            {
                "audio_path": Path(audio_path),
                "background_output_path": background_path,
                "voice_output_path": voice_path,
                "sample_rate": options.sample_rate,
                "overwrite": overwrite,
            }
        )

        background_path.parent.mkdir(parents=True, exist_ok=True)
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        background_path.write_bytes(b"background")
        voice_path.write_bytes(b"voice")

        return SourceSeparationResult(
            background_audio_path=background_path,
            voice_audio_path=voice_path,
        )


def test_default_source_separation_paths_use_input_directory(tmp_path: Path) -> None:
    audio_path = tmp_path / "zoo.wav"

    paths = default_source_separation_paths(audio_path)

    assert paths.background_audio_path == tmp_path / "zoo.background.wav"
    assert paths.voice_audio_path == tmp_path / "zoo.voice.wav"


def test_default_source_separation_paths_can_use_output_directory(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "zoo.wav"
    output_dir = tmp_path / "separated"

    paths = default_source_separation_paths(audio_path, output_dir=output_dir)

    assert paths.background_audio_path == output_dir / "zoo.background.wav"
    assert paths.voice_audio_path == output_dir / "zoo.voice.wav"


def test_separate_background_audio_calls_adapter(tmp_path: Path) -> None:
    audio_path = tmp_path / "zoo.wav"
    background_path = tmp_path / "out" / "zoo.background.wav"
    voice_path = tmp_path / "out" / "zoo.voice.wav"

    audio_path.write_bytes(b"audio")

    adapter = RecordingSourceSeparationAdapter()

    result = separate_background_audio(
        audio_path,
        adapter=adapter,
        background_output_path=background_path,
        voice_output_path=voice_path,
        sample_rate=16_000,
        overwrite=True,
    )

    assert result.background_audio_path == background_path
    assert result.voice_audio_path == voice_path
    assert background_path.read_bytes() == b"background"
    assert voice_path.read_bytes() == b"voice"
    assert adapter.calls == [
        {
            "audio_path": audio_path,
            "background_output_path": background_path,
            "voice_output_path": voice_path,
            "sample_rate": 16_000,
            "overwrite": True,
        }
    ]
