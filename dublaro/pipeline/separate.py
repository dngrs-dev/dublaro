from dataclasses import dataclass
from pathlib import Path

from dublaro.adapters.source_separation import (
    SourceSeparationAdapter,
    SourceSeparationOptions,
    SourceSeparationResult,
)


@dataclass(frozen=True)
class SourceSeparationPaths:
    background_audio_path: Path
    voice_audio_path: Path


def default_source_separation_paths(
    audio_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> SourceSeparationPaths:
    audio_file = Path(audio_path)
    output_directory = Path(output_dir) if output_dir is not None else audio_file.parent

    return SourceSeparationPaths(
        background_audio_path=output_directory / f"{audio_file.stem}.background.wav",
        voice_audio_path=output_directory / f"{audio_file.stem}.voice.wav",
    )


def separate_background_audio(
    audio_path: str | Path,
    *,
    adapter: SourceSeparationAdapter,
    background_output_path: str | Path,
    voice_output_path: str | Path,
    sample_rate: int,
    overwrite: bool = False,
) -> SourceSeparationResult:
    return adapter.separate_sources(
        audio_path,
        background_output_path=background_output_path,
        voice_output_path=voice_output_path,
        options=SourceSeparationOptions(sample_rate=sample_rate),
        overwrite=overwrite,
    )
