from pathlib import Path

from dublaro.adapters.source_separation import (
    SourceSeparationAdapter,
    SourceSeparationOptions,
    SourceSeparationResult,
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
